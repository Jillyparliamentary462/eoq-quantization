# Pesquisa: Kernels CUDA do llama.cpp para Quantizacao Q4

## Objetivo

Entender exatamente como os kernels CUDA INT4 mais rapidos existentes funcionam
(llama.cpp MMVQ e MMQ) para podermos implementar algo similar ou melhor.

---

## 1. Arquitetura Geral dos Kernels CUDA do llama.cpp

O llama.cpp possui **dois caminhos principais** para multiplicacao matricial com pesos quantizados:

### MMVQ (Matrix-Vector Quantized)
- Otimizado para **batch size 1** (geracao de tokens, decodificacao)
- Usa ponteiros de funcao `vec_dot_q_cuda` especializados por tipo de quantizacao
- Cada thread computa dot products parciais, com reducao via warp shuffle
- **Memory-bound**: o gargalo eh a largura de banda de memoria, nao computacao

### MMQ (Matrix-Matrix Quantized)
- Otimizado para **batches maiores** (prompt processing)
- Ativacoes sao convertidas para Q8_1 on-the-fly via `quantize_mmq_q8_1_cuda`
- Usa tiles em shared memory com layout transposto para evitar bank conflicts
- Suporta dp4a e MMA (tensor cores) dependendo da arquitetura

### Selecao Automatica
```
ggml_cuda_should_use_mmq() determina:
  - MMQ: batches pequenos em GPUs Volta+
  - cuBLAS: batches grandes ou quando nao ha vantagem
  - MMVQ: batch size 1 (token generation)
```

---

## 2. Estruturas de Dados Fundamentais

### block_q4_0 (QK4_0 = 32 pesos por bloco)
```c
#define QK4_0 32
typedef struct {
    ggml_half d;            // fator de escala (16-bit float)
    uint8_t qs[QK4_0 / 2]; // 16 bytes = 32 nibbles de 4 bits
} block_q4_0;
// Total: 18 bytes para 32 pesos = 4.5 bits/peso
```

### block_q4_K (QK_K = 256 pesos por super-bloco)
```c
#define QK_K 256
#define K_SCALE_SIZE 12
typedef struct {
    union {
        struct {
            ggml_half d;    // escala do super-bloco para escalas quantizadas
            ggml_half dmin; // escala do super-bloco para minimos quantizados
        };
        ggml_half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE]; // escalas e minimos, quantizados com 6 bits
    uint8_t qs[QK_K/2];          // 128 bytes = 256 quants de 4 bits
} block_q4_K;
// Total: 2 + 2 + 12 + 128 = 144 bytes para 256 pesos = 4.5 bits/peso
```

### block_q8_1 (formato intermediario para ativacoes)
```c
#define QK8_1 32
typedef struct {
    union {
        struct {
            ggml_half d; // delta (escala)
            ggml_half s; // d * sum(qs[i]) -- soma pre-computada
        };
        ggml_half2 ds;
    };
    int8_t qs[QK8_1]; // 32 quants de 8 bits
} block_q8_1;
```

**Nota crucial**: `block_q8_1` armazena a soma `s = d * sum(qs)` pre-computada.
Isso eh usado para subtrair o offset de 8 do Q4_0 sem loop extra:
`resultado = d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y)`

---

## 3. Constantes Fundamentais do Sistema de Quantizacao

```c
// Q4_0: bloco de 32 pesos
#define QK4_0 32
#define QR4_0 2    // ratio: 2 valores Q4 empacotados por byte (antes de deq.)
#define QI4_0 (QK4_0 / (4 * QR4_0))  // = 4 inteiros de 32 bits por bloco

// Q4_K: super-bloco de 256 pesos
#define QK_K  256
#define QR4_K 2
#define QI4_K (QK_K / (4 * QR4_K))   // = 32 inteiros de 32 bits por bloco

// Q8_1: bloco de 32 pesos (ativacoes)
#define QK8_1 32
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))  // = 8 inteiros de 32 bits por bloco

// VDR = Vector Dot Ratio (quantos int32 cada thread processa)
#define VDR_Q4_0_Q8_1_MMVQ 2   // MMVQ: 2 int32 por iteracao
#define VDR_Q4_0_Q8_1_MMQ  4   // MMQ:  4 int32 por iteracao
#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q4_K_Q8_1_MMQ  8

#define WARP_SIZE 32
#define MATRIX_ROW_PADDING 512  // alinhamento para acesso coalescido
```

**Significado dos QI/QR**:
- QR = "quant ratio" -- quantos sub-valores de 4 bits estao empacotados por posicao
- QI = "quant int32s" -- quantos int32 representam um bloco completo
- QI4_0 = 4 significa que 32 pesos Q4_0 cabem em 4 inteiros de 32 bits (8 nibbles cada)
- QI4_K = 32 significa que 256 pesos Q4_K precisam de 32 inteiros de 32 bits

---

## 4. Funcao dequantize_q4_0 (Standalone)

```cuda
static __device__ __forceinline__ void dequantize_q4_0(
    const void * vx, const int64_t ib, const int iqs, float2 & v) {

    const block_q4_0 * x = (const block_q4_0 *) vx;
    const float d = x[ib].d;
    const int vui = x[ib].qs[iqs];

    // Desempacota 2 nibbles de 4 bits de um unico byte
    v.x = vui & 0xF;        // nibble inferior (bits 0-3)
    v.y = vui >> 4;          // nibble superior (bits 4-7)

    // Subtrai bias de 8 e multiplica pela escala
    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}
```

**Como funciona**:
1. Cada byte `qs[iqs]` contem 2 pesos quantizados de 4 bits
2. Mascara `& 0xF` extrai o nibble inferior (0-15)
3. Shift `>> 4` extrai o nibble superior (0-15)
4. Subtrai 8 para obter faixa [-8, +7]
5. Multiplica pela escala `d` (compartilhada por 32 pesos)

---

## 5. Kernel MMVQ: mul_mat_vec_q (O Kernel Principal para Geracao)

### Template e Launch Configuration

```cuda
template <ggml_type type, int ncols_dst, bool has_fusion,
          bool is_multi_token_id = false, bool small_k = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, table_id) *
                  ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(...)
```

### Dimensoes de Thread Block

```cuda
// Para NVIDIA (Generic):
static constexpr int calc_nwarps(type, ncols_dst, table_id) {
    // ncols_dst=1 (batch 1): 4 warps = 128 threads
    // ncols_dst=2-4:          2 warps = 64 threads
    // ncols_dst=5-8:          2 warps = 64 threads
    // ncols_dst>8:            1 warp  = 32 threads
}

static constexpr int calc_rows_per_block(ncols_dst, table_id, small_k, nwarps) {
    // ncols_dst=1: small_k ? nwarps : 1  (1-4 linhas por bloco)
    // ncols_dst=2-8: 2 linhas por bloco
    // ncols_dst>8:   1 linha por bloco
}

// Configuracao de lancamento:
const dim3 block_dims(warp_size, nwarps, 1);  // ex: (32, 4, 1) = 128 threads
const dim3 grid((nrows_x + rpb - 1) / rpb, nchannels_dst, nsamples);
```

**Para batch size 1 (caso mais comum de TG)**:
- **128 threads** (4 warps de 32)
- **1 linha por bloco** (cada bloco CUDA processa 1 linha da matriz de pesos)
- Grid: `nrows_x` blocos na dimensao X

### Shared Memory

```cuda
// Shared memory para reducao inter-warp:
__shared__ float tmp_shared[nwarps-1][ncols_dst][rows_per_cuda_block][warp_size];
// Para batch=1, nwarps=4: 3 * 1 * 1 * 32 * 4 = 384 bytes

// Se fusion habilitada, shared memory adicional para gate:
__shared__ float tmp_shared_gate[...];
```

**Nota**: Shared memory eh usada APENAS para reducao entre warps, NAO para
cachear o vetor de entrada. O vetor de entrada eh lido diretamente da
global memory (mas fica cacheado em L2 automaticamente).

### Loop Principal (Dot Product Fusionado)

```cuda
// Cada thread itera sobre blocos quantizados da linha
const int blocks_per_iter = vdr * nwarps * warp_size / qi;
// Para Q4_0: vdr=2, nwarps=4, warp_size=32, qi=4
// blocks_per_iter = 2 * 4 * 32 / 4 = 64 blocos por iteracao

float tmp[ncols_dst][rows_per_cuda_block] = {0};

for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    const int kqs = vdr * (tid % (qi/vdr));
    // Chama vec_dot especializada para o tipo Q4
    for (int j = 0; j < ncols_dst; ++j) {
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp[j][i] += vec_dot_q_cuda(vx, &y[...], kbx_offset, kqs);
        }
    }
}
```

### Reducao

```cuda
// 1. Reducao intra-warp via shuffle
tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);

// 2. Warps nao-lideres escrevem em shared memory
if (threadIdx.y > 0) {
    tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
    return;  // sai do kernel
}

// 3. Warp lider agrega contribuicoes dos outros warps
__syncthreads();
for (int w = 0; w < nwarps - 1; ++w) {
    tmp[j][i] += tmp_shared[w][j][i][threadIdx.x];
}

// 4. Segunda reducao intra-warp
tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);

// 5. Thread 0 escreve resultado final
if (threadIdx.x == 0) {
    dst[j * stride + row] = tmp[j][0];
}
```

### warp_reduce_sum

```cuda
template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}
// Para width=32: 5 iteracoes (16, 8, 4, 2, 1)
```

---

## 6. vec_dot_q4_0_q8_1: O Dot Product Fusionado para Q4_0

### Funcao de Interface

```cuda
static __device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const void * __restrict__ vbq,
    const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {

    const block_q4_0 * bq4_0 = (const block_q4_0 *) vbq + kbx;

    int v[VDR_Q4_0_Q8_1_MMVQ];        // v[2] -- pesos Q4
    int u[2*VDR_Q4_0_Q8_1_MMVQ];      // u[4] -- ativacoes Q8

    #pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        // Carrega 4 pesos Q4 empacotados em um int32 (via 2x uint16)
        v[i]     = get_int_b2(bq4_0->qs, iqs + i);
        // Carrega 4 ativacoes Q8 empacotadas em um int32 (direto)
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI4_0);
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}
```

### Implementacao Core com DP4A

```cuda
template <int vdr>
static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
    const int * v, const int * u, const float & d4, const half2 & ds8) {

    int sumi = 0;

    #pragma unroll
    for (int i = 0; i < vdr; ++i) {  // vdr=2
        // Extrai nibbles inferiores (4 pesos de 4 bits -> 4 bytes)
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        // Extrai nibbles superiores (4 pesos de 4 bits -> 4 bytes)
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // DP4A: dot product de 4 bytes com acumulacao int32
        // Processa 4 pares peso*ativacao por instrucao!
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);  // nibbles baixos
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);  // nibbles altos
    }
    // Total por chamada: 2 * 2 * 4 = 16 multiplicacoes-acumulacoes

    const float2 ds8f = __half22float2(ds8);

    // Aplica escalas e corrige bias:
    // d4 = escala Q4, ds8f.x = escala Q8, ds8f.y = d8 * sum(q8)
    // O termo "- 8 * vdr/QI4_0 * ds8f.y" corrige o bias de +8 do Q4_0
    return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);
}
```

### Helpers de Carga

```cuda
// Carrega 32 bits a partir de dados empacotados em 16 bits (Q4)
static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;
    return x32;
}

// Carrega 32 bits diretamente (Q8)
static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32];
}
```

### Intrinseco DP4A

```cuda
static __device__ __forceinline__ int ggml_cuda_dp4a(const int a,
    const int b, int c) {
#if defined(GGML_USE_HIP)
    // Branches para AMD: __builtin_amdgcn_sdot4, sudot4, etc.
#else
    return __dp4a(a, b, c);  // NVIDIA: instrucao nativa sm_61+
#endif
}
```

**O que `__dp4a(a, b, c)` faz**:
```
c += a.byte0 * b.byte0 + a.byte1 * b.byte1 + a.byte2 * b.byte2 + a.byte3 * b.byte3
```
Dot product de 4 pares de int8 acumulado em int32, em **uma unica instrucao**.

---

## 7. vec_dot_q4_K_q8_1: O Dot Product Fusionado para Q4_K

### Funcao Completa

```cuda
static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq,
    const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {

    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;

    int    v[2];
    int    u[2*QR4_K];     // u[4]
    float  d8[QR4_K];      // d8[2]

    // Offset para o sub-bloco Q8_1 correspondente
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));

    // Carrega 8 pesos Q4 empacotados em 2 int32
    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    // Desempacota escalas de 6 bits do formato comprimido
    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;  // escalas
    const uint8_t * m  = sc + 2;                 // minimos

    // Carrega ativacoes Q8_1 correspondentes
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}
```

### Implementacao Core Q4_K

```cuda
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v,
    const int * __restrict__ u,
    const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m,
    const half2 & dm4,
    const float * __restrict__ d8) {

    float sumf_d = 0.0f;  // acumulador para termo de escala
    float sumf_m = 0.0f;  // acumulador para termo de minimo

    #pragma unroll
    for (int i = 0; i < QR4_K; ++i) {  // QR4_K=2
        // Extrai nibbles para a iteracao i
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        // DP4A para dot product dos pesos com ativacoes
        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1],
                         ggml_cuda_dp4a(v0i, u[2*i+0], 0));
        // DP4A para soma das ativacoes (para correcao de minimo)
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1],
                         ggml_cuda_dp4a(0x01010101, u[2*i+0], 0));

        sumf_d += d8[i] * (dot1 * sc[i]);  // escala * dot_product
        sumf_m += d8[i] * (dot2 * m[i]);   // minimo * soma_ativacoes
    }

    const float2 dm4f = __half22float2(dm4);

    // resultado = d * sum_escalas - dmin * sum_minimos
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}
```

**Diferenca chave Q4_K vs Q4_0**:
- Q4_K tem **escalas e minimos por sub-grupo** (6 bits cada, empacotados em 12 bytes)
- Q4_K usa super-blocos de 256 pesos (vs 32 do Q4_0) com sub-grupos de 32
- A dequantizacao Q4_K eh: `valor = d * sc[i] * quant - dmin * m[i]`
- Mais complexa mas melhor qualidade de quantizacao

---

## 8. MMQ: O Kernel para Prompt Processing (Batch > 1)

### Parametros do Template

```cuda
#define MMQ_DP4A_MAX_BATCH_SIZE 64
#define MMQ_ITER_K 256
#define MMQ_NWARPS 8
#define MMQ_TILE_NE_K 32
```

### Funcoes de Carga de Tiles (load_tiles)

```cuda
template <int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_q4_0(
    const char * x, int * x_tile, const int kbx0,
    const int i_max, const int stride) {

    // Distribuicao de threads:
    // threads_per_row = MMQ_ITER_K / (4 * QR4_0) = 256 / 8 = 32
    // nrows = warp_size / threads_per_row = 32/32 = 1

    // Dequantizacao inline durante carga:
    x_qs[...] = __vsubss4((qs0 >> 0) & 0x0F0F0F0F, 0x08080808);  // nibbles baixos - 8
    x_qs[...] = __vsubss4((qs0 >> 4) & 0x0F0F0F0F, 0x08080808);  // nibbles altos - 8
}
```

### Shared Memory Layout no MMQ

```cuda
// Formato de tile Q8_1 para shared memory:
union block_q8_1_mmq {
    float d4[4];     // 4 escalas
    half2 ds4[4];    // escalas emparelhadas
    half  d2s6[8];   // layout misto
    int8_t qs[4*QK8_1];  // 128 quants
};
// Layout projetado para evitar bank conflicts em shared memory
```

### Dot Product no MMQ (DP4A path)

```cuda
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q4_0_q8_1_dp4a(
    const int * x, const int * y, float * sum, const int k00) {

    // Itera sobre dimensao K em passos de QR4_0*VDR
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_0*VDR_Q4_0_Q8_1_MMQ) {
        // Carrega pares quantizados e faz DP4A
        // Acumula em sum[mmq_x][mmq_y]
    }
}
```

---

## 9. Analise de Desempenho e Benchmarks

### RTX 4090 -- Llama 2 7B Q4_0

| Metrica | Sem Flash Attention | Com Flash Attention |
|---------|--------------------|--------------------|
| Prompt Processing (pp512) | 11,993 t/s | 14,771 t/s |
| Token Generation (tg128) | 186 t/s | 189 t/s |

### RTX 4090 -- Outros Modelos Q4

| Modelo | Formato | Token Gen |
|--------|---------|-----------|
| Llama 3 8B | Q4 (int4) | ~150 t/s |
| Qwen3-30B | Q4_K_M | 198 t/s (otimizado) |
| Llama 2 7B | Q4_0 | 186-190 t/s |
| Llama 3.1 70B | Q4_K_M | 52-70 t/s |

### Comparacao com RTX 5090

| Modelo | RTX 4090 | RTX 5090 | Melhoria |
|--------|----------|----------|----------|
| Qwen3-30B Q4_K | 271 t/s | 352 t/s | +30% |
| GPT-OSS-20B | 272 t/s | 419 t/s | +54% |

### Comparacao com Outros Frameworks

| Framework | Velocidade Relativa | Notas |
|-----------|-------------------|-------|
| llama.cpp MMVQ | baseline | Melhor para batch=1 |
| ExLlamaV2 | ~2.2x mais rapido (prompt) | Kernels CUDA otimizados para GPTQ |
| TensorRT-LLM | ~1.7x mais rapido | Otimizacoes de grafo e fusao |
| vLLM | melhor em multi-usuario | Throughput escala com concorrencia |

### Utilizacao de Bandwidth

- **RTX 4090**: ~1,008 GB/s bandwidth VRAM
- **RTX 5090**: ~1,792 GB/s bandwidth VRAM
- **Eficiencia**: ~80% da bandwidth teorica maxima eh atingida
- **Token generation eh 100% memory-bandwidth bound**

Calculo de referencia para Llama 7B Q4_0:
```
Pesos ~= 7B * 0.5 bytes = 3.5 GB
Bandwidth RTX 4090 = 1008 GB/s
Tokens/s teorico = 1008 / 3.5 = 288 t/s
Real = 189 t/s = 65% da bandwidth
(overhead de KV cache, escalas, ativacoes, overhead de kernel)
```

---

## 10. Otimizacoes-Chave e Licoes

### O que torna o MMVQ rapido:

1. **Dequantizacao fusionada com dot product**: Nunca materializa tensores float32.
   Os pesos permanecem em INT4 na memoria e sao dequantizados "on-the-fly"
   dentro do kernel de multiplicacao.

2. **DP4A intrinsic**: Uma instrucao processa 4 pares int8*int8 -> int32.
   Para Q4_0, 2 chamadas dp4a processam 8 pesos por instrucao.
   No loop interno com VDR=2, sao 4 dp4a = 16 MACs por iteracao por thread.

3. **Warp shuffle para reducao**: `__shfl_xor_sync` evita shared memory para
   reducao intra-warp. Apenas a reducao inter-warp usa shared memory.

4. **Pre-computacao de soma Q8**: `block_q8_1.s = d * sum(qs)` permite
   corrigir o bias de quantizacao sem loop separado.

5. **Layout de memoria coalescido**: Padding para MATRIX_ROW_PADDING (512)
   garante acessos alinhados. Blocos quantizados sao sequenciais na memoria.

6. **Nao usa shared memory para vetor de entrada**: O vetor Y (ativacoes)
   eh lido direto da global memory, confiando no cache L2 para reuso.
   Isso simplifica o kernel e reduz pressao de shared memory.

7. **Nao usa loads vetorizados uint4**: Os loads sao via `get_int_b2` e
   `get_int_b4` que carregam int32 (4 bytes) por vez, nao uint4 (16 bytes).
   A coalescencia de threads vizinhas ja garante transacoes largas.

8. **1 linha por bloco CUDA**: No caso batch=1, cada bloco processa exatamente
   1 linha da matriz de pesos. Com 128 threads, todos os threads colaboram
   na reducao dessa unica linha.

### O que NAO fazem (oportunidades):

1. **Nao usam tensor cores para INT4**: Usam dp4a (CUDA cores), nao WMMA/MMA
   para a multiplicacao. Tensor cores poderiam ser mais rapidos.

2. **Nao fazem prefetch explicito**: Sem `__prefetch_global_l2()` ou
   double buffering. Dependem do hardware prefetcher.

3. **Nao processam multiplas linhas por bloco** (batch=1): Cada bloco faz
   1 linha, o que pode deixar SMs subutilizados para modelos menores.

4. **Escalas carregadas separadamente**: Escala `d` do block_q4_0 nao esta
   no mesmo cache line que os dados `qs` necessariamente.

---

## 11. Resumo para Nossa Implementacao

### Padroes a seguir:

1. **Fusionar dequantizacao com GEMV** -- nunca materializar float32
2. **Usar dp4a/dp2a** para dot products INT8 com acumulacao INT32
3. **Warp shuffle** para reducao intra-warp (5 passos para warp de 32)
4. **Shared memory** apenas para reducao inter-warp
5. **128 threads por bloco** (4 warps) para batch=1
6. **1 linha por bloco** CUDA, grid de nrows blocos
7. **Empacotar** 4 nibbles por int32, desempacotar com shifts e mascaras
8. **Pre-computar somas** de Q8 para correcao de bias

### Onde podemos superar:

1. **Compressao melhor** (nosso DCT/entropy coding) reduz bytes lidos,
   o que diretamente melhora throughput em workloads bandwidth-bound
2. **Tensor cores** para o dot product se tivermos formato compativel
3. **Prefetch explicito** pode ajudar em padroes de acesso nao-sequenciais
4. **Multiplas linhas por bloco** com registro cuidadoso
5. **Loads vetorizados** (float4/uint4) para dados de pesos

---

## Fontes

- [llama.cpp GitHub Repository](https://github.com/ggml-org/llama.cpp)
- [llama.cpp CUDA mmvq.cu](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu)
- [llama.cpp CUDA vecdotq.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vecdotq.cuh)
- [llama.cpp CUDA mmq.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cuh)
- [llama.cpp ggml-common.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h)
- [llama.cpp CUDA common.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh)
- [Performance of llama.cpp on Nvidia CUDA (Discussion #15013)](https://github.com/ggml-org/llama.cpp/discussions/15013)
- [Optimizing Token Generation in llama.cpp CUDA Backend (Discussion #17621)](https://github.com/ggml-org/llama.cpp/discussions/17621)
- [Accelerating LLMs with llama.cpp on NVIDIA RTX Systems](https://developer.nvidia.com/blog/accelerating-llms-with-llama-cpp-on-nvidia-rtx-systems/)
- [CUDA Backend DeepWiki](https://deepwiki.com/ggml-org/llama.cpp/5.1-command-line-tools)
- [NVIDIA DP4A Intrinsic Documentation](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html)
- [ExLlamaV2 GitHub](https://github.com/turboderp-org/exllamav2)
- [Quantization Evaluation (arXiv)](https://arxiv.org/html/2601.14277v1)
- [Q4_K Quantization Scheme Discussion](https://github.com/ggml-org/llama.cpp/discussions/6760)
