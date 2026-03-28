# Pesquisa: Arquitetura NVIDIA Blackwell e Suporte a INT4

## 1. Arquitetura NVIDIA Blackwell e Tensor Cores INT4

### 1.1 Compute Capability e Variantes SM

A arquitetura Blackwell introduz duas familias de compute capability:

| GPU | SM | Compute Capability | Segmento |
|-----|----|--------------------|----------|
| B100, B200, GB200 | sm_100 | 10.0 | Data center |
| RTX 5090, RTX 5080 | sm_120 | 12.0 | Consumer |
| RTX PRO 6000 | sm_122 | 12.2 | Workstation |

O CUDA Toolkit 12.8 adicionou suporte para sm_100 e sm_120. As flags de compilacao tipicas sao:
```
-gencode arch=compute_100,code=sm_100
-gencode arch=compute_120,code=sm_120
```

### 1.2 INT4 nos Tensor Cores: Situacao Atual

**INT4 NAO e suportado nativamente no Blackwell.** Esta e a descoberta mais importante desta pesquisa.

Historico do suporte INT4 nos Tensor Cores NVIDIA:

| Arquitetura | Geracao TC | INT4 Nativo? | Instrucao SASS |
|-------------|-----------|--------------|----------------|
| Turing (sm_75) | 2a geracao | Sim | `IMMA.8832.S4.S4.SAT` |
| Ampere (sm_80) | 3a geracao | Sim | `IMMA.16832.S4.S4` |
| Ada Lovelace (sm_89) | 4a geracao | Sim | `IMMA.16832.S4.S4` |
| Hopper (sm_90) | 4a geracao | **Deprecated** | Emulado via `IMAD` nos CUDA cores |
| Blackwell (sm_100) | 5a geracao | **Nao** | Emulado via INT8 (`IMMA.8816.S8.S8`) |

**O que aconteceu:**
- No Turing e Ampere, instrucoes `mma.sync` com tipos `.s4` e `.u4` eram compiladas em instrucoes SASS nativas `IMMA` nos Tensor Cores
- No Hopper, INT4 foi marcado como *deprecated*. As instrucoes `mma.sync` com INT4 passaram a ser compiladas em sequencias de `IMAD` que executam nos **CUDA cores regulares**, nao nos Tensor Cores
- A instrucao `wgmma` (warp-group MMA) do Hopper **nunca suportou INT4**
- No Blackwell, `tcgen05.mma` tambem **nao suporta INT4**. Operacoes INT4 sao emuladas via descompactacao para INT8

**Confirmacao oficial:** No forum NVIDIA Developer, o usuario rs277 confirmou que Blackwell (SM 100) nao possui suporte nativo a INT4 nos Tensor Cores.

### 1.3 Instrucoes PTX no Blackwell: tcgen05.mma

Blackwell substitui completamente o paradigma anterior de instrucoes MMA:

| Aspecto | mma.sync (Ampere) | wgmma (Hopper) | tcgen05.mma (Blackwell) |
|---------|-------------------|----------------|------------------------|
| Escopo | Warp-sync | Warp-group async | **Single-thread** |
| Entrada | Registradores | Shared memory | Shared memory |
| Saida | Registradores | Registradores | **Tensor Memory (TMEM)** |
| Max BF16 | m16n8k16 | m64n256k16 | **m128n256k16** |
| Latencia | Baseline | ~2x baseline | **2.9-11.6x menor** que Hopper |

**Tipos de dados suportados pelo tcgen05.mma:**
- FP64 -> instrucao SASS `DMMA`
- FP32/TF32 -> instrucao SASS `HMMA`
- BF16/FP16 -> instrucao SASS `HMMA`
- FP8 -> instrucao SASS `QMMA`
- FP6/FP4 -> instrucao SASS `OMMA` (nova no Blackwell)
- INT8 -> instrucao SASS `IMMA`
- ~~INT4~~ -> **NAO suportado nativamente**

O qualificador `.kind::f8f6f4` do tcgen05.mma suporta operacoes mistas entre 5 tipos de dados de ponto flutuante de baixa precisao:
- E5M2 (FP8)
- E4M3 (FP8)
- E3M2 (FP6)
- E2M3 (FP6)
- E2M1 (FP4)

Nota: **Todos sao ponto flutuante**. Nenhum tipo inteiro sub-byte e suportado.

### 1.4 Peak TOPS por Tipo de Dado

#### RTX 5090 (Consumer Blackwell)
| Precisao | Dense | Sparse |
|----------|-------|--------|
| FP4 | ~1,676 TOPS | ~3,352 TOPS |
| FP8 | ~838 TFLOPS | ~1,676 TFLOPS |
| INT8 | ~838 TOPS | ~1,676 TOPS |
| FP16 | ~419 TFLOPS | ~838 TFLOPS |
| INT4 | N/A (emulado) | N/A (emulado) |

- CUDA Cores: 21.760
- Tensor Cores: 680 (5a geracao)
- Memoria: 32 GB GDDR7
- Bandwidth: 1.792 GB/s
- AI TOPS total: 3.352 (FP4 sparse)

#### B200 (Data Center Blackwell)
| Precisao | Dense | Sparse |
|----------|-------|--------|
| FP4 | 9 PFLOPS | 18 PFLOPS |
| FP8 | 4.5 PFLOPS | 9 PFLOPS |
| INT8 | 4.5 POPS | 9 POPS |

- Memoria: 192 GB HBM3e
- Bandwidth: 8 TB/s

#### Microbenchmarks Reais (B200, fonte: arxiv 2512.02189)
| Precisao | Throughput Medido | % do Pico Teorico |
|----------|-------------------|-------------------|
| FP64 | 44.8 TFLOPS | 99.6% |
| FP32 | 481.2 TFLOPS | 96.2% |
| FP16 | 1.929.2 TFLOPS | 96.5% |
| FP8 | 3.851.4 TFLOPS | 96.3% |
| FP4 | 7.702.5 TFLOPS | 96.3% |
| INT8 | 3.927.1 TOPS | 98.2% |

**Observacao critica:** INT8 tem throughput ligeiramente superior a FP8, sugerindo que operacoes inteiras e de ponto flutuante compartilham as mesmas unidades de execucao.

---

## 2. cuBLAS e INT4 GEMM no Blackwell

### 2.1 Estado do Suporte em cuBLAS 12.9/13.2

**cuBLAS NAO oferece suporte a INT4 GEMM.** A biblioteca foca em formatos de ponto flutuante:

**Novos tipos de dados em cuBLAS 12.9 para Blackwell:**
- `CUDA_R_4F_E2M1` (FP4) com escalas `CUDA_R_UE4M3` e blocos de 16 elementos
- `CUDA_R_8F` (FP8) com escalas `CUDA_R_UE8` e blocos de 32 elementos

**Block Scaling no cuBLAS:**
- 1D Block Scaling: fator de escala por bloco de 128 elementos na dimensao K
- 2D Block Scaling: fatores para blocos 128x128 (melhor performance)
- Modos mistos: 1D x 1D, 1D x 2D, 2D x 1D
- Calculo dinamico de escalas de saida

**Performance cuBLAS 12.9 no Blackwell:**
- FP4 block-scaled: **4.6x mais rapido** que FP8 baseline no H200
- FP4 block-scaled absoluto: ate 6.787 TFLOPS no GB200
- FP8 scaling: ate 1.75x speedup vs BF16 baseline
- FP32 emulado via BF16 TC: 3-4x mais rapido que FP32 nativo

### 2.2 API cublasLtMatmul

Para operacoes FP4 no cublasLt:
```c
// Tipo de dado FP4 com block scaling
cublasLtMatmulDescSetAttribute(matmulDesc,
    CUBLASLT_MATMUL_DESC_A_DATA_TYPE, CUDA_R_4F_E2M1);
// Escalas micro-block FP8
cublasLtMatmulDescSetAttribute(matmulDesc,
    CUBLASLT_MATMUL_DESC_A_SCALE_TYPE, CUDA_R_UE4M3);
```

**Nao existe `CUDA_R_4I` ou equivalente INT4 no cublasLt.**

### 2.3 Alternativas para INT4 GEMM

Como cuBLAS nao suporta INT4, as opcoes sao:
1. **CUTLASS 3.x** - Templates C++ para kernels GEMM customizados
2. **Marlin** - Kernel otimizado para INT4xFP16 (ver secao 5)
3. **TensorRT-LLM** - Kernels internos para INT4/INT8
4. **Conversao para NVFP4** - Usar o formato nativo do Blackwell

---

## 3. FP4 (NVFP4) vs INT4 no Blackwell

### 3.1 Formato NVFP4 (E2M1)

O NVFP4 e o formato de 4 bits de ponto flutuante nativo da NVIDIA para Blackwell:

```
Formato E2M1: [S][E1 E0][M0]
  S  = 1 bit de sinal
  E  = 2 bits de expoente
  M  = 1 bit de mantissa
```

**Valores representaveis pelo E2M1:**
```
{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} e seus negativos
```

**Diferenca fundamental de espacamento:**

```
INT4 (uniforme):
-8 -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7
 |  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
 Espacamento igual em toda a faixa

FP4/E2M1 (exponencial):
-6.0  -4.0  -3.0  -2.0  -1.5  -1.0  -0.5   0   0.5  1.0  1.5  2.0  3.0  4.0  6.0
  *     *     *     *      *     *     *    *    *    *    *    *    *    *    *
  |--largo--|--medio--|----denso----|----denso----|--medio--|--largo--|
```

### 3.2 Block Scaling do NVFP4

O NVFP4 usa uma estrategia de escala em dois niveis:

1. **Nivel micro-bloco:** Fator de escala E4M3 (FP8) para cada bloco de **16 valores**
   - Menor que MXFP4 (que usa blocos de 32)
   - Permite adaptacao mais localizada ao range dinamico
2. **Nivel tensor:** Escalar FP32 por tensor inteiro

Reconstrucao: `x_real = x_fp4 * scale_fp8 * scale_fp32`

### 3.3 Comparacao de Qualidade para LLM

**Benchmarks com DeepSeek-R1-0528 (NVFP4 vs FP8):**
| Benchmark | FP8 | NVFP4 | Degradacao |
|-----------|-----|-------|------------|
| MMLU-PRO | 85% | 84% | -1% |
| GPQA Diamond | 81% | 80% | -1% |
| Math-500 | 98% | 98% | 0% |
| AIME 2024 | 89% | 91% | **+2%** |

Degradacao tipica: **1% ou menos** em tarefas de modelagem de linguagem.

### 3.4 INT4 vs FP4: Qual e Melhor para Inferencia LLM?

| Aspecto | INT4 | FP4 (NVFP4) |
|---------|------|-------------|
| Precisao perto de zero | Ruim (uniforme) | Excelente (denso) |
| Tratamento de outliers | Risco de overflow | Adaptativo |
| Suporte Ampere (A100) | Sim (via software) | Nao |
| Suporte Hopper (H100) | Sim (via software) | Nao |
| Suporte Blackwell | **Nao nativo** | **Nativo nos TC** |
| Distribuicao de pesos | Subotimo | Alinhado com redes neurais |
| Throughput no Blackwell | Emulado (~INT8 speed) | 2x do FP8 |

**Conclusao:** Para Blackwell, FP4 e objetivamente superior:
- Suporte nativo nos Tensor Cores com instrucao OMMA dedicada
- 2x throughput vs FP8
- Melhor qualidade de inferencia para distribuicoes de pesos tipicas de LLMs
- Pipeline completo: TensorRT-LLM, vLLM, SGLang

**Porem,** INT4 continua relevante para:
- GPUs pre-Blackwell (Ampere, Ada Lovelace)
- Modelos ja quantizados em INT4 (GPTQ, AWQ)
- Ecosistema enorme de modelos INT4 existentes

### 3.5 Implicacao Estrategica

Existe uma fragmentacao no ecosistema:
- **Modelos chineses** (ex: Kimi K2) usam INT4 por terem acesso a Ampere/Hopper
- **Ecosistema Blackwell** empurra FP4 como formato padrao
- Conversao INT4 -> FP4 via PTQ tem perda de qualidade (valores aprendidos no grid INT4 caem entre niveis FP4)
- Treinamento nativo em FP4 > Treinamento INT4 > Conversao PTQ (INT4->FP4)

---

## 4. CUDA PTX para INT4: Estado Atual

### 4.1 Instrucoes mma.sync para Tipos Sub-byte

**Historico das instrucoes PTX para INT4:**

```
# Turing/Ampere (funcionava nativamente):
mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {d0,d1}, {a0}, {b0}, {c0,c1};
mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {d0..d3}, {a0..a3}, {b0,b1}, {c0..c3};

# Hopper: ainda compila mas executa em CUDA cores (IMAD), nao Tensor Cores
# Blackwell: emulado via descompactacao INT4->INT8 + IMMA INT8
```

**Instrucoes tcgen05.mma no Blackwell (substituem mma.sync):**
```
# Exemplo FP4 (funciona nativamente):
tcgen05.mma.cta_group::1.kind::f8f6f4 [tmem_addr], desc_a, desc_b, idesc, enable;

# O descriptor 'idesc' codifica os tipos em runtime:
#   - atype = E2M1 (FP4)
#   - btype = E2M1 (FP4)
#   - dtype = FP32 (acumulador)
# Nao existe encoding para tipos inteiros sub-byte no idesc
```

### 4.2 dp4a e Operacoes de Dot Product

O `dp4a` (Dot Product of 4 x INT8, Accumulate) foi introduzido no Pascal (SM 6.1):

```c
// dp4a: dot product de 4 int8s empacotados em int32
int dp4a(int a, int b, int c) {
    // a = [a3|a2|a1|a0] (4x int8 em 1x int32)
    // b = [b3|b2|b1|b0] (4x int8 em 1x int32)
    return c + a0*b0 + a1*b1 + a2*b2 + a3*b3;
}
```

**dp4a NAO funciona para INT4 diretamente.** Ele opera em granularidade de 8 bits. Para usar INT4 com dp4a, seria necessario:
1. Descompactar INT4 -> INT8 (shift + mask)
2. Executar dp4a nos valores INT8
3. Isso e essencialmente o que a emulacao faz

**Nao existe instrucao PTX equivalente a "dp8a" para 8x INT4.**

### 4.3 Instrucoes PTX para INT4 Dot Products

Nao existem instrucoes PTX dedicadas para dot products INT4 no Blackwell. As unicas opcoes sao:

1. **Emulacao via INT8:** Descompactar, usar `mma` INT8 ou `dp4a`
2. **Conversao para FP4:** Converter para E2M1, usar `tcgen05.mma` com `.kind::f8f6f4`
3. **Bit manipulation manual:** Extrair valores INT4, operar com instrucoes escalares

A NVIDIA claramente decidiu que o futuro de 4 bits e **ponto flutuante (FP4)**, nao inteiro (INT4).

---

## 5. Kernels INT4 de Producao em GPUs NVIDIA

### 5.1 Marlin (IST-DASLab)

Marlin e o kernel INT4xFP16 mais otimizado para inferencia de LLMs. Alcanca throughput proximo do ideal em GPUs Ampere e posteriores.

**Principio fundamental:** Na inferencia com batch pequeno (1-32 tokens), o gargalo e **bandwidth de memoria**, nao compute. Com pesos INT4 (4 bits vs 16 bits do FP16), o throughput teorico maximo e **4x** o do FP16.

**Tecnicas-chave de otimizacao:**

#### 5.1.1 Exploracao do L2 Cache
```
Pipeline duplo:
  Global Memory -> L2 cache (pesos B)     [simultaneo]
  L2 cache -> L1/Shared Memory (ativacoes A)  [simultaneo]

Inequacao critica para overlap:
(2*M*K_sm + 0.5*K_sm*N_sm) / BW_L2 < (0.5*K_sm*N_sm) / BW_global
```

#### 5.1.2 Pipeline Assincrono com Profundidade 4
- Usa `cp.async` do Ampere com hint `evict_first`
- 4 estagios de pipeline com double buffering
- Loop principal completamente unrolled
- Todos os enderecos de shared memory sao **estaticos** (sem overhead de calculo de indice)

#### 5.1.3 Desquantizacao Eficiente via Bit Manipulation
```
Em vez de: INT4 -> cast -> FP16 (caro)
Marlin faz: manipulacao binaria direta em registradores

Para extrair INT4 das posicoes 12-15 de um INT16:
1. AND com mascara (extrair bits)          ]
2. OR para setar expoente = 0110 (exp 50)  ] -> tudo em 1 instrucao lop3
3. SUB para extrair mantissa via FP16
4. SUB 8 para tornar signed

Dois INT4 em um INT32 sao desquantizados simultaneamente
usando operacoes de 16 bits empacotadas em registrador de 32 bits.
```

#### 5.1.4 Layout de Pesos Pre-processado
- Tiles 16x64 reorganizados contiguamente na memoria
- Pesos intercalados no padrao `64207531` para descodificacao paralela
- 8 threads carregam 128 bytes (uma cache line) por instrucao
- Um warp carrega 1024 pesos INT4 por instrucao

#### 5.1.5 Shared Memory sem Conflitos de Banco
```
Enderecos de shared memory para ativacoes A:
  store(i, j) -> location[i * (i XOR j)]
  XOR de indices garante acesso sem conflitos para ldmatrix.sync
```

**Performance do Marlin:**
- ~3.9x speedup vs FP16 no A10 (batch 16-32) -- proximo do maximo teorico de 4x
- ~2.8x speedup no vLLM em servico multi-usuario (batch 16)
- Gradualmente reduz para ~1.5x em batch 128 (regime compute-bound)

### 5.2 ExLlamaV2

ExLlamaV2 e a implementacao de referencia para inferencia GPTQ em GPUs consumer.

**Tecnicas de desquantizacao:**
- Kernels CUDA fusionados que desquantizam INT4 -> FP16 on-the-fly
- Mesma tecnica de bit manipulation que Marlin (AND + OR via `lop3`)
- Dois INT4 desquantizados simultaneamente em um registrador INT32
- Pesos nunca sao materializados em FP16 na memoria global

**Diferencial:** Kernels altamente otimizados para batch=1 (geracao token a token), que e o caso de uso mais comum em inferencia local.

### 5.3 vLLM: Arquitetura de Kernels Quantizados

vLLM implementa multiplos backends de kernels quantizados:

| Kernel | Formato | GPUs Suportadas | Caso de Uso |
|--------|---------|-----------------|-------------|
| Marlin | INT4 (W4A16) | SM 8.0+ | GPTQ/AWQ otimizado |
| ExLlamaV2 | INT4 (W4A16) | SM 7.5+ | GPTQ padrao |
| AWQ nativo | INT4 (W4A16) | SM 7.5+ | AWQ padrao |
| Machete | Mixed | SM 9.0a | Hopper otimizado |
| CUTLASS W8A8 | INT8/FP8 | SM 9.0a, 10.0a+ | Alta precisao |
| NVFP4 | FP4 | SM 10.0a+ | Blackwell nativo |

**Performance comparativa em vLLM (Llama):**
- Marlin GPTQ: 712 tok/s (2.6x vs GPTQ padrao)
- Marlin AWQ: 741 tok/s (10.9x vs AWQ padrao)
- Os pesos quantizados sao identicos -- a diferenca e **100% do kernel**

### 5.4 Como Esses Kernels Alcancam Near-Peak Bandwidth

A receita fundamental e a mesma em todos os kernels otimizados:

**1. Problema e memory-bound (batch pequeno):**
```
Arithmetic Intensity de INT4xFP16:
  = 2*M*N*K FLOPs / (0.5*N*K + 2*M*K + 2*M*N) bytes
  Para M=1 (inferencia): ~25-50 FLOPs/byte
  GPU A10: ~200 FLOPs/byte de capacity
  -> Claramente memory-bound para batch < 32
```

**2. Minimizar bytes lidos = maximizar throughput:**
- INT4: 0.5 bytes por peso vs 2 bytes (FP16) = 4x menos dados
- Desquantizacao e "gratis" -- compute sobrando enquanto espera memoria

**3. Pipeline assincrono esconde latencia:**
```
Tempo 0: Carrega bloco[0] da global memory
Tempo 1: Carrega bloco[1] | Computa bloco[0]
Tempo 2: Carrega bloco[2] | Computa bloco[1]
...
O tempo total = tempo de leitura da memoria (compute totalmente escondido)
```

**4. Fusao completa:**
- Desquantizacao + GEMM + bias + activation em um unico kernel
- Zero materializacao de intermediarios na memoria global
- Reducao de launches de kernel (overhead de sincronizacao)

**5. Acesso coalescido e layout otimizado:**
- Pesos pre-permutados offline para acesso sequencial
- Shared memory com XOR-swizzle para evitar conflitos de banco
- Loads de 128 bits (16 bytes) por thread -- maximo da GPU

---

## 6. Resumo e Implicacoes Praticas

### Para quem esta desenvolvendo kernels INT4:

1. **No Blackwell:** INT4 nao tem aceleracao de hardware. Considere migrar para NVFP4 (E2M1)
2. **Em Ampere/Ada:** INT4 nos Tensor Cores ainda funciona nativamente -- Marlin e o estado da arte
3. **Conversao INT4 -> FP4:** Possivel mas com perda de qualidade. Melhor re-quantizar do modelo FP16/BF16 original

### Para quem esta fazendo inferencia de LLMs:

1. **GPU Blackwell:** Use modelos NVFP4 com TensorRT-LLM ou vLLM. Performance: ~2x do FP8
2. **GPU Ampere/Ada:** Use modelos GPTQ/AWQ INT4 com kernel Marlin. Performance: ~3.9x do FP16
3. **GPU Hopper:** Use FP8 (nativo) ou INT4 via Marlin (sem aceleracao nativa INT4 nos TC)

### Numeros-chave para referencia rapida:

| Metrica | RTX 5090 | B200 |
|---------|----------|------|
| FP4 Sparse TOPS | 3.352 | 18.000 |
| FP8 Dense TFLOPS | 838 | 4.500 |
| INT8 Dense TOPS | 838 | 4.500 |
| HBM/GDDR BW | 1.792 GB/s | 8.000 GB/s |
| INT4 nativo | Nao | Nao |

---

## Fontes

### Arquitetura e Especificacoes
- [RTX PRO 6000 Blackwell](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)
- [RTX 5090 Specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)
- [RTX 5090 vs RTX 4080 Super AI Specs](https://www.bestgpusforai.com/gpu-comparison/5090-vs-4080-super)
- [Comparing Blackwell vs Hopper](https://www.exxactcorp.com/blog/hpc/comparing-nvidia-tensor-core-gpus)
- [NVIDIA RTX Blackwell GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)

### INT4 e PTX
- [Does Blackwell support INT4 native? - NVIDIA Forums](https://forums.developer.nvidia.com/t/does-blackwell-support-int4-native/326513)
- [Microbenchmarking Blackwell Architecture](https://arxiv.org/html/2512.02189v1)
- [NVIDIA Tensor Core Evolution: Volta to Blackwell](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell)
- [tcgen05 for dummies](https://gau-nernst.github.io/tcgen05/)
- [CUTLASS Tutorial: Sub-byte GEMM on Blackwell](https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/)
- [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html)

### FP4 / NVFP4
- [Introducing NVFP4 - NVIDIA Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [INT4 vs FP4: Future of 4-Bit Quantization - HuggingFace](https://huggingface.co/blog/onekq/nvfp4-int4)
- [FP4 Quantization on Blackwell GPUs](https://www.spheron.network/blog/fp4-quantization-blackwell-gpu-cost/)
- [NVIDIA TensorRT FP4 Image Generation](https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus/)

### cuBLAS
- [cuBLAS 12.9 Blog Post](https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9/)
- [cuBLAS 13.2 Documentation](https://docs.nvidia.com/cuda/cublas/)
- [Grouped GEMM APIs in cuBLAS](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/)

### Kernels de Producao
- [Marlin Paper (IST-DASLab)](https://arxiv.org/html/2408.11743v1)
- [How Marlin Pushes Boundaries - Red Hat](https://developers.redhat.com/articles/2024/04/17/how-marlin-pushes-boundaries-mixed-precision-llm-inference)
- [vLLM Quantization Kernels](https://deepwiki.com/bytedance-iaas/vllm/11.4-quantization-kernels)
- [vLLM Quantization Docs](https://docs.vllm.ai/en/latest/features/quantization/)
- [ExLlamaV2 GitHub](https://github.com/turboderp-org/exllamav2)
- [Accelerating Triton Dequantization - PyTorch Blog](https://pytorch.org/blog/accelerating-triton/)
