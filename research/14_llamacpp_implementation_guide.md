# Guia Pratico: Como Adicionar um Novo Tipo de Quantizacao ao llama.cpp

**Data:** 2026-03-28
**Escopo:** Guia passo-a-passo completo para implementar um novo tipo de quantizacao no llama.cpp, cobrindo desde a definicao do tipo ate testes e integracao com codificacao por entropia (rANS).
**Base de codigo:** llama.cpp (ggml-org/llama.cpp) -- codebase 2025-2026.

---

## Sumario Executivo

Adicionar um novo tipo de quantizacao ao llama.cpp requer modificacoes em **8-12 arquivos** distribuidos em quatro camadas: (1) definicao do tipo no sistema de tipos GGML, (2) implementacao dos kernels de quantizacao/dequantizacao, (3) registro nas tabelas de type traits, e (4) integracao com ferramentas de conversao e teste. Este guia detalha cada etapa usando como referencia PRs reais: o commit TurboQuant (TBQ3_0/TBQ4_0), o PR #19769 (NVFP4), e o PR #4897 (IQ2_XXS/IQ2_XS).

---

## 1. Visao Geral da Arquitetura

### 1.1 Estrutura de Diretorios Relevante

```
llama.cpp/
  ggml/
    include/
      ggml.h                    # Enum ggml_type, struct ggml_type_traits
    src/
      ggml.c                    # Tabela type_traits[] (metadata dos tipos)
      ggml-common.h             # Definicoes de block structs (compartilhado CPU/GPU)
      ggml-quants.h             # Declaracoes de funcoes quantize/dequantize
      ggml-quants.c             # Implementacoes de referencia (scalar)
      CMakeLists.txt            # Build do ggml-base (inclui ggml-quants.c)
      ggml-cpu/
        ggml-cpu.c              # Tabela type_traits_cpu[] (vec_dot, from_float)
        quants.c                # Implementacoes genericas (fallback)
        arch-fallback.h         # Macros para fallback de ISA
        CMakeLists.txt          # Build do backend CPU
        arch/
          arm/
            quants.c            # Kernels NEON otimizados
          x86/
            quants.c            # Kernels AVX2/AVX512 otimizados
      ggml-cuda/                # Kernels CUDA (fase posterior)
      ggml-metal/               # Kernels Metal (fase posterior)
  src/
    llama-model-quantize.cpp    # Logica interna de quantizacao de modelo
  include/
    llama.h                     # Enum llama_ftype (LLAMA_FTYPE_MOSTLY_*)
  tools/
    quantize/
      quantize.cpp              # Frontend da ferramenta de quantizacao
  gguf-py/
    gguf/
      constants.py              # Enum GGMLQuantizationType (Python)
      quants.py                 # Implementacao Python de quantize/dequantize
  convert_hf_to_gguf.py         # Conversor HuggingFace -> GGUF
  tests/
    test-quantize-fns.cpp       # Testes unitarios de quantizacao
    test-backend-ops.cpp        # Testes de operacoes por backend
```

### 1.2 As Duas Tabelas de Type Traits

O GGML separa os metadados de tipo em **duas tabelas**:

**Tabela 1 -- `type_traits[]`** (em `ggml/src/ggml.c`, linhas ~609-899):
```c
struct ggml_type_traits {
    const char * type_name;        // Ex: "q4_0"
    int64_t      blck_size;        // Elementos por bloco (32 ou 256)
    size_t       type_size;        // Bytes por bloco
    bool         is_quantized;     // true para tipos quantizados
    ggml_to_float_t    to_float;   // Ponteiro para funcao de dequantizacao
    ggml_from_float_t  from_float_ref; // Ponteiro para funcao de quantizacao
};
```

**Tabela 2 -- `type_traits_cpu[]`** (em `ggml/src/ggml-cpu/ggml-cpu.c`, linhas ~205-388):
```c
struct ggml_type_traits_cpu {
    ggml_from_float_t  from_float; // Quantizacao otimizada (pode ser SIMD)
    ggml_vec_dot_t     vec_dot;    // Dot product otimizado
    enum ggml_type     vec_dot_type; // Tipo companion para dot product
    int                nrows;      // Linhas processadas simultaneamente
};
```

A separacao permite que `type_traits[]` seja universal (usada por todos os backends), enquanto `type_traits_cpu[]` contem as implementacoes SIMD especificas para CPU.

### 1.3 Fluxo de Multiplicacao de Matrizes

```
ggml_compute_forward_mul_mat()
  |
  +-- Obtém tipo de src0 (pesos quantizados, ex: Q4_0)
  |
  +-- Consulta type_traits_cpu[src0->type].vec_dot_type
  |     -> retorna GGML_TYPE_Q8_0 (tipo de ativacao)
  |
  +-- Quantiza src1 (ativacoes FP32) para Q8_0 on-the-fly
  |     usando type_traits_cpu[vec_dot_type].from_float()
  |
  +-- Chama type_traits_cpu[src0->type].vec_dot()
  |     -> executa dot product fused: Q4_0 * Q8_0
  |
  +-- Resultado em FP32
```

Pontos criticos:
- Os pesos (src0) ficam em formato quantizado na memoria
- As ativacoes (src1) sao quantizadas dinamicamente para `vec_dot_type` (geralmente Q8_0 ou Q8_K)
- O `vec_dot` computa o dot product **diretamente nos dados quantizados** sem dequantizacao completa
- Cada tipo de quantizacao define qual e seu `vec_dot_type` correspondente

---

## 2. Passo 1: Definir o Tipo no Enum (ggml.h)

### 2.1 Adicionar ao Enum `ggml_type`

**Arquivo:** `ggml/include/ggml.h`

```c
enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // ... (4, 5 deprecados)
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    // 31-33 reservados
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    // 36-38 reservados
    GGML_TYPE_MXFP4   = 39,
    GGML_TYPE_NVFP4   = 40,

    // === SEU NOVO TIPO AQUI ===
    GGML_TYPE_DCT4    = 41,  // DCT-coded 4-bit quantization
    GGML_TYPE_COUNT   = 42,  // Atualizar!
};
```

**Regras importantes:**
- IDs nao podem ser reutilizados (compatibilidade GGUF)
- Gaps sao permitidos (veja 4-5, 31-33, 36-38)
- `GGML_TYPE_COUNT` deve ser o valor mais alto + 1
- O ID sera armazenado no arquivo GGUF como tipo do tensor

### 2.2 Adicionar ao Enum `llama_ftype`

**Arquivo:** `include/llama.h`

```c
enum llama_ftype {
    LLAMA_FTYPE_ALL_F32              = 0,
    LLAMA_FTYPE_MOSTLY_F16           = 1,
    LLAMA_FTYPE_MOSTLY_Q4_0          = 2,
    // ...
    LLAMA_FTYPE_MOSTLY_NVFP4         = 36,

    // === SEU NOVO TIPO ===
    LLAMA_FTYPE_MOSTLY_DCT4          = 37,
};
```

O `llama_ftype` descreve a quantizacao **predominante** no modelo (cada tensor pode ter tipo diferente).

---

## 3. Passo 2: Definir a Estrutura de Bloco (ggml-common.h)

### 3.1 Constantes de Bloco

**Arquivo:** `ggml/src/ggml-common.h`

As constantes-chave pre-existentes:
```c
#define QK_K 256        // Super-bloco: 256 elementos (K-quants, I-quants)
#define K_SCALE_SIZE 12 // Bytes para scales em K-quants
#define QK4_0 32        // Bloco basico: 32 elementos (legacy quants)
```

Para um novo tipo, voce pode definir:
```c
#define QK_DCT4 256     // Usar super-blocos de 256 (recomendado)
```

### 3.2 Definir a Struct do Bloco

A struct do bloco e definida em `ggml-common.h` porque e compartilhada entre todos os backends (CPU, CUDA, Metal, Vulkan).

**Exemplo -- struct hipotetica para DCT-coded 4-bit:**
```c
// DCT4: 256 valores quantizados a 4-bit com coeficientes DCT
// Layout: header (metadados) + coeficientes DCT quantizados
typedef struct {
    ggml_half d;                    // Scale global do bloco (FP16, 2 bytes)
    uint8_t dct_scales[QK_DCT4/32]; // Scale por sub-bloco de 32 (8 bytes)
    uint8_t qs[QK_DCT4/2];          // 4-bit quants empacotados (128 bytes)
} block_dct4;
// Total: 2 + 8 + 128 = 138 bytes para 256 valores
// Bits por peso: 138 * 8 / 256 = 4.3125 bpw

static_assert(sizeof(block_dct4) == sizeof(ggml_half) + QK_DCT4/32 + QK_DCT4/2,
    "wrong dct4 block size/padding");
```

### 3.3 Padroes de Struct Existentes (Referencia)

**Tipo legacy simples -- Q4_0 (18 bytes, 32 elementos):**
```c
typedef struct {
    ggml_half d;           // Scale (2 bytes)
    uint8_t qs[QK4_0 / 2]; // 32 valores x 4-bit = 16 bytes
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2);
// 4.5 bpw
```

**K-quant hierarquico -- Q4_K (144 bytes, 256 elementos):**
```c
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d;       // Super-block scale
            ggml_half dmin;    // Super-block minimum
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE];  // 12 bytes: sub-block scales (6-bit)
    uint8_t qs[QK_K/2];            // 128 bytes: 4-bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2);
// 4.5 bpw
```

**I-quant com lookup table -- IQ2_XXS (66 bytes, 256 elementos):**
```c
typedef struct {
    ggml_half d;                     // Scale global (2 bytes)
    uint16_t qs[QK_K/8];            // Indices na grid (64 bytes)
} block_iq2_xxs;
static_assert(sizeof(block_iq2_xxs) == sizeof(ggml_half) + QK_K/8*sizeof(uint16_t));
// ~2.06 bpw
```

**TurboQuant -- TBQ3_0 (98 bytes, 256 elementos):**
```c
typedef struct {
    ggml_half norm;                  // Norma do vetor (2 bytes)
    uint8_t qs[96];                  // 3-bit codebook indices, packed
} block_tbq3_0;
// ~3.06 bpw
```

### 3.4 Requisitos de Alinhamento e Memoria

- **Sem pragma pack**: as structs usam layout natural do compilador. `static_assert` garante o tamanho correto.
- **Macros de compatibilidade**: `GGML_EXTENSION`, `GGML_COMMON_AGGR_S`, `GGML_COMMON_AGGR_U` garantem portabilidade entre C/C++/CUDA/Metal/Vulkan.
- **Alinhamento GGUF**: o alinhamento padrao no arquivo GGUF e 32 bytes (`GGUF_DEFAULT_ALIGNMENT = 32`), mas pode ser customizado via metadado `general.alignment`.
- **Tamanho de bloco**: deve ser divisor do numero total de pesos no tensor. Blocos de 256 (`QK_K`) sao preferidos para K-quants/I-quants; blocos de 32 para legacy quants.

---

## 4. Passo 3: Implementar Quantizacao e Dequantizacao

### 4.1 Declarar Funcoes (ggml-quants.h)

**Arquivo:** `ggml/src/ggml-quants.h`

```c
// Dequantizacao: block_dct4 -> float32
void dequantize_row_dct4(const block_dct4 * GGML_RESTRICT x,
                         float * GGML_RESTRICT y, int64_t k);

// Quantizacao referencia: float32 -> block_dct4
void quantize_row_dct4_ref(const float * GGML_RESTRICT x,
                           block_dct4 * GGML_RESTRICT y, int64_t k);

// Quantizacao wrapper (pode delegar para versao otimizada)
void quantize_row_dct4(const float * GGML_RESTRICT x,
                       void * GGML_RESTRICT y, int64_t k);
```

### 4.2 Implementar Funcoes de Referencia (ggml-quants.c)

**Arquivo:** `ggml/src/ggml-quants.c`

As funcoes de referencia sao implementacoes **escalares** (sem SIMD) que servem como:
1. Implementacao correta de fallback
2. Referencia para validar implementacoes otimizadas
3. Funcionalidade base para plataformas sem SIMD

```c
// === DEQUANTIZACAO ===
void dequantize_row_dct4(const block_dct4 * GGML_RESTRICT x,
                         float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK_DCT4;  // 256
    assert(k % qk == 0);
    const int nb = k / qk;  // numero de blocos

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);  // Scale global

        for (int sb = 0; sb < qk/32; sb++) {
            const float sd = x[i].dct_scales[sb];  // Scale do sub-bloco

            for (int j = 0; j < 16; j++) {
                // Desempacotar 2 valores de 4-bit de 1 byte
                const uint8_t packed = x[i].qs[sb * 16 + j];
                const int8_t v0 = (packed & 0x0F) - 8;  // Lower nibble
                const int8_t v1 = (packed >> 4)   - 8;  // Upper nibble

                y[i*qk + sb*32 + j]      = d * sd * (float)v0;
                y[i*qk + sb*32 + j + 16] = d * sd * (float)v1;
            }
        }
    }
}

// === QUANTIZACAO REFERENCIA ===
void quantize_row_dct4_ref(const float * GGML_RESTRICT x,
                           block_dct4 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK_DCT4;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        // 1. Encontrar o valor absoluto maximo no bloco
        float amax = 0.0f;
        for (int j = 0; j < qk; j++) {
            amax = fmaxf(amax, fabsf(x[i*qk + j]));
        }

        // 2. Computar scale global
        const float d = amax / 7.0f;  // Para 4-bit signed: [-7, 7]
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = GGML_FP32_TO_FP16(d);

        // 3. Quantizar sub-blocos
        for (int sb = 0; sb < qk/32; sb++) {
            // Encontrar scale local do sub-bloco
            float sb_max = 0.0f;
            for (int j = 0; j < 32; j++) {
                sb_max = fmaxf(sb_max, fabsf(x[i*qk + sb*32 + j]));
            }
            const float sd = (d > 0) ? sb_max / d : 0.0f;
            y[i].dct_scales[sb] = (uint8_t)(sd * 255.0f);

            // Quantizar e empacotar 4-bit
            const float sd_inv = sd > 0 ? 1.0f / (d * sd) : 0.0f;
            for (int j = 0; j < 16; j++) {
                const float v0 = x[i*qk + sb*32 + j] * sd_inv;
                const float v1 = x[i*qk + sb*32 + j + 16] * sd_inv;

                const uint8_t q0 = MIN(15, (uint8_t)(v0 + 8.5f));
                const uint8_t q1 = MIN(15, (uint8_t)(v1 + 8.5f));

                y[i].qs[sb * 16 + j] = q0 | (q1 << 4);
            }
        }
    }
}

// === WRAPPER ===
void quantize_row_dct4(const float * GGML_RESTRICT x,
                       void * GGML_RESTRICT y, int64_t k) {
    quantize_row_dct4_ref(x, (block_dct4 *)y, k);
}
```

### 4.3 Padrao Real do Q4_0 (Referencia)

Para comparacao, a implementacao real do `quantize_row_q4_0_ref`:

```c
void quantize_row_q4_0_ref(const float * GGML_RESTRICT x,
                           block_q4_0 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_0;  // 32
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j] * id;
            const float x1 = x[i*qk + qk/2 + j] * id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}
```

---

## 5. Passo 4: Registrar na Tabela type_traits (ggml.c)

### 5.1 Tabela Principal

**Arquivo:** `ggml/src/ggml.c`

Adicionar entrada no array `type_traits[]`:

```c
static const struct ggml_type_traits type_traits[GGML_TYPE_COUNT] = {
    // ... entradas existentes ...

    [GGML_TYPE_Q4_0] = {
        .type_name     = "q4_0",
        .blck_size     = QK4_0,            // 32
        .type_size     = sizeof(block_q4_0), // 18
        .is_quantized  = true,
        .to_float      = (ggml_to_float_t) dequantize_row_q4_0,
        .from_float_ref = (ggml_from_float_t) quantize_row_q4_0_ref,
    },

    // ... mais entradas ...

    [GGML_TYPE_IQ2_XXS] = {
        .type_name     = "iq2_xxs",
        .blck_size     = QK_K,              // 256
        .type_size     = sizeof(block_iq2_xxs), // 66
        .is_quantized  = true,
        .to_float      = (ggml_to_float_t) dequantize_row_iq2_xxs,
        .from_float_ref = NULL,  // IQ types nao tem quantizacao de referencia simples
    },

    // === SEU NOVO TIPO ===
    [GGML_TYPE_DCT4] = {
        .type_name     = "dct4",
        .blck_size     = QK_DCT4,           // 256
        .type_size     = sizeof(block_dct4),  // 138
        .is_quantized  = true,
        .to_float      = (ggml_to_float_t) dequantize_row_dct4,
        .from_float_ref = (ggml_from_float_t) quantize_row_dct4_ref,
    },
};
```

### 5.2 Funcoes de Acesso

O GGML fornece acesso type-safe via:
```c
const struct ggml_type_traits * ggml_get_type_traits(enum ggml_type type) {
    return &type_traits[type];
}
```

Isso permite que qualquer backend consulte o tamanho do bloco, funcao de dequantizacao, etc.

---

## 6. Passo 5: Implementar o Vec Dot e Registrar em type_traits_cpu

### 6.1 Implementacao Generica do Vec Dot (quants.c)

**Arquivo:** `ggml/src/ggml-cpu/quants.c`

O vec_dot computa o dot product entre pesos quantizados e ativacoes quantizadas (tipicamente Q8_0 ou Q8_K):

```c
void ggml_vec_dot_dct4_q8_K_generic(
    int n,                          // Numero de elementos
    float * GGML_RESTRICT s,        // Resultado (float scalar)
    size_t bs,                      // Stride do resultado
    const void * GGML_RESTRICT vx,  // Pesos quantizados (DCT4)
    size_t bx,                      // Stride dos pesos
    const void * GGML_RESTRICT vy,  // Ativacoes quantizadas (Q8_K)
    size_t by,                      // Stride das ativacoes
    int nrc                         // Numero de linhas concorrentes
) {
    const int qk = QK_DCT4;  // 256
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);  // Versao basica: 1 linha por vez

    const block_dct4 * GGML_RESTRICT x = (const block_dct4 *) vx;
    const block_q8_K * GGML_RESTRICT y = (const block_q8_K *) vy;

    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d_x = GGML_FP16_TO_FP32(x[i].d);
        const float d_y = y[i].d;

        int32_t sumi = 0;

        for (int sb = 0; sb < qk/32; sb++) {
            const float sd = (float)x[i].dct_scales[sb] / 255.0f;
            int32_t sub_sum = 0;

            for (int j = 0; j < 16; j++) {
                const uint8_t packed = x[i].qs[sb * 16 + j];
                const int8_t v0 = (packed & 0x0F) - 8;
                const int8_t v1 = (packed >> 4)   - 8;

                sub_sum += v0 * y[i].qs[sb*32 + j];
                sub_sum += v1 * y[i].qs[sb*32 + j + 16];
            }
            sumi += (int32_t)(sub_sum * sd);
        }

        sumf += d_x * d_y * (float)sumi;
    }

    *s = sumf;
}
```

### 6.2 Registrar na Tabela type_traits_cpu

**Arquivo:** `ggml/src/ggml-cpu/ggml-cpu.c`

```c
static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
    // ... entradas existentes ...

    [GGML_TYPE_Q4_0] = {
        .from_float  = (ggml_from_float_t) quantize_row_q4_0,
        .vec_dot     = (ggml_vec_dot_t) ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type = GGML_TYPE_Q8_0,  // Ativacoes sao quantizadas para Q8_0
        .nrows       = 1,
    },

    // === SEU NOVO TIPO ===
    [GGML_TYPE_DCT4] = {
        .from_float   = (ggml_from_float_t) quantize_row_dct4,
        .vec_dot      = (ggml_vec_dot_t) ggml_vec_dot_dct4_q8_K_generic,
        .vec_dot_type = GGML_TYPE_Q8_K,  // Usar Q8_K para super-blocos de 256
        .nrows        = 1,
    },
};
```

**Nota sobre `vec_dot_type`:**
- `GGML_TYPE_Q8_0` (bloco de 32): usado por Q4_0, Q4_1, Q5_0, Q5_1
- `GGML_TYPE_Q8_K` (bloco de 256): usado por K-quants (Q4_K, Q5_K, Q6_K), I-quants, e TurboQuant
- O `vec_dot_type` determina como as ativacoes FP32 sao quantizadas antes do dot product

**Nota sobre `nrows`:**
- `1` = processa 1 linha por chamada (padrao para a maioria)
- `2` = processa 2 linhas simultaneamente (otimizacao ARM MATMUL para Q4_0)

---

## 7. Passo 6: Implementar Kernels SIMD Otimizados

### 7.1 Estrutura de Arquivos SIMD

Os kernels otimizados ficam em arquivos separados por arquitetura:

```
ggml/src/ggml-cpu/
  quants.c              # Implementacoes genericas (fallback scalar)
  arch/
    arm/quants.c        # ARM NEON, i8mm, SVE, SME
    x86/quants.c        # AVX2, AVX512
```

### 7.2 Padrao ARM NEON para Vec Dot

O padrao tipico para NEON em `arch/arm/quants.c`:

```c
void ggml_vec_dot_dct4_q8_K(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by,
    int nrc)
{
    const int qk = QK_DCT4;
    const int nb = n / qk;

    const block_dct4 * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    // Acumulador float32x4 (4 floats NEON)
    float32x4_t sumv = vdupq_n_f32(0.0f);

    for (int i = 0; i < nb; i++) {
        const float d_x = GGML_CPU_FP16_TO_FP32(x[i].d);
        const float d_y = y[i].d;

        for (int sb = 0; sb < qk/32; sb++) {
            // Carregar 16 bytes de dados quantizados (32 valores de 4-bit)
            const uint8x16_t qx_raw = vld1q_u8(x[i].qs + sb * 16);

            // Desempacotar nibbles
            const uint8x16_t qx_lo = vandq_u8(qx_raw, vdupq_n_u8(0x0F));  // Lower 4-bit
            const uint8x16_t qx_hi = vshrq_n_u8(qx_raw, 4);                // Upper 4-bit

            // Converter para signed e subtrair bias (8)
            const int8x16_t qx_lo_s = vsubq_s8(vreinterpretq_s8_u8(qx_lo), vdupq_n_s8(8));
            const int8x16_t qx_hi_s = vsubq_s8(vreinterpretq_s8_u8(qx_hi), vdupq_n_s8(8));

            // Carregar 32 bytes de Q8_K (ativacoes)
            const int8x16_t qy_0 = vld1q_s8(y[i].qs + sb*32);
            const int8x16_t qy_1 = vld1q_s8(y[i].qs + sb*32 + 16);

            // Dot product usando vdotq_s32 (4x int8 -> int32)
            int32x4_t acc = vdupq_n_s32(0);
            acc = vdotq_s32(acc, qx_lo_s, qy_0);   // 16 multiply-adds
            acc = vdotq_s32(acc, qx_hi_s, qy_1);    // 16 multiply-adds

            // Acumular com scale
            const float sd = (float)x[i].dct_scales[sb] / 255.0f;
            const float scale = d_x * d_y * sd;
            sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(acc), scale);
        }
    }

    // Reducao horizontal: somar os 4 lanes do float32x4
    *s = vaddvq_f32(sumv);
}
```

### 7.3 Padrao AVX2 para Vec Dot

O padrao tipico para AVX2 em `arch/x86/quants.c`:

```c
void ggml_vec_dot_dct4_q8_K(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by,
    int nrc)
{
#if defined(__AVX2__)
    const int qk = QK_DCT4;
    const int nb = n / qk;

    const block_dct4 * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    __m256 acc = _mm256_setzero_ps();  // 8 floats acumulador

    for (int i = 0; i < nb; i++) {
        const float d_x = GGML_CPU_FP16_TO_FP32(x[i].d);
        const float d_y = y[i].d;

        for (int sb = 0; sb < qk/32; sb++) {
            // Carregar 16 bytes (32 valores 4-bit empacotados)
            const __m128i qx_raw = _mm_loadu_si128((__m128i*)(x[i].qs + sb*16));

            // Desempacotar nibbles para 8-bit
            const __m128i mask_lo = _mm_set1_epi8(0x0F);
            const __m128i qx_lo = _mm_and_si128(qx_raw, mask_lo);
            const __m128i qx_hi = _mm_and_si128(_mm_srli_epi16(qx_raw, 4), mask_lo);

            // Subtrair bias (8) para obter valores signed
            const __m128i bias = _mm_set1_epi8(8);
            const __m128i qx_lo_s = _mm_sub_epi8(qx_lo, bias);
            const __m128i qx_hi_s = _mm_sub_epi8(qx_hi, bias);

            // Expandir para 256-bit para maddubs
            const __m256i qx_256 = _mm256_set_m128i(qx_hi_s, qx_lo_s);

            // Carregar 32 bytes de Q8_K
            const __m256i qy_256 = _mm256_loadu_si256((__m256i*)(y[i].qs + sb*32));

            // Multiply-add: signed * signed -> int16, depois horizontal add -> int32
            const __m256i prod = _mm256_maddubs_epi16(
                _mm256_sign_epi8(qx_256, qx_256),  // abs(x)
                _mm256_sign_epi8(qy_256, qx_256)    // sign(x) * y
            );
            const __m256i sum32 = _mm256_madd_epi16(prod, _mm256_set1_epi16(1));

            // Converter para float e escalar
            const float sd = (float)x[i].dct_scales[sb] / 255.0f;
            const float scale = d_x * d_y * sd;
            acc = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(sum32),
                _mm256_set1_ps(scale),
                acc
            );
        }
    }

    // Reducao horizontal
    const __m128 r4 = _mm_add_ps(
        _mm256_castps256_ps128(acc),
        _mm256_extractf128_ps(acc, 1)
    );
    const __m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
    const __m128 r1 = _mm_add_ss(r2, _mm_movehdup_ps(r2));
    *s = _mm_cvtss_f32(r1);
#else
    ggml_vec_dot_dct4_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}
```

### 7.4 Intrinsics Comuns

**ARM NEON:**
| Intrinsic | Funcao |
|---|---|
| `vld1q_u8(ptr)` | Carregar 16 bytes unsigned |
| `vld1q_s8(ptr)` | Carregar 16 bytes signed |
| `vandq_u8(a, b)` | AND bitwise |
| `vshrq_n_u8(a, n)` | Shift right por n bits |
| `vreinterpretq_s8_u8(a)` | Reinterpretar unsigned como signed |
| `vdotq_s32(acc, a, b)` | 4x int8 dot product -> int32 |
| `vmlaq_n_f32(acc, a, s)` | Multiply-accumulate float |
| `vaddvq_f32(a)` | Reducao horizontal float |
| `vcvtq_f32_s32(a)` | int32 -> float32 |

**x86 AVX2:**
| Intrinsic | Funcao |
|---|---|
| `_mm256_loadu_si256(ptr)` | Carregar 256 bits |
| `_mm256_maddubs_epi16(a, b)` | Multiply unsigned*signed + horizontal add |
| `_mm256_madd_epi16(a, b)` | Multiply int16 + horizontal add -> int32 |
| `_mm256_fmadd_ps(a, b, c)` | Fused multiply-add float |
| `_mm256_cvtepi32_ps(a)` | int32 -> float32 |
| `_mm256_sign_epi8(a, b)` | Negar se b < 0 (truque para signed mul) |

---

## 8. Passo 7: Integrar na Ferramenta de Quantizacao

### 8.1 Frontend (tools/quantize/quantize.cpp)

**Arquivo:** `tools/quantize/quantize.cpp`

Adicionar opcao no vetor `QUANT_OPTIONS`:

```c
static const std::vector<quant_option> QUANT_OPTIONS = {
    // ... entradas existentes ...
    { "Q4_0",    LLAMA_FTYPE_MOSTLY_Q4_0,    "4.34G, +0.4685 ppl @ Llama-3-8B" },
    { "Q4_K_M",  LLAMA_FTYPE_MOSTLY_Q4_K_M,  "4.58G, +0.0382 ppl @ Llama-3-8B" },
    // ...

    // === SEU NOVO TIPO ===
    { "DCT4",    LLAMA_FTYPE_MOSTLY_DCT4,    "N/AG, +X.XXXX ppl @ Llama-3-8B" },
};
```

### 8.2 Backend de Quantizacao (src/llama-model-quantize.cpp)

**Arquivo:** `src/llama-model-quantize.cpp`

O `llama_model_quantize_internal` e a funcao que processa cada tensor. Voce precisa adicionar:

1. **Mapeamento ftype -> ggml_type**: Na funcao que converte `llama_ftype` para `ggml_type`:
```c
case LLAMA_FTYPE_MOSTLY_DCT4:
    return GGML_TYPE_DCT4;
```

2. **Logica de quantizacao por tensor**: O llama.cpp permite regras diferentes por tensor. Por exemplo, embeddings de token sao tipicamente mantidos em maior precisao. A funcao decide qual tipo usar para cada tensor baseado no nome e no ftype.

### 8.3 Quantizacao Avancada (com importance matrix)

Se o novo tipo requer importance matrix (como IQ types), adicionar na funcao `ggml_quantize_chunk`:

**Arquivo:** `ggml/src/ggml-quants.c` ou equivalente

```c
// Funcao de quantizacao com importance matrix
size_t quantize_dct4(const float * GGML_RESTRICT src,
                     void * GGML_RESTRICT dst,
                     int64_t nrows, int64_t n_per_row,
                     const float * imatrix) {
    // imatrix: peso de importancia por elemento (pode ser NULL)
    // Usar imatrix para priorizar precisao em pesos mais importantes
    // ...
}
```

---

## 9. Passo 8: Integrar no Conversor Python (GGUF)

### 9.1 Adicionar ao Enum Python (constants.py)

**Arquivo:** `gguf-py/gguf/constants.py`

```python
class GGMLQuantizationType(IntEnum):
    F32     = 0
    F16     = 1
    Q4_0    = 2
    Q4_1    = 3
    Q5_0    = 6
    # ...
    NVFP4   = 40

    # === SEU NOVO TIPO ===
    DCT4    = 41
```

### 9.2 Implementar Quant/Dequant em Python (quants.py)

**Arquivo:** `gguf-py/gguf/quants.py`

O sistema usa heranca com registro automatico:

```python
class DCT4(__Quant, qtype=GGMLQuantizationType.DCT4):
    block_size = 256
    type_size  = 138  # sizeof(block_dct4) em bytes

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        # blocks shape: (n_blocks, 256) float32
        # return shape: (n_blocks, 138) uint8
        n_blocks = blocks.shape[0]
        result = np.zeros((n_blocks, cls.type_size), dtype=np.uint8)

        for i in range(n_blocks):
            block = blocks[i]
            # 1. Computar scale global
            amax = np.max(np.abs(block))
            d = amax / 7.0 if amax > 0 else 0.0
            # Armazenar d como FP16
            d_fp16 = np.float16(d)
            result[i, 0:2] = np.frombuffer(d_fp16.tobytes(), dtype=np.uint8)

            # 2. Quantizar sub-blocos
            for sb in range(8):
                sub = block[sb*32:(sb+1)*32]
                sb_max = np.max(np.abs(sub))
                sd = sb_max / d if d > 0 else 0.0
                result[i, 2 + sb] = int(sd * 255.0)

                # 3. Empacotar 4-bit
                inv = 1.0 / (d * sd) if d * sd > 0 else 0.0
                for j in range(16):
                    v0 = int(np.clip(sub[j] * inv + 8.5, 0, 15))
                    v1 = int(np.clip(sub[j+16] * inv + 8.5, 0, 15))
                    result[i, 10 + sb*16 + j] = v0 | (v1 << 4)

        return result

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        # blocks shape: (n_blocks, 138) uint8
        # return shape: (n_blocks, 256) float32
        n_blocks = blocks.shape[0]
        result = np.zeros((n_blocks, cls.block_size), dtype=np.float32)

        for i in range(n_blocks):
            d = np.frombuffer(blocks[i, 0:2], dtype=np.float16).item()
            for sb in range(8):
                sd = blocks[i, 2 + sb] / 255.0
                for j in range(16):
                    packed = blocks[i, 10 + sb*16 + j]
                    v0 = (packed & 0x0F) - 8
                    v1 = (packed >> 4) - 8
                    result[i, sb*32 + j] = d * sd * v0
                    result[i, sb*32 + j + 16] = d * sd * v1

        return result
```

O hook `__init_subclass__` na classe base `__Quant` registra automaticamente a subclasse no dicionario `_type_traits`, permitindo que `quantize()` e `dequantize()` encontrem a implementacao pelo ID do tipo.

### 9.3 Conversor HuggingFace (convert_hf_to_gguf.py)

**Arquivo:** `convert_hf_to_gguf.py`

Para modelos que ja estao no formato DCT4 (ex: pre-quantizados no HuggingFace):

```python
# No mapeamento de output types:
_output_type_map = {
    "f32":  gguf.GGMLQuantizationType.F32,
    "f16":  gguf.GGMLQuantizationType.F16,
    # ...
    "dct4": gguf.GGMLQuantizationType.DCT4,
}
```

Para conversao direta durante a exportacao, adicionar suporte no `prepare_tensors()`:
```python
if self.ftype == gguf.GGMLQuantizationType.DCT4:
    data = gguf.quants.quantize(data, gguf.GGMLQuantizationType.DCT4)
```

---

## 10. Passo 9: Testes

### 10.1 Teste de Funcoes de Quantizacao (test-quantize-fns.cpp)

**Arquivo:** `tests/test-quantize-fns.cpp`

Este teste verifica:
1. `quantize_row_*` e `dequantize_row_*` sao inversos (dentro de tolerancia)
2. Erro de quantizacao esta dentro de limites aceitaveis
3. Consistencia entre implementacao de referencia e otimizada

O novo tipo e automaticamente incluido se `ggml_type_traits[type].is_quantized == true`.

### 10.2 Teste de Operacoes de Backend (test-backend-ops.cpp)

**Arquivo:** `tests/test-backend-ops.cpp`

Testa operacoes como mul_mat com tensores quantizados:
- Valida que a dequantizacao produz erro NMSE aceitavel
- Testa com e sem importance matrix (para IQ types)
- Executa em todos os backends disponveis

### 10.3 Teste de Perplexidade

A forma primaria de validar qualidade e medir perplexidade no WikiText-2:

```bash
# 1. Quantizar modelo
./build/bin/llama-quantize modelo-f16.gguf modelo-dct4.gguf DCT4

# 2. Medir perplexidade
./build/bin/llama-perplexity -m modelo-dct4.gguf \
    -f wikitext-2-raw/wiki.test.raw \
    --ctx-size 512 --batch-size 512

# 3. Comparar com tipo de referencia (Q4_K_M)
./build/bin/llama-perplexity -m modelo-q4km.gguf \
    -f wikitext-2-raw/wiki.test.raw \
    --ctx-size 512 --batch-size 512
```

### 10.4 KL Divergence

Alem da perplexidade, o llama.cpp pode calcular KL divergence entre distribuicoes logit FP16 vs quantizado:

```bash
./build/bin/llama-perplexity -m modelo-dct4.gguf \
    -f wikitext-2-raw/wiki.test.raw \
    --kl-divergence-base modelo-f16.gguf
```

Metricas adicionais:
- Razao de perplexidade media FP16/quantizado
- RMS das mudancas de probabilidade de tokens
- KL = 0 indica distribuicoes identicas

### 10.5 Compilar e Executar Testes

```bash
# Build
cmake -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build -j$(nproc)

# Executar testes unitarios
./build/bin/test-quantize-fns
./build/bin/test-backend-ops

# Listar tipos disponiveis
./build/bin/llama-quantize --help
```

---

## 11. Passo 10: Alteracoes no Build System

### 11.1 Camada ggml-base (ggml/src/CMakeLists.txt)

Se voce criou um novo arquivo `.c` para funcoes de quantizacao:

```cmake
# Em ggml/src/CMakeLists.txt
target_sources(ggml-base PRIVATE
    ggml.c
    ggml-quants.c
    # Se separou em arquivo proprio:
    ggml-quants-dct4.c    # NOVO
)
```

**Na maioria dos casos**, nao e necessario um arquivo separado. Adicionar as funcoes diretamente em `ggml-quants.c` e suficiente.

### 11.2 Camada ggml-cpu (ggml/src/ggml-cpu/CMakeLists.txt)

Se adicionou kernels SIMD em arquivos separados:

```cmake
# Em ggml/src/ggml-cpu/CMakeLists.txt
target_sources(ggml-cpu PRIVATE
    ggml-cpu.c
    quants.c
    # Os arquivos arch/ geralmente ja sao incluidos via glob
)
```

Os arquivos em `arch/arm/quants.c` e `arch/x86/quants.c` geralmente ja sao incluidos automaticamente pelo build system.

### 11.3 Build com Variantes de ISA

Quando `GGML_CPU_ALL_VARIANTS=ON`, o build gera multiplas bibliotecas CPU otimizadas (uma para AVX2, uma para AVX512, uma para NEON, etc.). As funcoes nos arquivos `arch/` sao compiladas com as flags corretas automaticamente via `ggml_add_cpu_backend_variant()`.

---

## 12. Codificacao por Entropia (rANS) como Camada de Pre-Processamento

### 12.1 Contexto e Oportunidade

**Dados empiricos de entropia em pesos quantizados:**

| Quantizacao | Bits Armazenados | Entropia Medida | Compressao Potencial |
|---|---|---|---|
| 8-bit | 8 | ~4-5 bits | 1.6-2x |
| 4-bit | 4 | ~2.17 bits | 1.84x |
| 3-bit | 3 | ~1.8 bits | 1.7x |
| 2-bit | 2 | ~1.5 bits | 1.3x |

Fonte: Medicoes em modelos Llama e Qwen. Os pesos quantizados concentram-se em poucos valores (tipicamente perto de zero), resultando em entropia significativamente abaixo do bit-width nominal.

**Implicacao:** Aplicar rANS pode comprimir pesos 4-bit por ate ~3.27x adicionalmente (lossless), conforme demonstrado pelo projeto ECQ.

### 12.2 Estado Atual: Nenhuma Implementacao no llama.cpp

Ate marco de 2026, **nao existe codificacao por entropia integrada ao llama.cpp**. Esforcos existentes:

- **ECQ (github.com/drxddy/ecq):** Prototipo de pesquisa usando rANS sobre pesos 4-bit. Resultados: 3.27x compressao adicional em Qwen2.5, GPT-2, SmolLM. Focado em Apple Silicon com design de kernel Metal.
- **Feature request MLX #3043:** Proposta detalhada para rANS em mlx com API `EntropyCodedLinear`. Fechada, desenvolvida como extensao standalone.
- **ik_llama.cpp:** Fork com trellis quants, sem codificacao por entropia.

### 12.3 Arquitetura Proposta para Integracao

A integracao de rANS no llama.cpp pode seguir duas estrategias:

**Estrategia A -- Tipo de quantizacao com decodificacao embutida:**
```
Arquivo GGUF:  [header rANS] [stream rANS comprimido]
                       |
                       v
Carregamento:  rANS decode -> block_q4_K padrao na memoria
                       |
                       v
Inferencia:    vec_dot_q4_K padrao (sem mudanca)
```

Vantagens: reutiliza todos os kernels existentes, sem impacto em performance de inferencia.
Desvantagens: nao economiza VRAM (decodifica na carga), apenas tamanho de arquivo.

**Estrategia B -- Decodificacao fused durante inferencia:**
```
Arquivo GGUF:  [stream rANS comprimido] (armazenado na VRAM)
                       |
                       v
Inferencia:    rANS decode on-the-fly -> dequant + vec_dot
```

Vantagens: economiza VRAM (dados comprimidos na memoria).
Desvantagens: overhead de decodificacao por token, complexidade de implementacao.

**Estrategia C (recomendada) -- Tipo hibrido:**
```
GGML_TYPE_DCT4_RANS = 42

Carregamento:
  1. Ler stream rANS do GGUF
  2. Decodificar para block_dct4 padrao
  3. Armazenar decodificado na VRAM

Quantizacao:
  1. Quantizar float32 -> block_dct4
  2. Codificar com rANS -> stream comprimido
  3. Salvar stream no GGUF
```

### 12.4 Implementacao do Codificador rANS

O rANS (range Asymmetric Numeral Systems) opera por-linha da matriz de pesos, permitindo decodificacao paralela por threadgroup:

```c
// Estrutura do stream rANS para um tensor
typedef struct {
    uint32_t n_rows;           // Numero de linhas
    uint32_t n_cols;           // Numero de colunas
    uint32_t freq_table[16];   // Tabela de frequencias (para 4-bit: 16 simbolos)
    uint32_t row_offsets[];    // Offset de cada linha no stream
    // Seguido pelo stream rANS comprimido
} rans_tensor_header;
```

**Codificacao (offline, durante quantizacao):**
```c
void rans_encode_tensor(const block_dct4 * blocks, int n_blocks,
                        uint8_t * out_stream, size_t * out_size) {
    // 1. Extrair simbolos (nibbles 4-bit) de todos os blocos
    // 2. Computar tabela de frequencias
    // 3. Codificar com rANS (backward pass)
    // 4. Escrever header + stream
}
```

**Decodificacao (durante carregamento do modelo):**
```c
void rans_decode_tensor(const uint8_t * stream, size_t stream_size,
                        block_dct4 * out_blocks, int n_blocks) {
    // 1. Ler header e tabela de frequencias
    // 2. Decodificar rANS (forward pass) por linha
    // 3. Reconstruir blocos quantizados
}
```

### 12.5 Modificacoes Necessarias para rANS

Para integrar rANS como pre-processamento, os seguintes pontos de modificacao:

1. **Novo tipo no enum** (se usar tipo separado):
   ```c
   GGML_TYPE_DCT4_RANS = 42  // ou reutilizar metadado no GGUF
   ```

2. **Carregamento do modelo** (`src/llama-model.cpp`):
   - Detectar tensor com tipo rANS
   - Decodificar stream rANS para formato quantizado padrao
   - Alocar buffer com tipo base (DCT4) e tamanho decodificado

3. **Ferramenta de quantizacao** (`tools/quantize/quantize.cpp`):
   - Apos quantizar tensor, aplicar codificacao rANS
   - Salvar stream comprimido no GGUF

4. **Formato GGUF**:
   - Opcao A: tipo GGML separado (ex: GGML_TYPE_DCT4_RANS)
   - Opcao B: metadado extra no tensor indicando compressao rANS
   - Opcao C: novo campo no header GGUF para compressao de tensor

### 12.6 Alternativa Mais Simples: Compressao Transparente no GGUF

Uma abordagem mais pragmatica e adicionar rANS como camada de compressao **transparente** no formato GGUF, sem criar um novo GGML type:

```
GGUF v4 (proposta):
  tensor_info {
    name: "blk.0.attn_q.weight"
    type: GGML_TYPE_Q4_K       // Tipo base inalterado
    compression: GGUF_COMPRESSION_RANS  // NOVO campo
    compressed_size: 52428800   // Tamanho comprimido
    decompressed_size: 73400320 // Tamanho original
    rans_freq_table: [...]      // Tabela de frequencias
  }
```

Vantagens:
- Nao requer novo GGML type
- Todos os kernels existentes funcionam sem modificacao
- Economia de tamanho de arquivo e largura de banda de download
- Decodificacao transparente durante carregamento

---

## 13. Referencia: Checklist Completo de Arquivos

### Minimo Viavel (apenas CPU, sem SIMD otimizado):

| # | Arquivo | Alteracao |
|---|---|---|
| 1 | `ggml/include/ggml.h` | Adicionar `GGML_TYPE_DCT4` no enum, atualizar `GGML_TYPE_COUNT` |
| 2 | `ggml/src/ggml-common.h` | Definir `block_dct4` struct + `static_assert` |
| 3 | `ggml/src/ggml-quants.h` | Declarar funcoes `quantize_row_dct4*`, `dequantize_row_dct4` |
| 4 | `ggml/src/ggml-quants.c` | Implementar funcoes de referencia (scalar) |
| 5 | `ggml/src/ggml.c` | Adicionar entrada em `type_traits[]` |
| 6 | `ggml/src/ggml-cpu/ggml-cpu.c` | Adicionar entrada em `type_traits_cpu[]` |
| 7 | `ggml/src/ggml-cpu/quants.c` | Implementar `ggml_vec_dot_dct4_q8_K_generic` |
| 8 | `include/llama.h` | Adicionar `LLAMA_FTYPE_MOSTLY_DCT4` |
| 9 | `src/llama-model-quantize.cpp` | Mapeamento ftype -> ggml_type |
| 10 | `tools/quantize/quantize.cpp` | Adicionar em `QUANT_OPTIONS` |

### Para Suporte Completo (adicionar apos MVP):

| # | Arquivo | Alteracao |
|---|---|---|
| 11 | `ggml/src/ggml-cpu/arch/arm/quants.c` | Kernel NEON otimizado |
| 12 | `ggml/src/ggml-cpu/arch/x86/quants.c` | Kernel AVX2 otimizado |
| 13 | `ggml/src/ggml-cpu/arch-fallback.h` | Macros de fallback para novas funcoes |
| 14 | `gguf-py/gguf/constants.py` | `GGMLQuantizationType.DCT4 = 41` |
| 15 | `gguf-py/gguf/quants.py` | Classe `DCT4(__Quant)` com quantize/dequantize |
| 16 | `convert_hf_to_gguf.py` | Suporte ao tipo no conversor |
| 17 | `ggml/src/ggml-cuda/` | Kernels CUDA (PR separado) |
| 18 | `ggml/src/ggml-metal/` | Kernels Metal (PR separado) |

---

## 14. Referencia: PRs Reais como Modelo

### 14.1 TurboQuant (TBQ3_0 / TBQ4_0)

**Commit:** `mudler/llama.cpp@dee102d`
**Arquivos modificados (6):**
1. `ggml/include/ggml.h` -- GGML_TYPE_TBQ3_0 = 41, GGML_TYPE_TBQ4_0 = 42
2. `ggml/src/CMakeLists.txt` -- Build configuration
3. `ggml/src/ggml-common.h` -- block_tbq3_0, block_tbq4_0 structs
4. `ggml/src/ggml-cpu/arch-fallback.h` -- Macros de fallback
5. `ggml/src/ggml-cpu/ggml-cpu.c` -- Registro em type_traits_cpu
6. `ggml/src/ggml-cpu/quants.c` -- quantize_row_tbq*, ggml_vec_dot_tbq*

**Abordagem:** SRHT random rotation + Max-Lloyd codebook quantization. Block sizes de 256. Vec_dot generico com decodificacao para float seguida de dot product com Q8_K.

### 14.2 NVFP4 (PR #19769)

**Abordagem faseada:**
- Fase 1 (PR principal): Tipo base + CPU scalar + ARM NEON
- Fase 2 (PRs separados): CUDA, Metal, Vulkan, x86 SIMD

**Arquivos core modificados:**
- `ggml-common.h` -- Struct block_nvfp4, UE4M3 lookup tables
- `ggml-quants.h/c` -- Funcoes de quantizacao
- `ggml-cpu/ggml-cpu.c` -- Registro CPU
- `ggml-cpu/quants.c` -- Fallback generico
- `ggml-cpu/arch/arm/quants.c` -- Kernel NEON (3.1x speedup)
- `gguf-py/` -- Suporte Python completo
- `convert_hf_to_gguf.py` -- Deteccao de ModelOpt NVFP4

**Licao importante:** Subnormals no formato FP4 causaram underflow catastrofico ate serem corrigidos, reduzindo PPL de 5.8M para 15.25.

### 14.3 IQ2_XXS / IQ2_XS (PR #4897)

**Abordagem:** Quantizacao baseada em importance matrix + lookup table (grid).
- Quantizacao sem importance matrix produz "garbage"
- Quantizacao extremamente lenta (~5 min para 7B no M2 Max)
- `from_float_ref = NULL` no type_traits (nao tem quantizacao simples)
- Funcao `quantize_iq2_xxs` separada que requer imatrix

---

## 15. Fontes e Referencias

### Repositorio Principal
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

### PRs e Commits de Referencia
- [PR #4897: 2-bit quantizations (IQ2_XXS/IQ2_XS)](https://github.com/ggml-org/llama.cpp/pull/4897)
- [PR #4856: SOTA 2-bit quants part 2](https://github.com/ggml-org/llama.cpp/pull/4856)
- [PR #19769: NVFP4 quantization type](https://github.com/ggml-org/llama.cpp/pull/19769)
- [Commit dee102d: TurboQuant support](https://github.com/mudler/llama.cpp/commit/dee102db1bfd723c91f67138b8018ce35a6be477)
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)

### Documentacao Tecnica
- [DeepWiki: Quantization Techniques](https://deepwiki.com/ggml-org/llama.cpp/6.3-quantization-techniques)
- [DeepWiki: GGML Tensor Library](https://deepwiki.com/ggml-org/llama.cpp/4-ggml-tensor-library)
- [DeepWiki: Build System](https://deepwiki.com/ggml-org/llama.cpp/8.1-build-system)
- [DeepWiki: whisper.cpp Quantization](https://deepwiki.com/ggml-org/whisper.cpp/5.2-quantization)
- [DeepWiki: GGML org](https://deepwiki.com/ggml-org/ggml)
- [K-quants implementation details](https://haroldbenoit.com/notes/ml/llms/quantization/llama.cpp/k-quants-implementation)
- [GGUF Quantization from Theory to Practice](https://atalupadhyay.wordpress.com/2025/08/24/gguf-quantization-from-theory-to-practice/)
- [GGML File Structure Guide](https://www.abhik.ai/articles/ggml-structure)

### Discussoes da Comunidade
- [Discussion #5063: Even more quantization types?](https://github.com/ggml-org/llama.cpp/discussions/5063)
- [Discussion #1796: How to create new quantization formats](https://github.com/ggml-org/llama.cpp/discussions/1796)
- [Discussion #20969: TurboQuant](https://github.com/ggml-org/llama.cpp/discussions/20969)

### Codificacao por Entropia
- [ECQ: Entropy Coded Quantization](https://github.com/drxddy/ecq)
- [MLX Issue #3043: rANS entropy-coded quantization](https://github.com/ml-explore/mlx/issues/3043)
- [FOSDEM 2025: History and advances of quantization in llama.cpp](https://archive.fosdem.org/2025/schedule/event/fosdem-2025-5991-history-and-advances-of-quantization-in-llama-cpp/)

### Avaliacoes de Quantizacao
- [Unified Evaluation of llama.cpp Quantization (arxiv)](https://arxiv.org/html/2601.14277v1)
- [FOSDEM 2026: Adventures in Model Quantization](https://fosdem.org/2026/schedule/event/QHBKKK-adventures-in-model-quantization/)
