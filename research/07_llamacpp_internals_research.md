# Pesquisa: Internals do llama.cpp e Integração do Formato DCT

## 1. Estrutura de Quantização no llama.cpp

### Tipos de Quantização Registrados

O llama.cpp define todos os tipos de quantização como enums em `ggml.h`. Cada tipo recebe um ID numérico único:

```c
GGML_TYPE_Q4_0  = 2,
GGML_TYPE_Q4_1  = 3,
GGML_TYPE_Q4_K  = 12,
GGML_TYPE_Q5_K  = 13,
GGML_TYPE_Q6_K  = 14,
GGML_TYPE_Q8_K  = 15,
GGML_TYPE_IQ2_XXS = 16,
// ...
GGML_TYPE_TBQ3_0  = 41,  // TurboQuant 3-bit (recente)
GGML_TYPE_TBQ4_0  = 42,  // TurboQuant 4-bit (recente)
```

Cada tipo também tem um `GGML_FTYPE` correspondente para o formato de arquivo.

### Estrutura de Blocos

Todos os tipos de quantização operam em **blocos de 256 elementos** (QK_K = 256, o "super-bloco"). Dentro do super-bloco, sub-blocos de 32 elementos são comuns.

Exemplo - **Q4_K** (4-bit K-quant):
```c
typedef struct {
    union {
        struct {
            ggml_half d;       // super-block scale (FP16)
            ggml_half dmin;    // super-block minimum (FP16)
        };
        ggml_half2 dm;
    };
    uint8_t scales[12];        // scales and mins (quantizados a 6-bit)
    uint8_t qs[QK_K/2];       // 4-bit quants (128 bytes)
} block_q4_K;
// Total: 144 bytes para 256 pesos = 4.5 bits/peso
```

Hierarquia K-quant:
- **Super-bloco** (256 pesos): contém scale `d` e minimum `dmin` em FP16
- **Sub-blocos** (32 pesos cada): scales e mins quantizados a 6-bit, armazenados em `scales[]`
- **Pesos individuais**: quantizados a N bits, armazenados em `qs[]`

### I-Quants (Importance Matrix)

Os I-quants usam uma abordagem fundamentalmente diferente:

```c
typedef struct {
    ggml_half d;               // scale
    uint16_t extra;            // flags para sub-blocos de alta importância
    uint8_t  qs[QK_K/4];      // codebook indices
    uint8_t  qh[QK_K/32];     // bits extras
    uint8_t  scales[QK_K/32]; // scales por sub-bloco
} block_iq3_s;
```

- Usam **codebooks fixos** (tabelas de lookup otimizadas offline)
- A `importance matrix` (imatrix) guia quais pesos recebem mais precisão
- Codebooks são arrays `const` compilados diretamente no binário

### TurboQuant (Exemplo recente de novo tipo)

O TurboQuant (TBQ3_0 e TBQ4_0) foi adicionado recentemente, mostrando o padrão para novos tipos:

```c
// Bloco TBQ3_0: 3.0625 bits/peso
typedef struct {
    uint8_t qs[QK_K * 3 / 8];  // 96 bytes: índices 3-bit
    ggml_half d;                 // norma L2
} block_tbq3_0;

// Bloco TBQ4_0: 4.0625 bits/peso
typedef struct {
    uint8_t qs[QK_K / 2];       // 128 bytes: índices 4-bit
    ggml_half d;                 // norma L2
} block_tbq4_0;
```

---

## 2. Formato GGUF

### Estrutura do Arquivo

```
[GGUF Header]
  - magic: "GGUF" (4 bytes)
  - version: uint32
  - n_tensors: uint64
  - n_kv: uint64

[Key-Value Metadata]
  - architecture, context length, vocab size, etc.
  - Cada tensor: name, dimensions, type (enum), offset

[Tensor Data]
  - Tensores armazenados sequencialmente
  - Alinhamento de 32 bytes entre tensores
  - SEM compressão adicional (sem zlib, sem zstd)
```

**Achado crítico**: O GGUF **não usa nenhuma forma de entropy coding**. Os dados quantizados são armazenados bit-packed, sem compressão. O gap de entropia identificado na pesquisa 06 (45-72%) está totalmente inexplorado.

### Memory Mapping

O llama.cpp usa `mmap()` para mapear o arquivo GGUF diretamente na memória:
- Não copia dados -- acessa diretamente do mmap
- Isso implica que os dados devem ser **random-accessible**
- Entropy coding (ANS/Huffman) quebraria o random access naive
- Solução: entropy coding por bloco com offsets pré-computados

---

## 3. Hot Path: Dequantização + Dot Product

### Operação Crítica

A operação mais chamada durante inferência é `vec_dot`:
```c
// Para Q4_K:
ggml_vec_dot_q4_K_q8_K(n, result, vx, vy, nrc)
// vx = pesos quantizados (Q4_K)
// vy = ativações quantizadas (Q8_K)
// result = dot product
```

A dequantização é **fundida** com a multiplicação -- pesos nunca são dequantizados para FP32 separadamente. Isso é crucial para performance (memory-bandwidth bound).

### Implementações SIMD

Cada tipo de quantização tem implementações otimizadas para:
- **ARM NEON**: `ggml-cpu-aarch64.cpp` (Apple Silicon, Android)
- **x86 AVX2/AVX-512**: `ggml-cpu-avx2.cpp`, `ggml-cpu-avx512.cpp`
- **Fallback genérico**: `arch-fallback.h`

O padrão para cada bloco:
```
1. Carregar bloco quantizado (qs, scales, d)
2. Dequantizar inline (shift, mask, multiply by scale)
3. Multiplicar com ativação Q8_K
4. Acumular resultado
```

### Performance

Inferência é **memory-bandwidth bound**, não compute-bound:
- Ler pesos da memória é o gargalo (não os FLOPs)
- Menos bits = menos bytes lidos = mais rápido
- A dequantização é "grátis" em termos de tempo (escondida pela latência de memória)
- Exceção: dequantização muito complexa (ex: MLP neural) pode se tornar o gargalo

---

## 4. Como Adicionar um Novo Tipo de Quantização

### Arquivos a Modificar (6 arquivos)

1. **`ggml/include/ggml.h`**
   - Adicionar enum `GGML_TYPE_DCT_XX`
   - Adicionar `GGML_FTYPE_MOSTLY_DCT_XX`
   - Incrementar `GGML_TYPE_COUNT`

2. **`ggml/src/ggml-common.h`**
   - Definir `block_dct_xx` struct com layout de memória

3. **`ggml/src/ggml-cpu/quants.c`**
   - Implementar `quantize_row_dct_xx()` -- float → formato comprimido
   - Implementar `dequantize_row_dct_xx()` -- formato → float
   - Implementar `ggml_vec_dot_dct_xx_q8_K()` -- dot product fundido

4. **`ggml/src/ggml-cpu/ggml-cpu.c`**
   - Registrar type traits:
   ```c
   [GGML_TYPE_DCT_XX] = {
       .from_float = quantize_row_dct_xx,
       .vec_dot = ggml_vec_dot_dct_xx_q8_K,
       .vec_dot_type = GGML_TYPE_Q8_K,
       .nrows = 1,
   }
   ```

5. **`ggml/src/ggml-cpu/arch-fallback.h`**
   - Adicionar macros de fallback genérico

6. **`ggml/src/CMakeLists.txt`**
   - Atualizar se necessário (normalmente não para novo tipo)

### Design Universal

- Todos os tipos usam **Q8_K como tipo de referência** para dot products
- O input (ativações) é sempre quantizado para Q8_K primeiro
- O `vec_dot` computa produto escalar entre pesos em formato X e ativações em Q8_K

---

## 5. Desafios Específicos para o Formato DCT

### 5.1 Delta Coding vs Random Access

**Problema**: Delta coding é sequencial por natureza. Para reconstruir camada N, precisa decodificar desde o keyframe anterior. O llama.cpp usa mmap + acesso direto.

**Soluções**:
- Keyframes frequentes (a cada 4-8 camadas) limitam o "lookback" máximo
- Pré-decodificar todas as camadas no load time (one-time cost)
- Cache de camadas decodificadas em memória

**Impacto**: Para um modelo de 32 camadas com keyframe a cada 8: max 7 deltas para reconstruir = ~10ms no load. Aceitável.

### 5.2 Transformada de Frequência no Hot Path

**Problema**: DCT/IDCT no hot path da inferência adicionaria latência.

**Soluções**:
- **Opção A**: Decodificar DCT no load time, armazenar pesos reconstruídos em memória. Overhead de memória mas sem impacto na inferência.
- **Opção B**: Fundir IDCT parcial com vec_dot (complexo mas possível para DCT 1D por linha).
- **Opção C**: Formato "DCT-aware" onde o dot product opera diretamente nos coeficientes DCT (requer reformulação matemática -- potencialmente pesquisável).

**Recomendação**: Opção A para primeiro protótipo. O tamanho em disco é menor (DCT + entropy coded), mas em memória os pesos são "expandidos" para formato dequantizado.

### 5.3 Entropy Coding (rANS)

**Problema**: Dados entropy-coded não são random-accessible byte-a-byte.

**Soluções**:
- Entropy coding **por bloco** (256 ou 512 pesos) com tabela de offsets
- Decodificar todos os blocos no load time
- rANS decodifica a ~3 GB/s com AVX2 → modelo de 2 GB em ~0.7s

**Impacto**: Adiciona < 1 segundo ao tempo de carga. Reduz tamanho em disco em 30-50%.

### 5.4 Neural Dequantizer no Kernel

**Problema**: MLP no hot path é ~500x mais lento que table lookup.

**Solução**: Conforme pesquisa 05, treinar com MLP mas **materializar como LUT expandida** para deploy:
- Treinar MLP: `code + context → weight`
- Exportar: para cada combinação (code, context_bucket), computar `weight = MLP(code, context)`
- Deploy: usar a LUT expandida, identica a codebooks tradicionais

**Impacto**: Zero overhead em inferência. O MLP serve apenas como método de design do codebook.

### 5.5 Unsloth Dynamic Quantization

A Unsloth Dynamic 2.0 **não é um fork** do llama.cpp. Ela usa os mesmos tipos de quantização padrão (Q4_K_M, IQ2_XXS, etc.), mas **escolhe tipos diferentes por camada** durante a conversão para GGUF. O arquivo GGUF resultante é 100% compatível com llama.cpp padrão.

A inovação é na **seleção de tipo por camada**, não no formato. Nosso DCT poderia adotar abordagem similar: usar tipos GGUF existentes para keyframes, e um tipo novo para deltas.

---

## 6. Estratégia de Integração Proposta

### Fase 1: Protótipo Puro Python (sem llama.cpp)
- Implementar pipeline completo em Python/PyTorch
- Validar compressão e qualidade
- Benchmark contra GGUF usando lm-eval-harness

### Fase 2: Formato Híbrido
- Salvar em GGUF usando tipos existentes (Q4_K para keyframes, Q2_K para deltas)
- Delta decoding em script Python pré-processador
- Resultado: GGUF padrão, compatível com llama.cpp existente, mas menor que naive

### Fase 3: Tipo Nativo GGUF
- Registrar `GGML_TYPE_DCT_2` e `GGML_TYPE_DCT_3` (2-bit e 3-bit DCT)
- Block struct:
  ```c
  typedef struct {
      uint8_t qs[QK_K * N / 8];   // quantized DCT coefficients
      ggml_half d;                  // scale
      uint8_t flags;                // is_keyframe, transform_type
      uint8_t keyframe_ref;         // relative index to keyframe
  } block_dct_N;
  ```
- Implementar `vec_dot_dct_N_q8_K` com decompressão inline
- Contribuir como PR ao llama.cpp

### Fase 4: Entropy Coding Integrado
- Adicionar camada rANS ao GGUF loader
- Decodificar entropy coding uma vez no load time
- Comprimir o arquivo GGUF em 30-50% adicional

---

## 7. Estimativa de Tamanho Final

Para o Qwen 3.5 4B (8.42 GB em BF16):

| Método | Tamanho Est. | bpw | Notas |
|--------|-------------|-----|-------|
| Q4_K_M (baseline) | 2.71 GB | 4.83 | GGUF padrão |
| IQ2_M | 1.76 GB | 2.76 | Melhor GGUF atual low-bit |
| DCT (delta+Q3 keyframes+Q2 deltas) | ~1.4 GB | ~2.2 | Sem entropy coding |
| DCT + entropy coding | ~0.9-1.1 GB | ~1.5-1.7 | Com rANS |
| DCT + entropy + neural codebook | ~0.7-0.9 GB | ~1.2-1.5 | Máxima compressão |

**Projeção conservadora**: 1.0-1.4 GB com qualidade comparável a IQ2_M (1.76 GB).
**Projeção otimista**: 0.7-0.9 GB com qualidade utilizável.

---

## 8. Referências

- [llama.cpp quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [GGUF Format Specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [TurboQuant Discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [Unified Evaluation of llama.cpp Quantization](https://arxiv.org/html/2601.14277v1)
- [Q4_K Adaptation Discussion #6760](https://github.com/ggml-org/llama.cpp/discussions/6760)
- [Even more quantization types? Discussion #5063](https://github.com/ggml-org/llama.cpp/discussions/5063)
