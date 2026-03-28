# Pesquisa 12: Operacoes Quantizadas em PyTorch e Ecossistema

> **Objetivo**: Investigar como bibliotecas existentes implementam camadas lineares quantizadas,
> matmul quantizada e desquantizacao rapida -- e o que podemos aplicar ao nosso `QuantizedLinear`.

---

## 1. bitsandbytes: Linear4bit e Linear8bitLt

### 1.1 Armazenamento de Pesos

**Linear4bit** (Params4bit):
- Os pesos sao armazenados como objetos `Params4bit`, uma subclasse customizada de `torch.nn.Parameter`
- O tensor quantizado fica em formato `uint8` (ou nf4), acompanhado de metadados de quantizacao (`QuantState`)
- O `QuantState` contem: valores absmax por bloco, codebook de quantizacao, tamanho do bloco, e informacoes de dtype
- Dois tipos de quantizacao 4-bit sao suportados:
  - **NF4** (Normal Float 4): bins com area igual sob distribuicao normal N(0,1) -- otimo para pesos neurais que seguem distribuicao gaussiana
  - **FP4** (Float Point 4): bins espacados logaritmicamente, semelhante a formato de ponto flutuante padrao
- Subclasses de conveniencia: `LinearNF4` e `LinearFP4`

**Linear8bitLt** (Int8Params):
- Pesos armazenados como `Int8Params` (int8) com buffers de escala acompanhantes
- Componentes armazenados:
  - `CB` (Code Book): pesos quantizados int8 com shape `(out_features, in_features)`
  - `SCB` (Scale Code Book): fatores de escala float por linha, shape `(out_features,)`
  - `weight_format`: identificador uint8 do layout (0 = row-major)
- A quantizacao acontece de forma **lazy**: pesos ficam em FP16/BF16 na CPU ate que `.to("cuda")` seja chamado, ativando `bnb.functional.int8_vectorwise_quant()`

**Quantizacao Dupla** (Double Quantization):
- O parametro `compress_statistics` habilita quantizacao aninhada dos proprios valores absmax
- Reduz overhead de memoria dos metadados de quantizacao, economizando ~50% adicional nos metadados

### 1.2 Forward Pass

**Linear4bit -- Passos do forward:**
1. Recuperacao de estado: `fix_4bit_weight_quant_state_from_module()` restaura metadados se necessario
2. Otimizacao CPU: converte formato empacotado para CPU quando as condicoes sao atendidas
3. Casting de dtype: garante que bias corresponda ao dtype do input
4. Selecao de tipo de computacao: determina automaticamente a precisao baseada no tipo do input
5. Chamada a `bnb.matmul_4bit()` com pesos quantizados -- **nao desquantiza completamente**, opera direto nos dados quantizados + estado de quantizacao

**Linear8bitLt -- Forward com deteccao de outliers (LLM.int8()):**
1. Inicializacao de estado: na primeira chamada, transfere buffers CB/SCB para `MatmulLtState`
2. Casting de bias para corresponder ao dtype do input
3. **Deteccao de outliers** (quando `threshold > 0.0`):
   - Colunas com valores de magnitude alta (ex: > 3.0 desvios padrao) sao marcadas como outliers
   - Outliers sao processados em FP16 para manter precisao
   - Colunas restantes usam int8
4. Chamada a `bnb.matmul()` que despacha para kernels 8-bit apropriados
5. Resultados das duas vias (FP16 outliers + int8 normais) sao combinados

### 1.3 Performance CPU vs GPU

- **GPU**: kernels CUDA otimizados para desquantizacao + matmul. Performance excelente.
- **CPU**: suporte existe para arquiteturas com AVX512_BF16
  - Em CPUs com AVX512-BF16, pesos 4-bit sao re-empacotados em formato otimizado para CPU
  - Porem, **sem GPU a inferencia e extremamente lenta** -- a biblioteca depende de kernels CUDA para eficiencia
  - O re-empacotamento CPU persiste entre chamadas forward, mas e convertido automaticamente ao mover para GPU

**Relevancia para nosso projeto:** Nossa `QuantizedLinear` faz desquantizacao em Python puro (loop por linha), o que e muito mais lento. Precisariamos de kernels C++ ou Metal para competir.

Sources:
- [bitsandbytes nn/modules.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/nn/modules.py)
- [Linear8bitLt DeepWiki](https://deepwiki.com/bitsandbytes-foundation/bitsandbytes/4.2-linear8bitlt-module)
- [4-bit Quantized Layers DeepWiki](https://deepwiki.com/bitsandbytes-foundation/bitsandbytes/4.1-4-bit-quantized-layers)
- [bitsandbytes GitHub](https://github.com/bitsandbytes-foundation/bitsandbytes)

---

## 2. torch.ao.quantization e Quantizacao Nativa do PyTorch

### 2.1 API Legada (torch.ao.quantization)

**AVISO: DEPRECIADO.** O `torch.ao.quantization` esta depreciado e sera removido no PyTorch 2.10+. A migracao recomendada e para o `torchao` (pytorch/ao).

**quantize_per_tensor:**
- Converte um tensor float para tensor quantizado com scale e zero_point dados
- Dtypes suportados: `torch.quint8`, `torch.qint8`, `torch.qint32`
- **NAO suporta QInt4 nativamente** -- int4 e tratado apenas pelo torchao

**Quantizacao Dinamica:**
```python
torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```
- Pesos quantizados antecipadamente, ativacoes quantizadas dinamicamente durante inferencia
- Ideal para modelos dominados por carregamento de pesos (LSTM, Transformers com batch pequeno)
- Scales computados em runtime

**Quantizacao Estatica:**
- Requer calibracao com dados representativos
- Scales gravados ("baked in") -- melhor performance que dinamica
- Backends: `fbgemm` (x86), `qnnpack` (ARM), `onednn` (x86)

**Quantization-Aware Training (QAT):**
- Melhor precisao: aprende scales otimos durante treinamento
- Usa fake quantization durante treinamento

### 2.2 Limitacoes para Nosso Caso

- `quantize_per_tensor` so suporta 8-bit e 32-bit, **nao 4-bit**
- Operacoes quantizadas esperem dtypes especificos (quint8 para ativacoes no fbgemm)
- O sistema de backends (fbgemm/qnnpack) e rigido e voltado para modelos INT8
- **Conclusao: torch.ao.quantization NAO e util para nosso QuantizedLinear INT4**

### 2.3 Caminho de Migracao

A migracao oficial e:
- Eager mode: `torch.ao.quantization.quantize` -> torchao `quantize_()` API
- FX graph mode: `prepare_fx/convert_fx` -> torchao `prepare_pt2e/convert_pt2e`
- Tensor subclasses (Int4Tensor, Int8Tensor, Float8Tensor) sao o novo paradigma

Sources:
- [PyTorch Quantization Docs](https://docs.pytorch.org/docs/stable/quantization.html)
- [quantize_per_tensor](https://docs.pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html)
- [torch.ao.quantization deprecation tracker](https://github.com/pytorch/ao/issues/2259)
- [Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/)

---

## 3. llama.cpp: Desquantizacao Rapida em CPU

### 3.1 Otimizacoes SIMD

llama.cpp e o estado da arte em inferencia quantizada de CPU. As otimizacoes incluem:

**Instrucoes vetoriais suportadas:**
- **ARM NEON**: operacoes vetoriais de 128-bit, ate 24x speedup de decodificacao em Armv9 (Yitian 710)
- **x86 AVX2**: registradores de 256-bit (__m256), processa 8 x F32 simultaneamente
- **x86 AVX-512**: registradores de 512-bit (__m512), processa 16 x F32 simultaneamente

**Alinhamento de memoria:**
- Todos os dados de tensor alinham a limites de **32 bytes**
- Coincide com tamanho de cache line e largura de registradores vetoriais do hardware moderno
- Habilita operacoes SIMD eficientes e coalescing de memoria GPU

### 3.2 Processamento em Blocos

**Sistema k-quants (Q4_K_M, Q5_K_S, Q6_K):**
- Quantizacao por blocos: pesos agrupados em blocos (tipicamente 32 ou 256 valores)
- Cada bloco tem seus proprios fatores de escala

**Estrutura Q4_0 (legado):**
- Blocos de 18 bytes: 1 scale float16 + 16 bytes de valores 4-bit empacotados
- Rendimento: 4.5 bits por peso
- Desquantizacao simples: mais rapido para CPU

**Estrutura Q4_K_M (moderno):**
- Super-blocos de 256 valores = 8 sub-blocos de 32 valores cada
- Super-scale FP16 + scales quantizados por sub-bloco
- **Quantizacao dupla**: reduz overhead dos scales significativamente
- 12 scales de 6-bit + 4 scales de 4-bit, com 8 valores de 6-bit para correcao de offset

**Formula de desquantizacao:**
- Tipo 1 (com zero-point): `weight' = scale * q + zero_point`
- Tipo 0 (sem zero-point): `weight' = scale * q`

### 3.3 Padrao de Desquantizacao Fusionada

A grande inovacao do llama.cpp: **desquantizacao e dot product sao fundidos em uma unica operacao**.

**Funcao `ggml_vec_dot_q4_0_q8_0()` -- passo a passo:**
1. Carregar `block_q4_0` (pesos quantizados)
2. Carregar `block_q8_0` (ativacoes quantizadas)
3. Desempacotar 4-bit para 8-bit (via SIMD)
4. Multiplicar-e-acumular inteiro (integer MAC)
5. Soma horizontal * (scale_pesos * scale_ativacoes)

**Por que isso e rapido:**
- Evita materializar pesos desquantizados na memoria
- Reduz frequencia de loads de memoria e aritmetica de ponteiros
- Operacao SIMD vetorizada com loop unrolling (`GGML_VEC_DOT_UNROLL = 2`)

### 3.4 Quantizacao On-the-Fly de Ativacoes

Um detalhe crucial: **ativacoes (src1, tipo F32) sao quantizadas internamente para Q8_0** durante a multiplicacao de matrizes. Entao o padrao real e:
1. Pesos ja armazenados em Q4_K / Q4_0 / etc.
2. Ativacoes convertidas on-the-fly para Q8_0
3. Dot product fusionado entre Q4 (pesos) e Q8 (ativacoes)

**Q8_0 permanece a opcao mais rapida para inferencia CPU** por causa de seu caminho de desquantizacao simples.

**Benchmarks (LLaMA 3 8B):**
- GPU (RTX 4090): Q4_K_M ~135 tokens/sec geracao
- CPU (Ryzen 7950X): Q4_0 ~18 tokens/sec
- Apple Silicon (M1 Max): Q4_K_M ~38 tokens/sec (vantagem de memoria unificada)

Sources:
- [Quantization Techniques DeepWiki](https://deepwiki.com/ggml-org/llama.cpp/6.3-quantization-techniques)
- [GGUF Optimization Deep Dive](https://medium.com/@michael.hannecke/gguf-optimization-a-technical-deep-dive-for-practitioners-ce84c8987944)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [LLaMA Now Goes Faster on CPUs](https://justine.lol/matmul/)

---

## 4. GGML: Operacoes com Tensores Quantizados

### 4.1 ggml_mul_mat com Tensores Quantizados

O `ggml_mul_mat` despacha para kernels especificos do backend. Na CPU:

**Despacho por tipo de quantizacao:**
- Cada tipo de quantizacao tem seu proprio kernel: `kernel_mul_mat_q4_0`, `kernel_mul_mat_q4_K`, etc.
- Operacoes diretas em dados quantizados **sem desquantizacao explicita completa**

**Conversao de tipos:**
- Pesos ficam no formato quantizado original (Q4_K, Q2_K, etc.)
- Ativacoes sao convertidas para Q8_0
- O `vec_dot_type` de cada formato indica o tipo de parceiro para dot product

### 4.2 Padrao Fusionado Dequant+Dot Product

```
Para cada bloco de pesos (q4) e ativacoes (q8):
  1. Carregar bloco q4 (pesos): scale_w, dados 4-bit
  2. Carregar bloco q8 (ativacoes): scale_a, dados 8-bit
  3. Desempacotar 4-bit -> 8-bit inline (SIMD)
  4. Multiply-accumulate: sum += w[i] * a[i]  (inteiro)
  5. Resultado do bloco: sum_float += sum * scale_w * scale_a
```

**Vantagens sobre desquantizar-depois-multiplicar:**
- Sem alocacao de tensor intermediario para pesos desquantizados
- Dados ficam "quentes" nos registradores SIMD / cache L1
- Memoria poupada: nao precisa materializar a matriz FP32 completa

### 4.3 Modelo de Threading

- Paralelizacao por linhas: cada thread processa linhas independentes da matriz de saida
- Distribuicao baseada em `ggml_compute_params` -- sem sincronizacao complexa
- Combina naturalmente com ordenacao topologica do grafo de computacao e planejamento de memoria do alocador

### 4.4 Relevancia para Nosso Projeto

Nosso `QuantizedLinear._dequantize_weight()` faz exatamente o oposto do padrao GGML:
1. Desempacota TODA a matriz de pesos (loop Python por linha!)
2. Aplica scales a blocos
3. So entao chama `F.linear()` para matmul

**Oportunidade**: implementar um kernel fusionado dequant+matmul em C++ (ou Metal para Mac) que processa blocos diretamente, sem materializar a matriz completa.

Sources:
- [Tensor Operations DeepWiki](https://deepwiki.com/ggml-org/ggml/2.1-tensor-operations)
- [ggml_mul_mat Issue #909](https://github.com/ggml-org/llama.cpp/issues/909)
- [GGML Deep Dive VII](https://xsxszab.github.io/posts/ggml-deep-dive-vii/)

---

## 5. Extensoes C++ Customizadas para PyTorch

### 5.1 Mecanismos de Compilacao

**Duas abordagens:**

1. **Ahead-of-Time** (setuptools):
   - Definir `setup.py` com `torch.utils.cpp_extension.CppExtension` ou `CUDAExtension`
   - Compila durante instalacao
   - Melhor para distribuicao

2. **Just-in-Time** (JIT):
   ```python
   from torch.utils.cpp_extension import load
   my_module = load(name="my_ext", sources=["my_ext.cpp"])
   ```
   - Compila e carrega on-the-fly
   - Otimo para desenvolvimento rapido

### 5.2 Integracao com Autograd

A API de extensoes C++ **nao gera automaticamente a funcao backward**. E necessario:
1. Implementar logica forward e backward em C++
2. Encapsular em `torch.autograd.Function`:
   ```python
   class QuantizedLinearFunction(torch.autograd.Function):
       @staticmethod
       def forward(ctx, input, packed_weight, scales):
           # Chama kernel C++ fusionado
           output = my_ext.fused_dequant_matmul(input, packed_weight, scales)
           ctx.save_for_backward(input, packed_weight, scales)
           return output

       @staticmethod
       def backward(ctx, grad_output):
           # Chama kernel C++ para backward
           ...
   ```
3. Usar Pybind11 (incluido internamente no PyTorch) para bindings Python-C++

### 5.3 Operadores Customizados (Nova API)

A partir do PyTorch 2.x, a abordagem recomendada e registrar operadores customizados:
```cpp
TORCH_LIBRARY(my_ops, m) {
  m.def("fused_dequant_matmul(Tensor input, Tensor packed_w, Tensor scales) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_ops, CPU, m) {
  m.impl("fused_dequant_matmul", &fused_dequant_matmul_cpu);
}
```

### 5.4 Suporte SIMD/Vetorizacao

- O backend C++/OpenMP do Inductor ja usa AVX2 e AVX-512 via biblioteca de vetorizacao do aten
- Para kernels customizados, podemos usar intrinsics SIMD diretamente:
  - `#include <immintrin.h>` para AVX2/AVX-512
  - `#include <arm_neon.h>` para ARM NEON
- A biblioteca [xsimd](https://github.com/xtensor-stack/xsimd) oferece wrappers portaveis para SIMD

### 5.5 Viabilidade para Dequant+Matmul Rapido

**Sim, e pratico.** O caminho seria:

1. Escrever kernel C++ que faz dequant+dot product fusionado (estilo GGML)
2. Usar SIMD intrinsics para plataformas alvo (NEON para Mac M-series, AVX2 para x86)
3. Compilar via JIT com `torch.utils.cpp_extension.load()`
4. Encapsular em `torch.autograd.Function` para integracao seamless

**Estado atual do ecossistema (2025):**
- Triton e considerado alternativa DSL para kernels GPU
- Para CPU, C++ com SIMD ainda e o padrao ouro
- A discussao RFC no PyTorch (Issue #152032) aborda o futuro das extensoes CUDA customizadas

Sources:
- [Custom C++ and CUDA Extensions Tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_extension.html)
- [torch.utils.cpp_extension Docs](https://docs.pytorch.org/docs/stable/cpp_extension.html)
- [RFC: State of Custom CUDA extensions](https://github.com/pytorch/pytorch/issues/152032)

---

## 6. Apple MPS (Metal Performance Shaders) para Matmul Quantizada

### 6.1 Suporte INT4 no MPS

**Sim, MPS suporta INT4.** Apresentado na WWDC 2024:

**Formatos de quantizacao suportados:**
- **8-bit integers**: reducao de 50% no footprint de memoria
- **4-bit integers** (NOVO): reducao de 75% comparado a FP16

**Tecnicas de quantizacao:**

1. **Quantizacao Linear:**
   - `scale = (max_value - min_value) / (2^bits - 1)`
   - Pontos uniformemente distribuidos ao longo da reta numerica

2. **Quantizacao por Lookup Table:**
   - Pontos quantizados customizados baseados na distribuicao dos dados
   - Cada peso recebe um indice 4-bit ou 8-bit na tabela
   - Melhor utilizacao de bits para distribuicoes nao-uniformes

### 6.2 Matmul Quantizada Fusionada

A grande feature: **MPSGraph faz fusao automatica de dequantize + matmul.**

Quando o grafo contem uma operacao de desquantizacao seguida de multiplicacao de matrizes, MPSGraph automaticamente substitui as duas operacoes por uma **unica multiplicacao de matrizes quantizada** que:
- Desquantiza pesos on-the-fly conforme necessario
- Evita armazenar copia temporaria dos pesos desquantizados
- Melhora eficiencia de bandwidth de memoria

### 6.3 Quantizacao por Blocos

Para melhor precisao:
```swift
let dequantized = graph.dequantize(quantizedWeights,
                                   scale: blockScales,
                                   zeroPoint: blockZeroPoints)
```
- Cada bloco pode ter valores independentes de scale e zero-point
- Melhora significativamente a precisao de reconstrucao

### 6.4 Limitacoes no PyTorch via MPS Backend

**Suporte real via PyTorch e limitado:**
- `torchao` permite simulacao de quantizacao low-bit (Int4/FP4) no Mac
- **Porem**: MPS executa como operacoes **emuladas** (frequentemente upcast para BF16 ou Float32)
- Apple Silicon **nao tem equivalente aos tensor cores INT4 da NVIDIA**
- Kernels experimentais de quantizacao ARM CPU em torchao (adicionados Jan 2025): 1-8 bit para ops linear e embedding

**bitsandbytes MPS backend:**
- PR aberto para backend MPS que registra kernels otimizados automaticamente
- Suporta quantizacao 4-bit (NF4/FP4), 8-bit blockwise, e operacoes INT8 lineares

**Para ganhos reais no Mac**: seria necessario escrever kernels Metal customizados via MPSGraph ou usar MLX diretamente.

Sources:
- [WWDC24: Accelerate ML with Metal](https://developer.apple.com/videos/play/wwdc2024/10218/)
- [Metal Performance Shaders Docs](https://developer.apple.com/documentation/metalperformanceshaders)
- [bitsandbytes MPS backend PR](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1853)
- [MPS backend PyTorch Docs](https://docs.pytorch.org/docs/stable/notes/mps.html)

---

## 7. Implementacoes Existentes de Matmul INT4 no PyTorch

### 7.1 TorchAO (PyTorch Architecture Optimization)

**O sucessor oficial do torch.ao.quantization.** Arquitetura baseada em tensor subclasses.

**Componentes centrais:**
- Tipos basicos: `torch.uint1-uint7`, `torch.int1-int8`, `torch.float8_e4m3fn`, etc.
- Operacoes primitivas: `choose_qparams`, `quantize`, `dequantize`
- Tensor subclasses: `Int4Tensor`, `Int8Tensor`, `Float8Tensor`

**Int4 Weight-Only Quantization:**
```python
from torchao.quantization import quantize_, Int4WeightOnlyConfig
quantize_(model, Int4WeightOnlyConfig(group_size=128))
```
- Pesos de `nn.Linear` convertidos para `AffineQuantizedTensor` (int4, assimetrico, quantizado por grupo)
- Packing: dois elementos int4 empacotados lado a lado em um valor uint8
- Layout para kernel tinygemm: `TensorCoreTiledLayoutType` ("tensor_core_tiled")
- Formatos: "tile_packed_to_4d" (CUDA), "plain_int32" (XPU)

**Performance:**
- Llama-3-8B: 1.89x mais rapido com 58% menos memoria (int4)
- Gemma3-12b-it no H100: 1.73x speedup com 65% menos memoria
- Integrado ao vLLM como backend de quantizacao (Abril 2025)

**Backends suportados:**
- MSLK library kernels
- Triton (via torch.compile)
- PyTorch nativo (`_scaled_mm`)
- ARM CPU kernels (1-8 bit) para linear e embedding (Jan 2025)

### 7.2 Quanto (optimum-quanto, Hugging Face)

**Biblioteca device-agnostica de quantizacao:**

**Tipos suportados:** int2, int4, int8, float8 (pesos); int8, float8 (ativacoes)

**Implementacao INT4:**
- Projecao **group-wise affine** com zero-point para int4 e int2
- Projecao **simetrica per-tensor ou per-channel** para int8 e float8
- Biases permanecem em float para preservar precisao da operacao addmm

**Matmul acelerada em CUDA:**
- fp16-int4, bf16-int4, bf16-int8, int8-int8
- Kernels para todos os dispositivos ainda nao implementados

**Compatibilidade:**
- CPU, CUDA, **MPS** (Apple Silicon)
- Compativel com `torch.compile`
- Facil adicionar kernels customizados por dispositivo
- Integracao direta com Hugging Face Transformers via `QuantoConfig`

**Calibracao:** Context manager `Calibration` com atualizacao baseada em momentum (default 0.9) dos ranges de ativacao.

### 7.3 AutoGPTQ

**Kernels CUDA especializados para GPTQ:**

**Kernel de desquantizacao fusionada:**
1. Desquantiza pesos empacotados int32 -> int4
2. Multiplica com tensor de ativacao em FP16 (esquema W4A16)
3. Armazena resultado na saida

**Backends de kernel:**
- **ExLlama v2**: kernel padrao int4*fp16 para matmul
- **Marlin**: kernel otimizado int4*fp16, ativado com `use_marlin=True`
- **Triton**: kernels de desquantizacao acelerados

**Triton Dequantization Kernels:**
- Blog oficial do PyTorch documenta aceleracao de kernels Triton para desquantizacao GPTQ
- Permite fusion de operacoes direto no compilador

Sources:
- [torchao GitHub](https://github.com/pytorch/ao)
- [torchao Quantization Overview](https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html)
- [optimum-quanto GitHub](https://github.com/huggingface/optimum-quanto)
- [Quanto Introduction Blog](https://huggingface.co/blog/quanto-introduction)
- [Accelerating Triton Dequantization (PyTorch Blog)](https://pytorch.org/blog/accelerating-triton/)
- [AutoGPTQ PyPI](https://pypi.org/project/auto-gptq/)

---

## 8. Comparacao com Nosso QuantizedLinear

### 8.1 O Que Temos Hoje

Nosso `core/quantized_linear.py` implementa:
- Empacotamento INT4 (2 valores por byte uint8) e INT2 (4 valores por byte)
- Scales per-block (absmax quantization)
- Desquantizacao completa no forward (Python puro, loop por linha)
- `F.linear()` apos desquantizacao

### 8.2 Gaps Identificados

| Aspecto | Nosso QuantizedLinear | Estado da Arte |
|---------|----------------------|----------------|
| Desquantizacao | Python puro, loop por linha | Kernel fusionado C++/CUDA com SIMD |
| Formato de packing | uint8 simples | Formatos otimizados para SIMD (Q4_K, TensorCoreTiled) |
| Matmul | F.linear() pos-dequant completa | Dequant+dot product fusionado (sem materializar FP32) |
| Tipo de quantizacao | Absmax simetrica | NF4, FP4, affine com zero-point, per-channel |
| Ativacoes | Nao quantizadas | Quantizacao on-the-fly para Q8_0 (GGML) |
| Double quant | Nao | bitsandbytes compress_statistics |
| Outlier handling | Nao | LLM.int8() mixed-precision |
| Device support | CPU apenas (Python) | CUDA, MPS, CPU com SIMD |

### 8.3 Caminhos de Otimizacao

**Prioridade 1 -- Kernel C++ fusionado (impacto alto, esforco medio):**
```
Implementar dequant_matmul_int4() em C++ com:
- AVX2/NEON intrinsics
- Processamento por blocos (estilo GGML)
- Dot product fusionado sem materializar FP32
- JIT compilation via torch.utils.cpp_extension.load()
```

**Prioridade 2 -- Usar torchao como backend (impacto alto, esforco baixo):**
```
Converter nosso QuantizedLinear para usar AffineQuantizedTensor do torchao:
- Int4WeightOnlyConfig ja faz tudo que precisamos
- Kernels otimizados prontos (CUDA, ARM CPU)
- Compativel com torch.compile
```

**Prioridade 3 -- Metal kernel para Mac (impacto medio, esforco alto):**
```
Escrever kernel Metal para dequant+matmul fusionado:
- MPSGraph faz fusao automatica se usar dequantize -> matmul
- Suporte nativo a INT4 no MPS
- Porem: emulado (upcast para FP16/FP32), sem tensor cores INT4
```

**Prioridade 4 -- Otimizar formato de packing (impacto medio, esforco baixo):**
```
Adotar packing estilo GGML:
- Blocos com scale embutido (como block_q4_0: scale fp16 + 32 valores 4-bit)
- Alinhamento a 32 bytes para SIMD
- Pre-shuffle para acesso sequencial eficiente
```

### 8.4 Recomendacao

Para nosso projeto de pesquisa, a abordagem mais pragmatica seria:

1. **Curto prazo**: Integrar `torchao` como backend alternativo -- reusa kernels otimizados existentes
2. **Medio prazo**: Implementar kernel C++ customizado via `cpp_extension.load()` com SIMD -- nos da controle total e serve como baseline de performance
3. **Exploracao**: Investigar quanto como alternativa device-agnostica que ja suporta MPS

O gap critico e que nosso forward pass atual (desquantizacao Python pura seguida de `F.linear`) e o pior padrao possivel para performance. Qualquer um dos caminhos acima eliminaria esse gargalo.

---

## Referencias Completas

### bitsandbytes
- [GitHub Repository](https://github.com/bitsandbytes-foundation/bitsandbytes)
- [nn/modules.py Source](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/nn/modules.py)
- [Hugging Face Docs](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)
- [4-bit Layers DeepWiki](https://deepwiki.com/bitsandbytes-foundation/bitsandbytes/4.1-4-bit-quantized-layers)
- [Linear8bitLt DeepWiki](https://deepwiki.com/bitsandbytes-foundation/bitsandbytes/4.2-linear8bitlt-module)

### PyTorch Quantization
- [Quantization Docs](https://docs.pytorch.org/docs/stable/quantization.html)
- [quantize_per_tensor](https://docs.pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html)
- [Practical Quantization Blog](https://pytorch.org/blog/quantization-in-practice/)
- [Deprecation Tracker](https://github.com/pytorch/ao/issues/2259)

### llama.cpp / GGML
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [Quantization Techniques DeepWiki](https://deepwiki.com/ggml-org/llama.cpp/6.3-quantization-techniques)
- [GGUF Optimization Deep Dive](https://medium.com/@michael.hannecke/gguf-optimization-a-technical-deep-dive-for-practitioners-ce84c8987944)
- [GGML Tensor Operations DeepWiki](https://deepwiki.com/ggml-org/ggml/2.1-tensor-operations)
- [LLaMA Now Goes Faster on CPUs](https://justine.lol/matmul/)

### TorchAO
- [GitHub Repository](https://github.com/pytorch/ao)
- [Quantization Overview Docs](https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html)
- [Quick Start Guide](https://docs.pytorch.org/ao/stable/quick_start.html)
- [GPU Quantization Tutorial](https://docs.pytorch.org/tutorials/unstable/gpu_quantization_torchao_tutorial.html)

### Quanto
- [GitHub Repository](https://github.com/huggingface/optimum-quanto)
- [Introduction Blog](https://huggingface.co/blog/quanto-introduction)
- [Transformers Docs](https://huggingface.co/docs/transformers/quantization/quanto)

### AutoGPTQ
- [Accelerating Triton Dequantization (PyTorch Blog)](https://pytorch.org/blog/accelerating-triton/)
- [AutoGPTQ PyPI](https://pypi.org/project/auto-gptq/)

### Apple MPS
- [WWDC24: Accelerate ML with Metal](https://developer.apple.com/videos/play/wwdc2024/10218/)
- [Metal Performance Shaders Docs](https://developer.apple.com/documentation/metalperformanceshaders)
- [MPS Backend PyTorch Docs](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [bitsandbytes MPS Backend PR](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1853)

### PyTorch C++ Extensions
- [C++ Extensions Tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_extension.html)
- [cpp_extension API Docs](https://docs.pytorch.org/docs/stable/cpp_extension.html)
- [RFC: State of Custom CUDA Extensions](https://github.com/pytorch/pytorch/issues/152032)
