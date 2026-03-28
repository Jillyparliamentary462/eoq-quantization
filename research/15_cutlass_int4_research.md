# Pesquisa: CUTLASS INT4, Blackwell Tensor Cores, e Kernels INT4 Rapidos

Data: 2026-03-28

---

## 1. Suporte CUTLASS para INT4 GEMM/GEMV

### 1.1 O CUTLASS suporta multiplicacao de matrizes INT4?

**Parcialmente, mas sem suporte oficial upstream completo.** A situacao e a seguinte:

- **CUTLASS nao possui suporte oficial upstream para GEMM de entrada mista FP16 x INT4.** A issue [#1122](https://github.com/NVIDIA/cutlass/issues/1122) solicitou esse recurso em outubro de 2023, e segundo as discussoes, o suporte "esta sendo integrado ao upstream NVIDIA/CUTLASS", mas INT4 permaneceu como prioridade secundaria em relacao ao FP16 x INT8.
- **FasterTransformer** (agora parte do TensorRT-LLM) possui kernels escritos com extensoes CUTLASS (`cutlass_extensions`) que suportam `fpA_intB_gemm` -- ou seja, FP16 x INT8/INT4 GEMM. Esse e o caminho mais maduro para usar CUTLASS com INT4.
- **CUTLASS 3.x** possui o **exemplo 86** (`example 86`) que demonstra GEMM de entrada mista (mixed input), suportando combinacoes como FP16 x INT8 em Hopper (SM90). Porem, o suporte nativo a INT4 nesse exemplo requer implementacao adicional (`FragmentShuffler` para S4).
- Um colaborador na issue #1122 comentou: *"nao e dificil de implementar ja que temos int8 agora... a versao simples seria fazer upcast de int4 para int8 primeiro e depois chamar o resto do codigo int8->fp16"*. Porem, advertiu que *"int4->fp16 vai prejudicar muito o mainloop. Entao nao vai ajudar o caso compute bound."*

### 1.2 Qual API usar para matmul INT4 x FP16?

As opcoes disponiveis:

1. **FasterTransformer/TensorRT-LLM `fpA_intB_gemm`**: API mais madura. Usa layouts intercalados com reordenamento de pesos para otimizar uso de cache lines e reduzir overhead de conversao de tipos.
2. **CUTLASS 3.x Device API**: Ponto de entrada via `cutlass::gemm::device::GemmUniversalAdapter`. Para entrada mista, usar o mainloop de entrada mista com KernelSchedule adequado.
3. **CUTLASS para Blackwell (SM100)**: Suporta sub-byte GEMM via instrucao PTX `tcgen05.mma` com qualificador `.kind::f8f6f4` -- mas apenas para formatos de ponto flutuante (FP4 E2M1, FP6, FP8). **INT4 nao e suportado nativamente no Blackwell.**

### 1.3 O CUTLASS pode ser usado via PyTorch `load_inline`?

**Sim, e viavel.** A abordagem e:

```python
from torch.utils.cpp_extension import load_inline

module = load_inline(
    name='cutlass_int4_gemm',
    cpp_sources=[cpp_source_code],
    cuda_sources=[cuda_source_code],
    extra_include_paths=['/caminho/para/cutlass/include'],
    extra_cuda_cflags=['-arch=sm_86'],  # ou sm_90, etc.
    with_cuda=True,
    functions=['minha_funcao_gemm']
)
```

- O parametro `extra_include_paths` permite apontar para os headers do CUTLASS.
- O parametro `extra_cuda_cflags` permite passar flags de compilacao CUDA (como `-gencode`).
- O CUTLASS e header-only para a maioria dos componentes, facilitando a integracao.
- **Cuidado**: a compilacao JIT do CUTLASS pode ser lenta devido a instanciacao pesada de templates C++.

**Fontes:**
- [torch.utils.cpp_extension docs](https://docs.pytorch.org/docs/stable/cpp_extension.html)
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [CUTLASS GEMM API 3x](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html)
- [CUTLASS Discussion #911 - F16 x S8/S4](https://github.com/NVIDIA/cutlass/discussions/911)
- [CUTLASS Issue #1122 - INT4 mixed dtypes](https://github.com/NVIDIA/cutlass/issues/1122)

---

## 2. Suporte INT4 nos Tensor Cores do NVIDIA Blackwell (RTX 6000)

### 2.1 O Blackwell suporta INT4 MMA (multiply-accumulate)?

**NAO nativamente. INT4 foi descontinuado (deprecated) desde Hopper (SM90).**

- No Hopper e no Blackwell, instrucoes INT4 MMA sao **emuladas** -- o compilador gera uma sequencia de emulacao que executa INT4 MMA via INT8 MMA.
- O INT4 foi *"marcado como experimental em arquiteturas anteriores"*, depois *"deprecated como aviso para uso"*, e eventualmente removido do suporte nativo de hardware.
- **Motivo da descontinuacao**: a popularidade tardia de tipos de dados inteiros de baixa precisao. Embora Turing ja suportasse INT8 e INT4, levou 4 anos ate que novos metodos de quantizacao para inferencia explorassem a compactacao do INT4 para servir LLMs.

### 2.2 O que substituiu o INT4?

O Blackwell introduziu **FP4 (E2M1)** e **FP6 (E3M2, E2M3)** como formatos de ponto flutuante de 4 e 6 bits:

| Formato | Bits | Expoente | Mantissa | Suporte Hardware |
|---------|------|----------|----------|------------------|
| E2M1    | 4    | 2        | 1        | Blackwell (SM100, SM120) |
| E3M2    | 6    | 3        | 2        | Blackwell (SM100, SM120) |
| E2M3    | 6    | 2        | 3        | Blackwell (SM100, SM120) |
| E4M3    | 8    | 4        | 3        | Hopper + Blackwell |
| E5M2    | 8    | 5        | 2        | Hopper + Blackwell |

### 2.3 Instrucoes PTX disponiveis para 4-bit no Blackwell

**SM100 (Datacenter -- B100, B200):**
- Instrucao principal: **`tcgen05.mma`** com qualificador `.kind::f8f6f4`
- Suporta operandos em qualquer dos 5 formatos de baixa precisao acima
- Entrada mista permitida (tipos A e B podem ser diferentes)
- Requer PTX ISA 8.7+
- Usa **Tensor Memory (TMEM)** -- 256KB por SM, dedicado aos tensor cores
- Processa tiles de 64x64 em uma unica operacao

**SM120 (Consumer -- RTX 5070, 5080, 5090, RTX PRO 6000):**
- Usa **`mma.sync`** estendido (mesma familia de instrucoes desde Ampere)
- Suporta FP4 e FP6 com block scaling
- **NAO** suporta `tcgen05.mma` nem `wgmma`
- Um tile de 64x64 que SM100 processa em 1 operacao requer **32 chamadas** `mma.sync` separadas (tiles 16x8) no SM120
- Requer PTX ISA 8.7+, CUDA 12.8+

### 2.4 Compute Capability

| GPU | Compute Capability | SM | Familia |
|-----|--------------------|----|---------|
| B100, B200 | 10.0 | SM100 | Datacenter Blackwell |
| B100a (features condicionais) | 10.0a | SM100a | Datacenter Blackwell |
| RTX 5070, 5080, 5090 | 12.0 | SM120 | Consumer Blackwell |
| RTX PRO 6000 | 12.0 | SM120 | Workstation Blackwell |

**Fontes:**
- [Does Blackwell support INT4 native? - NVIDIA Forums](https://forums.developer.nvidia.com/t/does-blackwell-support-int4-native/326513)
- [NVIDIA Tensor Core Evolution - SemiAnalysis](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell)
- [Dissecting Blackwell Architecture - arXiv](https://arxiv.org/html/2507.10789v2)
- [Colfax Sub-byte GEMM Blackwell Tutorial](https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/)
- [SM120 vs SM100 - NVIDIA Forums](https://forums.developer.nvidia.com/t/cuda-toolkit-12-8-what-gpu-is-sm-120/322128)
- [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)

---

## 3. Kernels INT4 GEMV Rapidos Existentes

### 3.1 llama.cpp: Kernel Q4_0 `vec_dot` em CUDA

**Arquitetura do kernel MMVQ (Matrix-Vector Quantizado):**

- O llama.cpp implementa dois tipos de kernel para multiplicacao quantizada:
  - **MMQ** (Matrix-Matrix Quantizado): para batch sizes maiores, quantiza ativacoes para Q8_1 on-the-fly e usa instrucoes `__dp4a` ou Tensor Cores
  - **MMVQ** (Matrix-Vector Quantizado): para batch size <= 8, especializado para geracao de tokens
- O kernel MMVQ usa uma tabela de ponteiros de funcao `get_vec_dot_q_cuda` para selecionar o dot product otimizado por tipo de quantizacao (ex: `vec_dot_q4_0_q8_1`)
- Ajuste dinamico de warps e linhas por bloco baseado na familia da GPU (GCN, RDNA, Turing, Ampere, etc.)

**Formato Q4_0:**
- Blocos de 32 elementos
- Cada bloco armazena: 1 valor `d` (scale, FP16) + 16 bytes (32 nibbles de 4 bits)
- Dequantizacao: `float_value = d * (int4_value - 8)`
- Padding para `MATRIX_ROW_PADDING` (512) para acesso coalescido a memoria

**Arquivos-chave no repositorio:**
- `ggml/src/ggml-cuda/mmvq.cu` -- kernel MMVQ principal
- `ggml/src/ggml-cuda/vecdotq.cuh` -- implementacoes de vec_dot por tipo
- Em Blackwell (CC 1200+), usa instrucoes MXFP4 nativas via `quantize_mmq_mxfp4_cuda`

### 3.2 GPTQ/AutoGPTQ: Implementacao INT4 GEMV

**Estrategia de kernel:**

- O AutoGPTQ **nao possui kernel GEMV INT4 proprio de alta performance**. Em vez disso, delega para backends externos:
  - **Padrao**: kernel exllamav2 `int4*fp16`
  - **Opcao Marlin**: a partir do AutoGPTQ 0.7.0, suporte ao kernel Marlin `int4*fp16` (apenas GPUs Ampere, CC 8.0/8.6)
- O kernel CUDA nativo do AutoGPTQ (`qlinear_cuda`) tem problemas de desempenho com matmul "magra" (skinny): *"desempenho pior que fp32 em alguns casos quando M,N,K = 8, 11008, 11008"*
- O Triton tambem e usado como backend alternativo para dequantizacao, com kernels acelerados pelo PyTorch

### 3.3 Marlin: Como atinge matmul INT4 rapido

**O Marlin e o estado da arte para FP16 x INT4 matmul**, alcancando ~3.87x speedup (proximo do ideal teorico de 4x).

**Principios fundamentais:**

1. **Exploracao do ratio FLOP/byte**: GPUs modernas tem ratio de 100-200 FLOP/byte para FP16. Com pesos de 4 bits, desde que se faca menos de 25-50 multiply-accumulates por peso, e possivel manter speedup proximo de 4x.

2. **Hierarquia de memoria cuidadosamente gerenciada:**
   - Ativacoes sempre buscadas do cache L2, reutilizadas em registradores
   - Pesos carregados assincronamente da memoria global com politica `evict_first` para evitar poluicao do L2
   - Double buffering na shared memory para sobrepor carregamento com computacao

3. **Pipeline de 4 niveis:**
   - Carga global -> shared memory (assincrono, `cp.async`)
   - Shared memory -> registradores (double buffering)
   - Dequantizacao INT4 -> FP16 (truques de manipulacao de bits)
   - Tensor Core MMA (execucao simultanea)
   - **A chave e sobrepor completamente carregamento de memoria e computacao Tensor Core**

4. **Dequantizacao elegante por manipulacao de bits:**
   - Dois INT4 em um INT32 sao dequantizados simultaneamente
   - Usa truques de expoente e mantissa FP16 para converter INT4 diretamente
   - Pesos reorganizados offline para que a dequantizacao alimente diretamente os registradores dos Tensor Cores no layout correto

5. **Particao listrada (striped partitioning):**
   - Tiles de computacao distribuidos entre SMs abrangendo multiplas colunas de saida
   - Garante utilizacao uniforme dos SMs
   - Minimiza etapas de reducao global

6. **Layout de shared memory sem conflitos:**
   - Vetores de 16 bytes armazenados em indices transformados via XOR
   - Execucao sem conflitos de `ldmatrix` em todos os 32 bancos de shared memory

**Desempenho:**
- ~3.87x speedup para batch sizes ate 16-32 (contabilizando 0.125-bit de overhead do scale)
- Degradacao suave para ~1.5x em batch 128
- Integracao com vLLM: 2.8x speedup end-to-end em modelos reais
- Sparse-MARLIN (esparsidade 2:4): 1.2x adicional

### 3.4 exllamav2: Kernel INT4

**Abordagem geral:**

- O exllamav2 implementa kernels CUDA customizados que fundem dequantizacao com multiplicacao de matrizes
- Os pesos quantizados sao usados diretamente: *"a conversao para FP16 acontece no kernel de matmul em pesos individuais conforme sao transmitidos da VRAM e aplicados"*
- Para sequencias longas e/ou batch sizes grandes, pesos sao dequantizados uma matriz (parcial) por vez
- Suporta formato EXL2 (baseado em GPTQ) com quantizacao mista de 2 a 8 bits por peso
- Permite misturar niveis de quantizacao dentro de uma mesma camada linear, produzindo algo semelhante a quantizacao esparsa onde pesos mais importantes recebem mais bits

**Otimizacoes:**
- CUDA Graphs para capturar e reproduzir grafos de computacao (reduz overhead de CPU)
- Fusao de kernels (kernel fusion) para reduzir uso de largura de banda de memoria
- Gerenciamento cuidadoso de buffers temporarios
- Selecao automatica de backend (Flash Attention 2, xFormers, PyTorch SDPA)

**Fontes:**
- [llama.cpp CUDA Backend - DeepWiki](https://deepwiki.com/ggml-org/llama.cpp/5.1-command-line-tools)
- [llama.cpp mmvq.cu](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu)
- [AutoGPTQ GitHub](https://github.com/AutoGPTQ/AutoGPTQ)
- [Accelerating Triton for GPTQ - PyTorch Blog](https://pytorch.org/blog/accelerating-triton/)
- [Marlin GitHub](https://github.com/IST-DASLab/marlin)
- [Marlin Paper - arXiv](https://arxiv.org/html/2408.11743v1)
- [How Marlin pushes boundaries - Red Hat](https://developers.redhat.com/articles/2024/04/17/how-marlin-pushes-boundaries-mixed-precision-llm-inference)
- [exllamav2 GitHub](https://github.com/turboderp-org/exllamav2)
- [exllamav2 DeepWiki](https://deepwiki.com/turboderp-org/exllamav2)

---

## 4. Largura de Banda de Memoria vs Computacao para GEMV

### 4.1 GEMV (batch=1) e sempre limitado por largura de banda?

**Sim, GEMV e fundamentalmente limitado por largura de banda de memoria (memory bandwidth-bound).**

A razao e a intensidade aritmetica:
- **GEMV** (n x n): O(n^2) operacoes de memoria para O(n^2) operacoes de computacao -> intensidade aritmetica = O(1)
- **GEMM** (n x n x n): O(n^2) operacoes de memoria para O(n^3) operacoes de computacao -> intensidade aritmetica = O(n)

Para batch size 1, a geracao de tokens e dominada por GEMV. Como modelos de IA generativa possuem bilhoes de parametros (varios GB ou mais), os caches se tornam ineficazes e a largura de banda DRAM se torna o fator limitante.

### 4.2 Se limitado por largura de banda, INT4 deveria ser ~4x mais rapido que FP16. Por que nao e?

**O speedup teorico de 4x e impedido por varios gargalos praticos:**

1. **Overhead de dequantizacao:**
   - Apos carregar pesos da GMEM para registradores, o GEMM W4A16 precisa dequantizar pesos de 4 bits para 16 bits nos **CUDA Cores** antes de executar MMA nos **Tensor Cores**
   - Existe uma enorme diferenca de desempenho entre CUDA Cores e Tensor Cores
   - Os CUDA Cores responsaveis pela dequantizacao nao conseguem acompanhar a vazao dos Tensor Cores
   - Algoritmos como QoQ (dequantizacao de multiplos elementos em um registrador de 32 bits) sofrem com potencial overflow e requerem dezenas de instrucoes para resolver

2. **Pouca computacao para amortizar:**
   - No GEMV, cada peso e usado apenas uma vez (ou poucas vezes)
   - Nao ha computacao suficiente para "esconder" a latencia da dequantizacao
   - Em contraste, no GEMM com batches maiores, a reutilizacao de pesos amortiza o custo

3. **Overhead de metadados:**
   - Alem dos 4 bits por peso, precisamos armazenar scales (e zero points) por grupo
   - Com group_size=128 e scale FP16, o overhead e de ~0.125 bits/peso
   - Com group_size=32, o overhead sobe para ~0.5 bits/peso

4. **Ineficiencia de acesso a memoria:**
   - Acesso a pesos de 4 bits requer operacoes de shift e mascaramento para extrair nibbles individuais
   - Alinhamento de memoria nao e natural para dados de 4 bits
   - Pode requerer padding para acesso coalescido

5. **Utilizacao incompleta da GPU:**
   - GEMV com batch=1 subutiliza massivamente a GPU
   - Muitos SMs ficam ociosos ou com baixa ocupacao
   - O overhead de lancamento do kernel se torna significativo em relacao ao tempo de computacao

### 4.3 Speedup teorico maximo de INT4 GEMV vs FP16

**Analise teorica (considerando apenas largura de banda):**

Para GEMV com matriz M x K e vetor K x 1:
- FP16: dados = 2*M*K bytes (matriz) + 2*K bytes (vetor) ~ 2*M*K bytes
- INT4: dados = 0.5*M*K bytes (matriz) + overhead_scale + 2*K bytes (vetor)
- Razao: **~4x reducao de dados da matriz**, mas o vetor de entrada permanece FP16

Na pratica, considerando o overhead de scales:
- **Speedup teorico maximo: ~3.87x** (segundo analise do Marlin, contabilizando 0.125 bits de scale overhead com group_size=128)

Na pratica real:
- **Marlin**: ~3.87x para batch 1-32, ~1.5x para batch 128
- **FastGEMV**: 1.4x a 2.7x dependendo da GPU
- **TorchAO tinygemm**: ~1.73x speedup geral (end-to-end com Gemma3-12b)
- **llama.cpp MMVQ**: variavel, tipicamente 2-3x
- **AutoGPTQ nativo**: em alguns casos, **mais lento que FP32** para shapes skinny

A diferenca entre teoria (~4x) e pratica (1.4-3.87x) se deve aos fatores listados em 4.2. **O Marlin e o unico kernel que consistentemente se aproxima do limite teorico**, e faz isso atraves de engenharia extraordinaria de pipeline.

**Fontes:**
- [FastGEMV GitHub](https://github.com/wangsiping97/FastGEMV)
- [LiquidGEMM Paper](https://arxiv.org/html/2509.01229v1)
- [ROCm LLM Inference Blog](https://rocm.blogs.amd.com/artificial-intelligence/llm-inference-optimize/README.html)
- [ATOM Paper](https://homes.cs.washington.edu/~arvind/papers/atom-mlsys.pdf)
- [TorchAO Quick Start](https://docs.pytorch.org/ao/stable/quick_start.html)
- [Accelerating LLM Inference with GemLite - PyTorch Blog](https://pytorch.org/blog/accelerating-llm-inference/)

---

## 5. PyTorch `torch.utils.cpp_extension` com CUTLASS

### 5.1 Podemos incluir headers CUTLASS no `load_inline`?

**Sim.** O CUTLASS e majoritariamente header-only, o que facilita a integracao:

```python
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
// ... codigo do kernel ...

torch::Tensor meu_gemm_int4(torch::Tensor A, torch::Tensor B_quantized) {
    // Implementacao usando CUTLASS
    // ...
}
"""

module = load_inline(
    name='cutlass_gemm',
    cpp_sources=[''],  # pode ser vazio se tudo estiver no CUDA
    cuda_sources=[cuda_source],
    extra_include_paths=[
        '/caminho/para/cutlass/include',
        '/caminho/para/cutlass/tools/util/include',
    ],
    extra_cuda_cflags=[
        '-std=c++17',
        '-gencode', 'arch=compute_86,code=sm_86',
        '-O3',
    ],
    with_cuda=True,
    functions=['meu_gemm_int4'],
)

# Uso:
resultado = module.meu_gemm_int4(ativacoes_fp16, pesos_int4)
```

### 5.2 Parametros importantes do `load_inline`

| Parametro | Descricao |
|-----------|-----------|
| `name` | Nome unico do modulo (usado para cache de compilacao) |
| `cpp_sources` | Lista de strings com codigo C++ |
| `cuda_sources` | Lista de strings com codigo CUDA (.cu) |
| `extra_include_paths` | Lista de diretorios para includes (ex: headers CUTLASS) |
| `extra_cuda_cflags` | Flags adicionais para nvcc |
| `extra_cflags` | Flags adicionais para o compilador C++ |
| `extra_ldflags` | Flags de linkagem adicionais |
| `with_cuda` | Adicionar headers/libs CUDA (auto-detectado se `cuda_sources` fornecido) |
| `no_implicit_headers` | Pular includes automaticos (melhora tempo de cold start) |
| `functions` | Lista de funcoes a expor do modulo |
| `verbose` | Mostrar saida de compilacao (util para debug) |
| `build_directory` | Diretorio customizado para build |

### 5.3 Consideracoes praticas

**Vantagens:**
- Prototipagem rapida sem necessidade de setup.py ou CMake
- Cache de compilacao automatico (recompila apenas se o codigo mudar)
- Integracao direta com tensores PyTorch

**Desvantagens:**
- Compilacao JIT de CUTLASS e **muito lenta** (minutos) devido a instanciacao pesada de templates C++
- A primeira execucao e lenta; execucoes subsequentes usam cache
- Debug de erros de compilacao de templates C++ em strings e dificil
- Limite de tamanho de codigo em string pode ser um problema para kernels complexos

**Alternativas recomendadas para producao:**
- **TorchAO**: ja integra kernels INT4 otimizados (tinygemm, Marlin)
- **torch.compile** com backend Triton: para prototipar kernels de dequantizacao
- **setup.py / CMake**: para projetos maiores com CUTLASS

### 5.4 Exemplos existentes de CUTLASS com PyTorch

Embora nao existam exemplos oficiais de `load_inline` + CUTLASS amplamente divulgados, existem referencias uteis:

- **TorchAO** (`pytorch/ao`): implementa kernels INT4 customizados em `torchao/csrc`, acessiveis via `torch.ops`
- **vLLM**: integra Marlin e outros kernels CUTLASS-based para servico de modelos quantizados
- **AutoGPTQ**: usa extensoes CUDA que compilam kernels baseados em CUTLASS/exllamav2
- **PyTorch RFC #152032**: discute o estado atual de extensoes CUDA customizadas em PyTorch, incluindo integracao com CUTLASS

**Fontes:**
- [torch.utils.cpp_extension - PyTorch Docs](https://docs.pytorch.org/docs/stable/cpp_extension.html)
- [Custom C++ and CUDA Operators - PyTorch Tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [PyTorch RFC #152032 - State of Custom CUDA Extensions](https://github.com/pytorch/pytorch/issues/152032)
- [TorchAO GitHub](https://github.com/pytorch/ao)
- [TorchAO RFC #697 - Which low bit CUDA kernels](https://github.com/pytorch/ao/issues/697)

---

## 6. Resumo e Recomendacoes

### Para nosso projeto de quantizacao DCT + INT4:

1. **CUTLASS nao e o melhor caminho para INT4 GEMV customizado.**
   - Suporte INT4 nao e oficial no upstream
   - INT4 foi descontinuado no hardware desde Hopper
   - O overhead de templates C++ dificulta prototipagem rapida

2. **O Marlin e a referencia-ouro para FP16 x INT4 matmul.**
   - Unico kernel que atinge consistentemente ~3.87x speedup
   - Porem, e otimizado para Ampere (SM80/86) e pode nao funcionar em todas as GPUs
   - Codigo altamente complexo e especifico de arquitetura

3. **Para prototipagem rapida, considerar:**
   - **TorchAO `Int4WeightOnlyConfig`**: integrado ao PyTorch, usa tinygemm
   - **Triton**: para escrever kernels de dequantizacao customizados
   - **FastGEMV**: implementacao simples e educativa de GEMV INT4

4. **Para Blackwell (RTX 6000 / SM120):**
   - Considerar **FP4 (E2M1)** em vez de INT4, ja que tem suporte nativo de hardware
   - O SM120 usa `mma.sync` estendido (nao `tcgen05.mma`)
   - CUDA 12.8+ e PTX ISA 8.7+ sao necessarios

5. **O gargalo fundamental do GEMV batch=1 e largura de banda de memoria.**
   - O speedup teorico maximo e ~3.87x (INT4 vs FP16)
   - Na pratica, o overhead de dequantizacao consume parte desse ganho
   - Kernels que fundem dequantizacao com MMA no pipeline sao essenciais
