# Apple Silicon: Bandwidth de Memoria e Inferencia de LLMs

## 1. Especificacoes de Bandwidth de Memoria do Apple Silicon

A arquitetura de memoria unificada (Unified Memory Architecture) da Apple e central para o
desempenho de inferencia de LLMs. Diferente de GPUs discretas que possuem VRAM dedicada,
o Apple Silicon compartilha a mesma memoria entre CPU e GPU, eliminando copias entre
dispositivos -- mas a bandwidth total e o fator limitante.

### Tabela Completa de Bandwidth por Chip

| Chip         | Bandwidth (GB/s) | Memoria Max | Barramento  |
|--------------|-------------------|-------------|-------------|
| **M1**       | 68.25             | 16 GB       | 128-bit     |
| **M1 Pro**   | 200               | 32 GB       | 256-bit     |
| **M1 Max**   | 400               | 64 GB       | 512-bit     |
| **M1 Ultra** | 800               | 128 GB      | 1024-bit    |
| **M2**       | 100               | 24 GB       | 128-bit     |
| **M2 Pro**   | 200               | 32 GB       | 256-bit     |
| **M2 Max**   | 400               | 96 GB       | 512-bit     |
| **M2 Ultra** | 800               | 192 GB      | 1024-bit    |
| **M3**       | 100               | 24 GB       | 128-bit     |
| **M3 Pro**   | 150               | 36 GB       | 192-bit     |
| **M3 Max**   | 300-400           | 128 GB      | 384/512-bit |
| **M3 Ultra** | 819               | 512 GB      | --          |
| **M4**       | 120               | 32 GB       | 128-bit     |
| **M4 Pro**   | 273               | 64 GB       | --          |
| **M4 Max**   | 410-546           | 128 GB      | --          |

**Notas importantes:**
- O M3 Pro sofreu uma reducao de 25% em bandwidth comparado ao M1/M2 Pro (150 vs 200 GB/s),
  devido ao barramento de 192-bit ao inves de 256-bit.
- O M3 Max com 14-core CPU tem 300 GB/s; o com 16-core tem 400 GB/s.
- O M4 Max com 32-core GPU binado tem 410 GB/s; a versao completa de 40-core atinge 546 GB/s.
- Nao existe M4 Ultra ate o momento (marco 2026). A Apple pulou diretamente do M3 Ultra para
  o esperado M5 Ultra.

### Quanto da Bandwidth Esta Disponivel para Inferencia?

Na pratica, a GPU do Apple Silicon nao pode usar 100% da memoria do sistema. O limite
pratico e de aproximadamente **75% da memoria total** do sistema para tarefas de GPU.
Exemplo: um Mac com 128 GB de RAM pode usar apenas ~96 GB para tarefas de GPU.

A bandwidth efetiva para inferencia tambem nao atinge o maximo teorico -- overheads do
sistema operacional, KV cache, e outras operacoes competem pela bandwidth. Estimativas
praticas sugerem 70-85% de utilizacao efetiva da bandwidth maxima para cargas de trabalho
otimizadas como llama.cpp.


## 2. Inferencia de LLMs no Apple Silicon

### Benchmarks do llama.cpp no Apple Silicon

Os dados abaixo sao baseados em benchmarks da comunidade llama.cpp (Discussion #4167)
e de multiplas fontes independentes.

#### Geracao de Texto (Token Generation, batch=1) -- LLaMA 7B

| Chip          | Q4_0 (t/s) | Q8_0 (t/s) | FP16 (t/s) | Bandwidth |
|---------------|-------------|-------------|-------------|-----------|
| M1 (8 GPU)    | ~14         | --          | --          | 68 GB/s   |
| M2 (10 GPU)   | ~18         | --          | ~6.5        | 100 GB/s  |
| M2 Pro         | --          | --          | ~13         | 200 GB/s  |
| M2 Max (38 GPU)| ~63        | ~44         | ~25         | 400 GB/s  |
| M2 Ultra       | --          | --          | ~41         | 800 GB/s  |
| M4 (10 GPU)   | ~54*        | --          | --          | 120 GB/s  |
| M4 Max (40 GPU)| ~54        | --          | --          | 546 GB/s  |

*Nota: valores aproximados de multiplas fontes; condicoes de teste variam.

#### Processamento de Prompt (Prompt Processing, batch=512) -- LLaMA 7B

| Chip             | Q8_0 (t/s)  |
|------------------|-------------|
| M1 (8 GPU)       | ~118        |
| M2 (10 GPU)      | ~181        |
| M3 (10 GPU)      | ~187        |
| M2 Ultra (76 GPU)| ~1.249      |
| M4 Max (40 GPU)  | ~892        |

O processamento de prompt (prefill) e **compute-bound**, portanto escala melhor
com o numero de cores GPU. A geracao de tokens (decode) e **memory-bandwidth-bound**,
e e aqui que a bandwidth de memoria domina o desempenho.

### Impacto da Quantizacao na Velocidade

Para um modelo 7B no Apple Silicon, a relacao entre quantizacao e velocidade e clara:

| Quantizacao | Tamanho (~7B) | tok/s (relativo) | Speedup vs FP16 |
|-------------|---------------|-------------------|------------------|
| FP16        | ~14 GB        | 1.0x (baseline)   | --               |
| Q8_0        | ~7.5 GB       | ~1.8x             | ~1.8x            |
| Q4_K_M      | ~4.0 GB       | ~2.8-3.0x         | ~2.8-3.0x        |
| Q4_0        | ~3.5 GB       | ~2.9-3.2x         | ~2.9-3.2x        |

Exemplo concreto (M2 Pro, 16GB, modelo 7B):
- FP16: ~24 t/s
- Q8_0: ~44 t/s
- Q4_0: ~70 t/s

**Observacao critica:** Q4_0 alcanca o melhor desempenho no Apple Silicon porque a
bandwidth de memoria disponivel esta em **alinhamento perfeito** com o compute que os chips
M oferecem. Q8_0 nao tem bandwidth suficiente para alimentar o compute, e formatos que
precisam de menos bandwidth nao tem compute suficiente.

### Modelos Maiores

Para modelos de 70B parametros:
- M2 Max (96GB): ~8 tok/s com Q4
- M2 Ultra: ~8-12 tok/s com Q4
- Um modelo de 70B em Q4 ocupa ~35 GB, exigindo no minimo 48-64 GB de RAM unificada


## 3. Inferencia Limitada por Bandwidth de Memoria

### Por que a Inferencia de LLM e Bandwidth-Bound?

A inferencia de LLMs possui duas fases distintas:

1. **Prefill (processamento de prompt):** Processa todos os tokens do prompt em paralelo.
   E **compute-bound** -- muitas operacoes por byte carregado da memoria.

2. **Decode (geracao de tokens):** Gera um token por vez, sequencialmente.
   E **memory-bandwidth-bound** -- para cada token gerado, todos os pesos do modelo
   precisam ser lidos da memoria, mas o compute por byte e minimo.

A intensidade aritmetica (operacoes por byte) da fase de decode e muito baixa.
Para o Llama 2 7B, a intensidade aritmetica e ~62 ops/byte, enquanto uma GPU A10
tem capacidade de 208 ops/byte. Isso significa que a GPU fica ociosa esperando os
dados chegarem da memoria -- o gargalo e puramente de bandwidth.

### O Modelo Roofline para Inferencia de LLMs

O modelo Roofline e um framework teorico que visualiza o desempenho maximo atingivel
de um algoritmo em um hardware especifico:

```
Desempenho = min(Pico_Compute, Bandwidth_Memoria x Intensidade_Aritmetica)
```

Para operacoes com **baixa intensidade aritmetica** (como a geracao autoregressiva de
tokens), o desempenho e limitado pela bandwidth de memoria ("memory wall").

Para operacoes com **alta intensidade aritmetica** (como o prefill com batches grandes),
o desempenho e limitado pelo pico de compute ("compute ceiling").

A fase de decode de LLMs cai firmemente no lado esquerdo do modelo Roofline (regiao
limitada por memoria), onde aumentar o compute nao melhora o desempenho -- apenas
mais bandwidth ajuda.

### Como Calcular o Maximo Teorico de tok/s a Partir da Bandwidth

A formula fundamental para a fase de decode e:

```
tempo_por_token = tamanho_do_modelo_em_bytes / bandwidth_de_memoria

tok/s_teorico = bandwidth_de_memoria / tamanho_do_modelo_em_bytes
```

**Exemplos concretos:**

| Hardware     | Bandwidth | Modelo (7B FP16=14GB) | Modelo (7B Q4=3.5GB) |
|-------------|-----------|------------------------|----------------------|
| M2          | 100 GB/s  | 7.1 tok/s              | 28.6 tok/s           |
| M2 Max      | 400 GB/s  | 28.6 tok/s             | 114 tok/s            |
| M4 Pro      | 273 GB/s  | 19.5 tok/s             | 78 tok/s             |
| M4 Max      | 546 GB/s  | 39 tok/s               | 156 tok/s            |
| A100 (80GB) | 2.039 TB/s| 145 tok/s              | 582 tok/s            |

**Nota:** Estes sao maximos teoricos. Na pratica, deve-se esperar 60-80% desses
valores devido a overheads de KV cache, dequantizacao, controle de fluxo, e
utilizacao imperfeita da bandwidth.

Exemplo pratico de validacao: o Baseten Guide calcula que um A10 (600 GB/s) rodando
um modelo 7B FP16 (14GB) leva ~23ms por token, o que da ~43.5 tok/s.
Calculo: 14 GB / 600 GB/s = 0.023s => 1/0.023 = 43.5 tok/s.


## 4. Speedup Teorico vs Real da Quantizacao

### Se Q4 e 4x Menor, a Inferencia e 4x Mais Rapida?

**Nao.** Embora a quantizacao Q4 reduza o tamanho do modelo em ~4x comparado a FP16,
o speedup real e significativamente menor que 4x. As razoes:

#### Overheads que Reduzem o Speedup

1. **Dequantizacao:** Pesos quantizados precisam ser dequantizados antes de cada
   multiplicacao matricial. Metodos state-of-the-art de INT4 sofrem overhead de
   **20-90%** na dequantizacao de pesos ou somas parciais em GPUs.

2. **Overhead de controle de fluxo:** Formatos de quantizacao como Q4_K_M possuem
   estrutura mais complexa (escalas por bloco, offsets) que requerem processamento
   adicional.

3. **KV Cache:** O KV cache nao e quantizado por padrao e cresce com o comprimento
   da sequencia. Em sequencias longas, a leitura do KV cache pode dominar a
   bandwidth, reduzindo o beneficio relativo da quantizacao dos pesos.

4. **Overhead para modelos pequenos:** Para modelos com menos de 3 bilhoes de
   parametros, a quantizacao NF4 pode **aumentar** o consumo de energia em
   25-56% apesar de atingir 75% de compressao de memoria. O overhead de
   dequantizacao excede a economia de bandwidth nessa escala.

5. **Overhead de geracao autoregressiva:** Em sequencias longas, o custo acumulado
   de quantizacao/dequantizacao dinamica repetida se torna consideravelmente caro.

#### Medidas Reais de Speedup

| Comparacao          | Speedup Teorico | Speedup Real     |
|---------------------|-----------------|------------------|
| Q4 vs FP16 (7B)    | 4.0x            | 2.5-3.2x         |
| Q8 vs FP16 (7B)    | 2.0x            | 1.6-1.9x         |
| Q4 vs Q8 (7B)      | 2.0x            | 1.5-1.7x         |

Exemplo concreto (modelo 7B, Apple Silicon):
- FP16: ~24 tok/s
- Q8_0: ~44 tok/s (speedup real: 1.83x; teorico: 2x)
- Q4_0: ~70 tok/s (speedup real: 2.92x; teorico: 4x)

O speedup real de Q4 vs FP16 e tipicamente **60-80% do teorico** para modelos de
7B+ parametros em hardware com bandwidth moderada.

#### Por que Q4_0 e "Perfeito" para Apple Silicon

O Q4_0 alcanca desempenho otimo no Apple Silicon porque existe um **equilibrio exato**
entre a bandwidth de memoria disponivel e a capacidade de compute dos chips M. Com Q8_0,
ha bandwidth suficiente mas o overhead de compute e maior por byte. Com formatos
mais agressivos (como Q2), ha bandwidth sobrando mas compute insuficiente para
a dequantizacao.


## 5. PyTorch no Apple MPS

### O Backend MPS (Metal Performance Shaders)

O MPS e o backend do PyTorch para aceleracao via GPU no Apple Silicon, usando a API
Metal da Apple. Permite mover tensors e computacao para a GPU dos chips M.

### Desempenho: MPS vs CPU no Mac

- Speedups de **varias vezes sobre CPU** sao comuns para inferencia de transformers e CNNs.
- O ganho e mais pronunciado em **batch sizes maiores**.
- Em modelos muito pequenos ou batch sizes minusculos, o overhead de despachar trabalho
  para a GPU pode **reduzir ou eliminar** o beneficio sobre a CPU.
- O MPS funciona melhor quando a utilizacao da GPU e alta.

### MPS vs Alternativas Nativas

Um estudo abrangente de novembro 2025 (arXiv:2511.05502) comparou frameworks de inferencia
em um M2 Ultra com 192 GB:

| Framework       | Throughput (tok/s) | Observacao                               |
|-----------------|--------------------|-----------------------------------------|
| **MLX**         | ~230               | Maior throughput sustentado              |
| **llama.cpp**   | Eficiente          | Melhor para uso leve single-stream       |
| **MLC-LLM**     | Bom                | Menor TTFT para prompts moderados        |
| **Ollama**      | Menor              | Boa ergonomia, menor throughput          |
| **PyTorch MPS** | ~7-9               | Limitado por memoria em modelos grandes  |

**MLX e 25-30x mais rapido que PyTorch MPS** para inferencia de LLMs no mesmo hardware.
Isso ocorre porque o MLX implementa operacoes nativamente para a memoria unificada com
operacoes zero-copy e avaliacao lazy, enquanto o PyTorch MPS adapta operacoes no estilo
CUDA para Metal.

### MPS e Inferencia Quantizada

**Limitacoes significativas:**

1. **INT4/FP4:** O `torchao` permite simular quantizacao low-bit no MPS, mas as
   operacoes sao **emuladas** (upcast para BF16 ou FP32 para computacao). Nao ha
   aceleracao real de hardware para INT4.

2. **INT8:** Suportado via `bitsandbytes` com backend MPS. Inclui quantizacao blockwise
   de 8-bit, operacoes lineares INT8, e otimizadores de 8-bit. Porem, variantes in-place
   como `int8_linear_matmul.out` nao sao implementadas.

3. **FP8:** **Nao suportado.** Apple Silicon nao tem suporte de hardware nativo para FP8.
   Operacoes FP8 sao emuladas via upcast para BF16, o que **anula** o beneficio de
   desempenho. O PyTorch levanta erro ao tentar usar float8 no MPS.

4. **Float64:** **Nao suportado.** O MPS nao suporta double precision. Se um modelo
   tenta alocar um tensor float64, o PyTorch levanta erro de runtime ou mantem o
   tensor na CPU silenciosamente.

### Outras Limitacoes do Backend MPS

- **Limite de memoria GPU:** A GPU nao pode usar mais que ~75% da memoria total do
  sistema.
- **Atencao (SDPA):** Relatos de crashes com `scaled_dot_product_attention` no
  macOS/Apple Silicon. Muitos usuarios forcam a implementacao "eager" no Transformers
  para evitar code paths nao-otimizados.
- **Sem batching automatico:** O MPS nao oferece continuous batching ou scheduler
  a nivel de step -- apenas batching estatico manual para inputs de mesmo comprimento.
- **Regressoes de memoria:** Problemas de regressao de memoria no MPS foram rastreados
  no PyTorch em 2025.
- **INT4 catching up:** O suporte a INT4 nativo via `torchao` esta melhorando
  gradualmente, mas ainda nao atinge a aceleracao real que GPUs NVIDIA com Tensor
  Cores oferecem.

### Recomendacao Pratica

Para inferencia de LLMs no Apple Silicon, as opcoes em ordem de desempenho sao:
1. **MLX** -- framework nativo da Apple, melhor throughput
2. **llama.cpp** -- otimizado para Metal, excelente para uso geral
3. **MLC-LLM** -- bom equilibrio de features e desempenho
4. **PyTorch MPS** -- util para prototipagem e fine-tuning, mas significativamente
   mais lento para inferencia de LLMs


## 6. Implicacoes para Compressao DCT de Pesos

### Relevancia para Este Projeto

A analise de bandwidth tem implicacoes diretas para o projeto de compressao DCT:

1. **O gargalo e a bandwidth, nao o compute:** Qualquer tecnica que reduza o tamanho
   dos pesos em memoria tera impacto direto na velocidade de inferencia, desde que o
   overhead de decompressao seja menor que a economia de bandwidth.

2. **Overhead de decompressao e critico:** Se a decompressao DCT for mais cara que a
   dequantizacao Q4 simples, o beneficio de compressao adicional pode ser anulado. O
   overhead de dequantizacao Q4 ja consome 20-80% do ganho teorico.

3. **Apple Silicon como caso de teste:** Com bandwidth de 100-546 GB/s (chips base a Max),
   o Apple Silicon e um excelente caso de teste para tecnicas de compressao -- a diferenca
   entre tamanhos de modelo e diretamente visivel em tok/s.

4. **Formula de planejamento:**
   ```
   tok/s_estimado = (bandwidth_efetiva * fator_utilizacao) / tamanho_comprimido_do_modelo
   ```
   Onde `fator_utilizacao` e tipicamente 0.6-0.8 e inclui todos os overheads.

5. **Break-even de overhead:** Para que a compressao DCT valha a pena em desempenho,
   o overhead de decompressao DCT precisa ser menor que:
   ```
   overhead_max = (1 - taxa_compressao) * tempo_leitura_original
   ```
   Exemplo: Se DCT comprime Q4 em mais 50% (de 3.5GB para 1.75GB), o tempo de
   leitura economizado no M4 Pro (273 GB/s) seria ~6.4ms. O DCT inverso precisa
   levar menos que ~6.4ms por token para valer a pena.


## Fontes

- [Performance of llama.cpp on Apple Silicon M-series - GitHub Discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167)
- [llama.cpp Performance & Apple Silicon - Andreas Kunar](https://medium.com/@andreask_75652/llama-cpp-performance-apple-silicon-051241dd6eae)
- [Benchmarking Apple's MLX vs. llama.cpp - Andreas Kunar](https://medium.com/@andreask_75652/benchmarking-apples-mlx-vs-llama-cpp-bbbebdc18416)
- [A guide to LLM inference and performance - Baseten](https://www.baseten.co/blog/llm-transformer-inference-guide/)
- [LLM Inference Unveiled: Survey and Roofline Model Insights - arXiv](https://arxiv.org/html/2402.16363v4)
- [Memory Bandwidth and Compute Bottlenecks in LLM Inference - ApXML](https://apxml.com/courses/llm-compression-acceleration/chapter-1-foundations-llm-efficiency-challenges/memory-compute-bottlenecks-inference)
- [Production-Grade Local LLM Inference on Apple Silicon - arXiv](https://arxiv.org/abs/2511.05502)
- [State of PyTorch Hardware Acceleration 2025](https://tunguz.github.io/PyTorch_Hardware_2025/)
- [Apple Silicon PyTorch MPS: Setup and Speed - Till Code](https://tillcode.com/apple-silicon-pytorch-mps-setup-and-speed-expectations/)
- [MPS backend - PyTorch Documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [Quantization Q4_K_M vs AWQ vs FP16 - SitePoint](https://www.sitepoint.com/quantization-q4km-vs-awq-fp16-local-llms/)
- [Quantization Guide 2025 - Local AI Zone](https://local-ai-zone.github.io/guides/what-is-ai-quantization-q4-k-m-q8-gguf-guide-2025.html)
- [Practical GGUF Quantization Guide for iPhone and Mac - Enclave AI](https://enclaveai.app/blog/2025/11/12/practical-quantization-guide-iphone-mac-gguf/)
- [Memory Bandwidth: How Does It Boost Tokens per Second - Hardware Corner](https://www.hardware-corner.net/memory-bandwidth-llm-speed/)
- [Apple M4 Wikipedia](https://en.wikipedia.org/wiki/Apple_M4)
- [Apple M3 Pro Chip Has 25% Less Memory Bandwidth - MacRumors](https://www.macrumors.com/2023/10/31/apple-m3-pro-less-memory-bandwidth/)
- [Apple introduces M4 Pro and M4 Max - Apple Newsroom](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/)
- [Mac Studio Specs - Apple](https://www.apple.com/mac-studio/specs/)
- [Apple reveals M3 Ultra - Apple Newsroom](https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/)
- [Model Quantization: Concepts, Methods, and Why It Matters - NVIDIA](https://developer.nvidia.com/blog/model-quantization-concepts-methods-and-why-it-matters/)
