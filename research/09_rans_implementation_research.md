# Pesquisa: Implementacao rANS, ECQ, e Codificacao por Entropia para Pesos Neurais

**Data:** 2026-03-28
**Escopo:** Pesquisa aprofundada sobre padroes de implementacao rANS (em Python e C), detalhes do paper ECQ, integracao com GGUF/llama.cpp, comparativo rANS vs Huffman vs codificacao aritmetica, e tANS/FSE como alternativa.

---

## Sumario Executivo

A implementacao de referencia **ryg_rans** de Fabian Giesen define o padrao para codificadores rANS, usando estado de 32 bits com renormalizacao por byte (ou 64 bits com palavras de 32 bits). A precisao de probabilidade padrao e **12 a 16 bits**, com 14 bits sendo o ponto ideal para estados de 32 bits. O projeto **ECQ** demonstra que rANS aplicado a pesos quantizados em 4 bits atinge **3.27x de compressao lossless** em media, com throughput projetado de **140 tokens/seg** em M3 Pro para modelos 7B. **tANS/FSE** (usado no zstd) oferece decodificacao ate **2x mais rapida** que rANS por eliminar multiplicacoes, mas requer tabelas maiores. A principal barreira para integracao com **GGUF/llama.cpp** e a arquitetura de memory-mapping, que exigiria descompressao on-the-fly com kernel fundido (fused decode+GEMM).

---

## 1. Implementacao rANS: Padroes de Referencia

### 1.1 ryg_rans -- Implementacao de Referencia em C

O repositorio **ryg_rans** de Fabian Giesen (rygorous) e a implementacao publica de referencia para rANS, disponibilizada em dominio publico. Oferece tres variantes principais:

**Variante byte-aligned (rans_byte.h):**
- Estado de 32 bits
- Renormalizacao emitindo/consumindo 1 byte por vez
- Funciona em todas as arquiteturas 32-bit
- Decodificador livre de divisao (division-free)

**Variante 64-bit (rans64.h):**
- Estado de 64 bits, emitindo palavras de 32 bits por vez
- Consideravelmente mais rapida em arquiteturas 64-bit
- Codificador aritmetico muito preciso

**Variante SIMD (rans_word_sse41.h):**
- Decodificador SSE 4.1 com I/O em unidades de 16 bits
- Otimizado para paralelismo de dados

**Fonte:** [ryg_rans - GitHub](https://github.com/rygorous/ryg_rans)

### 1.2 Implementacoes em Python

Existem varias implementacoes Python de rANS, cada uma com abordagens distintas:

**bits-back/rans.py (Jamie Townsend):**
- Acompanha o tutorial paper arXiv:2001.09186
- Usa `head_precision = 64` e `tail_precision = 32`
- Restricao: `tail_precision < head_precision <= 2 * tail_precision`
- Estado como tupla `(head, tail)` onde tail e uma lista cons (linked list) imutavel
- Encoding: verifica se `head >= prob << (head_precision - precision)` para overflow
- Decoding: extrai frequencia cumulativa via `cf = head & ((1 << precision) - 1)`
- Design funcional -- funcoes retornam novo estado em vez de mutar

**Python-rANSCoder (FGlazov):**
- Estado de 64 bits emitindo palavras de 32 bits
- Compativel com funcoes rans64 do ryg_rans em C
- Usa Numba JIT para performance
- LIFO explicito: "o ultimo simbolo codificado e o primeiro decodificado"
- Requer `reverse()` manual na saida para recuperar ordem original
- Encoder de uso unico (single-use): apos `get_encoded()`, instancia e invalidada

**py-rans (PyPI):**
- Pacote leve em Python puro
- Opera em 64 bits com emissao de palavras de 32 bits
- Instalavel via pip

**Fontes:**
- [bits-back/rans.py](https://github.com/bits-back/bits-back/blob/master/rans.py)
- [Python-rANSCoder](https://github.com/FGlazov/Python-rANSCoder)
- [py-rans - PyPI](https://pypi.org/project/py-rans/)

### 1.3 Tratamento Correto da Saida LIFO (Stack)

O rANS opera como uma **pilha (stack)** -- a ordem de codificacao e inversa a de decodificacao:

**Padrao correto:**
1. **Codificador** processa dados em ordem reversa (do final para o inicio)
2. **Codificador** emite bytes em reverso -- recebe ponteiro para o fim do buffer e move em direcao ao inicio
3. **Decodificador** processa em ordem forward (do inicio para o fim)
4. **Decodificador** produz simbolos na ordem original

**Abordagem por chunks (pratica):**
Em vez de acumular todos os simbolos emitidos sobre todo o stream e fazer um flush gigante no final, os dados sao processados em **chunks** (blocos). Cada chunk e codificado em reverso internamente, e o buffer de saida e depois invertido.

**Para rANS adaptativo:**
O codificador processa simbolos em ordem reversa, enquanto estima probabilidades em ordem forward e armazena na memoria do encoder para uso durante a codificacao reversa. Isso garante que encoder e decoder tenham exatamente a mesma estimativa de probabilidade para cada simbolo.

**Fonte:** [rANS in practice - ryg blog](https://fgiesen.wordpress.com/2015/12/21/rans-in-practice/)

### 1.4 Bugs Comuns em Implementacoes rANS

**1. Desincronizacao encoder/decoder por renormalizacao:**
O encoder entra em um estado que requer renormalizacao, emite um bit, e move para um novo estado. O decoder alcanca o mesmo estado mas sem saber que o encoder chegou la emitindo um bit -- o decoder ve um estado normalizado que nao requer leitura de mais bits, criando ambiguidade.

**2. Overshoot na leitura de bits pelo decoder:**
O decoder pode comecar abaixo do limite inferior do intervalo alvo e, apos ler um unico bit, ultrapassar o limite superior -- indicando que o intervalo e "muito pequeno".

**3. Violacao da restricao L = k*M:**
A soma de todas as frequencias (M) **deve** ser multiplo do parametro de normalizacao (L). Violacoes desta restricao produzem erros intermitentes e dificeis de diagnosticar. A solucao padrao: fazer ambos M e L potencias de 2.

**4. Bugs na renormalizacao de 16 bits:**
Implementacoes usando variante de renormalizacao de 16 bits tem sido reportadas como "falhando ocasionalmente", sugerindo bugs sutis nesta abordagem.

**5. Branch misprediction durante renormalizacao:**
Decisoes sobre quando ler/escrever bytes para normalizar o estado rANS podem causar problemas de performance por execucao especulativa. Solucao: implementacoes branch-free (como a do BitKnit).

**6. Propagacao de tipos NumPy em Python:**
Em implementacoes Python, escalares NumPy podem se propagar silenciosamente, causando overflow ou comportamento inesperado. Conversao explicita para inteiros Python e essencial.

**Fontes:**
- [rANS notes - ryg blog](https://fgiesen.wordpress.com/2014/02/02/rans-notes/)
- [rANS in practice - ryg blog](https://fgiesen.wordpress.com/2015/12/21/rans-in-practice/)

### 1.5 Bits de Precisao: Qual e o Padrao?

A escolha de bits de precisao para a escala de probabilidade depende do tamanho do estado e da aplicacao:

| Configuracao | Estado | Renorm | Precisao Prob | Uso Tipico |
|---|---|---|---|---|
| ryg_rans byte | 32 bits | 8-bit (byte) | 16 bits | Referencia, proposito geral |
| BitKnit | 32 bits | 16-bit (word) | **14 bits** | Jogos, compressao rapida |
| ryg_rans 64-bit | 64 bits | 32-bit (word) | 32 bits | Alta precisao, 64-bit arch |
| DietGPU (Facebook) | - | byte-level | **9-11 bits** | GPU, alta vazao |
| bits-back (Python) | 64 bits | 32-bit | Configuravel | Pesquisa, prototipagem |

**Regra pratica de Fabian Giesen:**
- Para estado de 32 bits com renormalizacao de 16 bits: usar **14 bits** de precisao (16 bits seria imprudente, pois a faixa de precisao e muito estreita)
- Para estado de 32 bits com renormalizacao de 8 bits: **16 bits** de precisao funciona bem
- Para estado de 64 bits com renormalizacao de 32 bits: ate **32 bits** de precisao
- Limitar variavel de estado a **31 bits** (ou 63 bits em 64-bit) melhora performance com metodo de reciproco

**Tamanhos de tabela recomendados:** 2^12 a 2^14 entradas oferecem melhor tradeoff entre precisao e eficiencia de cache. Tabelas muito grandes consomem cache ineficientemente; tabelas muito pequenas limitam severamente a resolucao de probabilidade.

**Fonte:** [rANS with static probability distributions - ryg blog](https://fgiesen.wordpress.com/2014/02/18/rans-with-static-probability-distributions/)

---

## 2. ECQ (Entropy-Coded Quantization) -- Detalhes do Projeto

### 2.1 Visao Geral

O ECQ e um projeto que demonstra como codificacao por entropia (rANS) pode comprimir pesos quantizados em 4 bits de LLMs com compressao lossless adicional significativa.

**Insight fundamental:** Pesos de LLMs quantizados em 4 bits possuem **entropia de Shannon significativamente abaixo de 4 bits**, tipicamente entre 1.12-1.54 bits. A distribuicao dos simbolos quantizados e nao-uniforme (formato de sino/bell-curve), permitindo que codificacao por entropia se aproxime dos limites teoricos.

### 2.2 Codificador de Entropia Usado

O ECQ utiliza **rANS** como codificador de entropia (nao Huffman, nao CABAC). O repositorio inclui `rans_codec.py` como implementacao do codec rANS em Python puro.

**Comparacao com trabalhos anteriores:**
- EntroLLM: usa codificacao Huffman
- Liguori et al.: usa ANS em FPGA
- ECQ: usa rANS em software, com plano para kernel Metal

### 2.3 Medicoes de Entropia de Shannon por Modelo

| Modelo | Entropia (bits) | Compressao sobre 4-bit | Tamanho Resultante |
|---|---|---|---|
| Qwen2.5-1.5B | 1.12 | 3.58x | - |
| Qwen2.5-0.5B | 1.15 | 3.47x | - |
| GPT-2 | 1.17 | 3.42x | - |
| SmolLM-135M | 1.54 | 2.59x | - |
| **Media** | **~1.25** | **3.27x** | - |

### 2.4 Como Atinge 3.27x de Speedup no M3 Pro

O speedup nao e computacional, mas sim de **bandwidth de memoria**:

**Calculo para modelo 7B no Apple Silicon M3 Pro (150 GB/s de bandwidth):**
- 4-bit quantizado padrao: **3.5 GB** de tamanho, **42.9 tokens/seg**
- 4-bit com entropia (ECQ): **1.07 GB** de tamanho, **140.0 tokens/seg**

O speedup vem de:
1. **Reducao do tamanho do modelo**: de 3.5 GB para 1.07 GB (3.27x menor)
2. **Inferencia e bandwidth-bound**: a velocidade e limitada por quao rapido os pesos podem ser lidos da memoria
3. **Ratio compute-to-bandwidth do Apple Silicon**: 47:1 -- ha enorme excesso de capacidade computacional
4. **Kernel fundido (fused decode+GEMM)**: descompressao rANS e multiplicacao de matriz sao realizadas simultaneamente, sem materializar pesos descomprimidos na memoria

A chave e que a descompressao rANS usa ciclos de compute que estao "sobrando" enquanto o hardware espera por dados de memoria. Como o gargalo e bandwidth (nao compute), a descompressao e essencialmente "gratis".

### 2.5 Detalhes de Implementacao

- **Linguagem:** Python puro (100% do repositorio)
- **Kernel Metal:** Planejado mas ainda nao implementado. Design documentado em `docs/metal_kernel_design.md`
- **Codificacao por linha (per-row):** Streams independentes permitem decodificacao paralela na GPU com complexidade O(N) em vez de O(rows x N)
- **Dois modos de decodificacao:**
  - **Fused:** Memoria-eficiente, decodifica direto nos registradores durante GEMV
  - **Cached:** Velocidade-otimizado, materializa blocos descomprimidos
- **Block size:** Nao especificado explicitamente no README; codificacao e per-row (por linha da matriz de pesos)

**Fonte:** [ECQ - GitHub](https://github.com/drxddy/ecq)

### 2.6 Feature Request para MLX (Apple)

O mesmo autor abriu uma feature request no MLX (framework ML da Apple) propondo integracao de rANS entropy-coded quantization:

**Dados de entropia medidos empiricamente:**

| Quantizacao | Bits Armazenados | Entropia Real | Upside de Compressao |
|---|---|---|---|
| 8-bit | 8 | ~4-5 bits | 1.6-2x |
| 4-bit | 4 | ~2.17 bits | 1.84x |
| 3-bit | 3 | ~1.8 bits | 1.7x |
| 2-bit | 2 | ~1.5 bits | 1.3x |

A proposta incluia API Python com `EntropyCodedLinear.from_quantized(linear)`, codificacao per-row, e kernel fundido decode+GEMV. O issue foi fechado em janeiro de 2026, com o autor optando por extensao standalone em vez de integracao direta no MLX.

**Fonte:** [MLX Feature Request #3043](https://github.com/ml-explore/mlx/issues/3043)

---

## 3. GGUF e Codificacao por Entropia no llama.cpp

### 3.1 Situacao Atual

**Nao existe proposta formal** para adicionar codificacao por entropia ao formato GGUF ou ao llama.cpp. As buscas extensivas em issues, discussions e PRs do repositorio nao revelaram nenhuma discussao tecnica especifica sobre integrar rANS, Huffman ou qualquer codificador de entropia ao pipeline de quantizacao do GGUF.

**Discussoes relacionadas encontradas:**
- **TurboQuant (Issue #20979):** Proposta de compressao extrema usando PolarQuant + QJL, mas sem codificacao por entropia. Issue fechado como duplicata.
- **Compressao de blocos (Discussion #8731):** Proposta de compressao LZ4 HC para blocos GGUF com descompressao on-the-fly. Autor reporta fator de compressao de 2.72x com LZ4 HC -9, mas sem discussao de codificacao por entropia.
- **Entropia em sampling (Discussion #15627):** Discussao sobre amostragem adaptativa baseada em entropia (tail-free sampling), nao relacionada a compressao de pesos.

### 3.2 Barreiras Tecnicas para Adicionar Codificacao por Entropia ao GGUF

**1. Arquitetura de memory-mapping:**
O formato GGUF e projetado para ser **memory-mappable** -- carregamento eficiente via `mmap()` com zero-copy. Codificacao por entropia exigiria descompressao, eliminando completamente esta vantagem. Tensores hoje sao mapeados diretamente do disco para a memoria sem processamento intermediario.

**2. Restricoes de alinhamento:**
Dados de tensores sao alinhados a 32 bytes (`GGUF_DEFAULT_ALIGNMENT`) para acesso SIMD eficiente. Dados codificados por entropia nao podem manter limites de bytes rigidos, quebrando esta propriedade.

**3. Independencia de blocos:**
Formatos de quantizacao atuais (Q4_0, Q4_K_M, etc.) permitem processamento independente de blocos (tipicamente 32 ou 256 pesos). Codificacao por entropia tipicamente introduz dependencias inter-blocos, complicando decodificacao paralela.

**4. Latencia de inferencia:**
Descompressao durante carregamento ou inferencia adicionaria overhead computacional, contradizendo o objetivo de "performance estado-da-arte" do projeto.

**5. Complexidade do kernel de dequantizacao:**
O llama.cpp ja possui kernels altamente otimizados (CUDA, Metal, Vulkan, CPU SIMD) para cada formato de quantizacao. Adicionar descompressao de entropia a cada kernel multiplicaria dramaticamente a complexidade do codigo.

**6. Multiplataforma:**
O llama.cpp suporta CPU, CUDA, Metal, Vulkan, SYCL e outros backends. Um kernel fundido decode+GEMM precisaria ser implementado em **cada backend**, representando esforco de engenharia massivo.

### 3.3 Caminho Viavel para Integracao

A abordagem mais realistica seria:
1. **Novo tipo de quantizacao** no enum `GGMLQuantizationType` (ex: `GGML_TYPE_Q4_0_RANS`)
2. **Descompressao no carregamento** (nao durante inferencia) -- os pesos sao descomprimidos uma vez ao carregar o modelo, usando mais tempo de carregamento em troca de tamanho de arquivo menor
3. **Tamanho de arquivo reduzido** sem impacto na performance de inferencia
4. **Alternativa:** codificacao por entropia como compressao de transporte (similar a gzip para downloads), descomprimida para GGUF padrao antes do uso

**Fontes:**
- [GGUF File Format - DeepWiki](https://deepwiki.com/ggml-org/llama.cpp/7.1-gguf-file-format)
- [Compressing blocks discussion #8731](https://github.com/ggml-org/llama.cpp/discussions/8731)
- [TurboQuant issue #20979](https://github.com/ggml-org/llama.cpp/issues/20979)

---

## 4. rANS vs Huffman vs Codificacao Aritmetica para Pesos Neurais

### 4.1 Comparativo Geral

| Caracteristica | Huffman | Codificacao Aritmetica | rANS |
|---|---|---|---|
| Taxa de compressao | Limitada a potencias de 2 (1 bit/simbolo minimo) | Proxima da entropia | Proxima da entropia |
| Velocidade de codificacao | Rapida | Lenta | Rapida |
| Velocidade de decodificacao | Muito rapida | Lenta | Rapida (division-free) |
| Complexidade | Simples | Complexa (underflow/overflow) | Moderada |
| Memoria | Arvore Huffman (pequena) | Poucos registradores | Poucos registradores |
| Paralelizavel | Sim (por bloco) | Dificil | Sim (interleaving) |
| Adaptativo | Possivel mas lento | Natural | Possivel |

### 4.2 Qual e o Mais Rapido para Decodificar?

**Benchmarks em CPU (Skylake 3.4 GHz, enwik9):**

| Codificador | Decodificacao (MB/s) |
|---|---|
| TurboANX (ANS otimizado) | **1,523.95** |
| TurboHF 0 (Huffman otimizado) | 1,457.74 |
| rANS_static32 | 1,346.12 |
| TurboHF 12 | 1,182.66 |
| fsehuf 32 | 831.45 |
| rANS_static16 | 676.69 |
| FSE 32 | 501.70 |

**Benchmarks em GPU (DietGPU, A100):**

| Codec | Throughput |
|---|---|
| ANS (rANS) | 250-410 GB/s |
| Float codec (rANS + expoente) | 250-600 GB/s |
| Recoil (rANS paralelo) | 90+ GB/s (Turing GPU) |

**Conclusao:** Em CPU, implementacoes otimizadas de ANS e Huffman atingem velocidades comparaveis (~1.3-1.5 GB/s). Em GPU, rANS atinge throughput massivo (centenas de GB/s) gracas a paralelizacao.

**Fontes:**
- [Entropy Coder Benchmark - powturbo](https://sites.google.com/site/powturbo/entropy-coder)
- [DietGPU - Facebook Research](https://github.com/facebookresearch/dietgpu)

### 4.3 Qual Comprime Mais Proximo da Entropia?

Tanto rANS quanto codificacao aritmetica se aproximam do limite de Shannon. A diferenca e negligivel na pratica:

- **Huffman:** Limitado a codigos de comprimento inteiro. Para distribuicoes muito enviesadas (como pesos quantizados), perde significativamente. Exemplo: se P(simbolo) = 0.9, entropia = 0.47 bits, mas Huffman usa minimo 1 bit.
- **Codificacao aritmetica:** Proxima do limite teorico, mas computacionalmente cara.
- **rANS:** Virtualmente identico a codificacao aritmetica em taxa de compressao, mas com custo computacional similar ao Huffman.

**Para pesos neurais quantizados:** rANS e a escolha optima porque as distribuicoes sao fortemente enviesadas (bell-curve), exatamente o cenario onde Huffman perde eficiencia e rANS mantem compressao proxima da entropia.

O paper de Jarek Duda que introduziu ANS descreve exatamente esta vantagem: "entropy coding combining speed of Huffman coding with compression rate of arithmetic coding."

**Fonte:** [Asymmetric numeral systems - Duda 2013](https://arxiv.org/abs/1311.2540)

### 4.4 Aplicacao Pratica: Deep Compression vs rANS

O trabalho "Deep Compression" (Han et al., 2016) usou **Huffman coding** e alcancou 35-49x de compressao. Trabalhos recentes (2025) mostram que substituir Huffman por rANS/ANS fornece:

- **1.3-2x compressao adicional** sobre quantizacao existente
- Pesos quantizados em 4 bits tem entropia medida de ~2.17 bits (upside de 1.84x)
- Pesos quantizados em 8 bits tem entropia de ~4-5 bits (upside de 1.6-2x)

O paper CERWU (2025) combina quantizacao rate-constrained com codificacao por entropia (DeepCABAC) e atinge **20-40% melhor compressao** que o padrao NNCodec (ISO/IEC 15938-17), com descompressao em 0.2-1.2 segundos para redes completas.

**Fonte:** [Reducing Storage of Pretrained Neural Networks](https://arxiv.org/abs/2505.18758)

---

## 5. tANS (Table ANS) e FSE como Alternativa ao rANS

### 5.1 Como Funciona o tANS

O tANS coloca **todo o comportamento** (incluindo renormalizacao) para estados x em [L, 2L-1] em uma tabela de lookup, resultando em uma **maquina de estados finitos** que evita a necessidade de multiplicacao.

**Algoritmo de decodificacao tANS:**
1. Consultar tabela[estado] -> retorna simbolo e proximo estado base
2. Ler bits do stream para completar transicao de estado
3. Nenhuma multiplicacao ou divisao necessaria

**Algoritmo de decodificacao rANS:**
1. Calcular `cf = estado mod M` (mascara de bits)
2. Lookup do simbolo via tabela de frequencias cumulativas
3. Calcular novo estado: `estado = freq * (estado / M) + (estado mod M) - cum_freq`
4. Renormalizar se necessario

### 5.2 tANS e Mais Rapido para Decodificacao?

**Sim, significativamente.** Benchmarks concretos no dataset "book1":

| Variante | Throughput (Mbps) |
|---|---|
| rANS non-interleaved | 235.97 |
| rANS 2x interleaved | 334.95 |
| tANS non-interleaved | 369.44 |
| tANS 2x interleaved | **509.63** |

O tANS e **~1.5-2x mais rapido** que rANS para decodificacao porque:
- Substitui operacoes `and + shift + mul + add` por um unico table lookup
- rANS requer ~15 ciclos de CPU para a multiplicacao
- tANS requer ~4 ciclos para o lookup
- Diferenca de ~11 ciclos por simbolo

**Em compressao de imagens neurais:** Experimentos demonstram que tANS reduz latencia em **77%** comparado a rANS, com perda de taxa de compressao de **12.6%**, mantendo qualidade de compressao superior ao JPEG2000.

### 5.3 FSE (Finite State Entropy) -- Implementacao de tANS no zstd

O **FSE** (Finite State Entropy) de Yann Collet e a implementacao pratica mais importante de tANS, usada nos compressores:
- **Facebook Zstandard (zstd)** -- usado no kernel Linux, Google Chrome, Android
- **Apple LZFSE** -- compressor padrao do macOS/iOS
- **Google Draco** -- compressor 3D (usado no Pixar USD)
- **Google PIK** -- compressor de imagens

**Benchmarks FSE (Core i5-3340M):**

| Dataset | Compressao (MB/s) | Descompressao (MB/s) | Taxa |
|---|---|---|---|
| win98-lz4-run | 325 | 415 | 2.688 |
| proba70.bin | - | 420 | 6.316 |
| proba80.bin | - | 440 | 8.84 |
| proba90.bin | - | 420 | 15.21 |

**Vantagens do FSE sobre Huffman classico:**
- Huffman e limitado a 1 bit por simbolo minimo (perda enorme para P > 0.5)
- FSE nao tem este limite -- mantem compressao proxima de Shannon para qualquer distribuicao
- Velocidade "muito estavel" independente da distribuicao de probabilidade

**Caracteristicas tecnicas do FSE:**
- Tabela de estados de **12 bits** (~1-16 KB de memoria)
- Operacoes: apenas adicoes, mascaras e shifts -- sem multiplicacoes ou divisoes
- Performance estavelmente em torno de 400-460 MB/s para descompressao em CPU single-core

**Fontes:**
- [FSE - Yann Collet](https://github.com/Cyan4973/FiniteStateEntropy)
- [FSE Blog Post](http://fastcompression.blogspot.com/2013/12/finite-state-entropy-new-breed-of.html)

### 5.4 FSE/tANS para Compressao de Pesos Neurais?

**Vantagens potenciais:**
1. **Decodificacao mais rapida** que rANS (~2x), crucial para inferencia bandwidth-bound
2. **Sem multiplicacoes** -- apenas table lookups, ideal para kernels GPU simples
3. **Implementacao madura** via zstd/FSE, amplamente testada e otimizada
4. **Distribuicoes estaticas** de pesos quantizados sao ideais para tANS (as probabilidades nao mudam durante inferencia)

**Desvantagens:**
1. **Tabelas maiores** que rANS -- para alfabeto de 16 simbolos (4-bit) com 12 bits de precisao, tabela de ~64 KB vs poucos bytes para rANS
2. **Compressao ~12.6% pior** que rANS em cenarios de alta compressao
3. **Menos flexivel** para probabilidades dinamicas (irrelevante para pesos, que sao estaticos)
4. **Mais memoria** por stream -- problematica para decodificacao paralela de muitas linhas simultaneamente

**Recomendacao para nosso caso:**
Para pesos de LLM quantizados em 4 bits (16 simbolos), tANS/FSE e uma **alternativa viavel e possivelmente superior ao rANS**:
- Distribuicoes sao estaticas (calculadas uma vez durante quantizacao)
- Alfabeto pequeno (16 simbolos) mantem tabelas compactas
- Decodificacao mais rapida e critica para inferencia bandwidth-bound
- A pequena perda de compressao (~12%) pode ser aceitavel dado o ganho de velocidade

---

## 6. DietGPU: ANS em GPU para ML (Facebook Research)

### 6.1 Visao Geral

O **DietGPU** da Meta/Facebook e a primeira implementacao publica de rANS generalizado em GPU, projetada especificamente para aplicacoes de ML/HPC.

### 6.2 Especificacoes Tecnicas

- **Variante ANS:** rANS byte-oriented generalizado
- **Precisao de probabilidade:** 9, 10 ou 11 bits (quantizado a 1/512, 1/1024 ou 1/2048)
- **Unidade de compressao:** Segmentos de **4 KiB** atribuidos a warps individuais
- **Tamanho minimo de dados:** 512 KiB para uso pratico
- **Operacao:** Completamente on-device (sem sincronizacao CPU-GPU)
- **APIs:** C++ e PyTorch

### 6.3 Performance

| Codec | Throughput (A100) |
|---|---|
| ANS (rANS) | 250-410 GB/s |
| Float codec (expoente) | 250-600 GB/s |

**Compressao de floats ML:**
- Expoentes de float16/bfloat16 tipicamente tem ~2.7 bits de entropia
- Compressao de ~0.67x para bfloat16, ~0.85x para float16 (apenas expoente)
- Estrategia: comprimir expoentes separadamente (alta compressibilidade) dos significandos (baixa compressibilidade)

### 6.4 Relevancia para Nosso Projeto

O DietGPU demonstra que rANS em GPU e **viavel e performante** para dados de ML. Porem, seu foco e compressao de comunicacao inter-GPU (NVLink/PCIe), nao compressao de pesos para inferencia. A arquitetura de segmentos de 4 KiB por warp e um modelo util para o design de nosso kernel Metal.

**Fonte:** [DietGPU - GitHub](https://github.com/facebookresearch/dietgpu)

---

## 7. Paper CERWU: Rate-Constrained Quantization com Entropy Coding (2025)

### 7.1 Abordagem

O paper "Reducing Storage of Pretrained Neural Networks by Rate-Constrained Quantization and Entropy Coding" (maio 2025) propoe uma abordagem conjunta rate-distortion:

1. **Funcao de perda dividida** em componentes quadratico e nao-quadratico
2. **Optimal Brain Surgeon (OBS)** iterativo para solucoes localmente exatas
3. **Parametro Lagrangiano lambda** que balanceia taxa de bits vs distorcao
4. **Contribuicao individual por peso** via conteudo informacional: `-log2 P(peso)` bits

### 7.2 Codificador de Entropia

Usa **DeepCABAC** (codificacao aritmetica adaptativa ao contexto), nao rANS. O paper nota que "codificacao aritmetica, range coding, ou ANS" podem atingir taxas proximas do optimo, mas implementa DeepCABAC especificamente por sua modelagem autogressiva de estatisticas de pesos neurais.

### 7.3 Resultados

- **20-40% melhor compressao** que NNCodec (padrao ISO/IEC 15938-17) mantendo acuracia proxima da original
- **Descompressao:** 0.2 a 1.2 segundos para redes completas (single-threaded)
- Supera OPTQ, Round-to-Nearest, e NNCodec em todos os cenarios testados
- Grid de quantizacao simetrico e uniforme com ate 2^8 = 256 niveis

**Fonte:** [CERWU - arXiv](https://arxiv.org/abs/2505.18758)

---

## 8. Sintese e Recomendacoes

### 8.1 Para Implementacao rANS em Nosso Projeto

1. **Precisao de probabilidade:** Usar **12-14 bits** para estado de 32 bits, ou **16 bits** para estado de 64 bits
2. **Variante recomendada:** Estado de 64 bits com emissao de palavras de 32 bits (compativel com ryg_rans/rans64.h)
3. **Tratamento LIFO:** Codificar em ordem reversa, decodificar forward. Processar por chunks/blocos
4. **Prototipar em Python** com bits-back/rans.py ou py-rans, depois portar para C seguindo ryg_rans
5. **Evitar bugs:** Garantir L = k*M (potencias de 2), conversao explicita de tipos NumPy, renormalizacao branch-free

### 8.2 Escolha entre rANS e tANS/FSE

| Criterio | rANS | tANS/FSE |
|---|---|---|
| Compressao | Melhor (~12% superior) | Boa |
| Decodificacao CPU | Rapida (~1.3 GB/s) | Mais rapida (~2x) |
| Memoria de tabela | Minima | ~1-16 KB por distribuicao |
| Implementacao GPU | Requer multiplicacao | Apenas lookup |
| Maturidade | ryg_rans (excelente) | FSE/zstd (excelente) |
| Para 4-bit (16 simbolos) | Bom | **Ideal** (tabela pequena) |

**Recomendacao:** Comecar com **rANS** para validacao e prototipagem (mais simples, melhor compressao), depois avaliar **tANS/FSE** para o kernel Metal de producao (decodificacao mais rapida, sem multiplicacao).

### 8.3 Integracao com GGUF/llama.cpp

A integracao direta e **inviavel no curto prazo** devido a arquitetura de memory-mapping. Alternativas:
1. **Formato proprietario** para nosso pipeline (mais provavel)
2. **Compressao de transporte** -- arquivo comprimido que descomprime para GGUF padrao no carregamento
3. **Novo tipo GGML** que descomprime uma vez no carregamento (maior tempo de load, mesma performance de inferencia)
4. **Contribuicao upstream** a longo prazo, se os resultados forem convincentes

---

## Fontes Consolidadas

### Implementacoes rANS
- [ryg_rans - GitHub](https://github.com/rygorous/ryg_rans)
- [rANS in practice - ryg blog](https://fgiesen.wordpress.com/2015/12/21/rans-in-practice/)
- [rANS notes - ryg blog](https://fgiesen.wordpress.com/2014/02/02/rans-notes/)
- [rANS with static probability distributions - ryg blog](https://fgiesen.wordpress.com/2014/02/18/rans-with-static-probability-distributions/)
- [bits-back/rans.py](https://github.com/bits-back/bits-back/blob/master/rans.py)
- [Python-rANSCoder](https://github.com/FGlazov/Python-rANSCoder)
- [py-rans - PyPI](https://pypi.org/project/py-rans/)
- [jkbonfield/rans_static](https://github.com/jkbonfield/rans_static)

### ECQ e Compressao de Pesos Neurais
- [ECQ - GitHub](https://github.com/drxddy/ecq)
- [MLX Feature Request #3043](https://github.com/ml-explore/mlx/issues/3043)
- [CERWU - arXiv](https://arxiv.org/abs/2505.18758)
- [Deep Compression - Han et al. 2016](https://arxiv.org/abs/1510.00149)

### GGUF e llama.cpp
- [GGUF File Format - DeepWiki](https://deepwiki.com/ggml-org/llama.cpp/7.1-gguf-file-format)
- [Compressing blocks discussion #8731](https://github.com/ggml-org/llama.cpp/discussions/8731)
- [TurboQuant issue #20979](https://github.com/ggml-org/llama.cpp/issues/20979)

### tANS/FSE
- [FSE - GitHub](https://github.com/Cyan4973/FiniteStateEntropy)
- [FSE Blog Post](http://fastcompression.blogspot.com/2013/12/finite-state-entropy-new-breed-of.html)
- [tANS Essay - Annon Inglorion](https://inglorion.net/documents/essays/data_compression/tans/)

### Benchmarks e Comparativos
- [Entropy Coder Benchmark - powturbo](https://sites.google.com/site/powturbo/entropy-coder)
- [DietGPU - Facebook Research](https://github.com/facebookresearch/dietgpu)
- [ANS - Duda 2013](https://arxiv.org/abs/1311.2540)
- [ANS Wikipedia](https://en.wikipedia.org/wiki/Asymmetric_numeral_systems)
