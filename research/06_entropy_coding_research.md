# Pesquisa: Codificacao por Entropia Aplicada a Compressao de Pesos de Redes Neurais

**Data:** 2026-03-28
**Escopo:** Levantamento abrangente sobre codificacao por entropia (entropy coding) para compressao de pesos de modelos de linguagem (LLMs), com foco em aplicabilidade ao formato GGUF e ao projeto llama.cpp.

---

## Sumario Executivo

A pesquisa revela uma **lacuna de entropia (entropy gap) massiva e inexplorada** nos pesos quantizados de LLMs. Pesos quantizados em 4 bits possuem entropia de Shannon real de apenas **1.12-2.17 bits**, significando que 45-72% do espaco de armazenamento e desperdicado. Codificacao por entropia (rANS, Huffman, CABAC) pode comprimir esses pesos em ate **3.27x adicionais** sem nenhuma perda de qualidade. Implementacoes modernas de rANS atingem **3 GB/s** de decodificacao em CPU unica, tornando a abordagem viavel para inferencia em tempo real.

---

## 1. Codificacao por Entropia Aplicada a Compressao de Pesos Neurais

### 1.1 Contexto Historico

O trabalho seminal de **Han, Mao e Dally (2016)** -- "Deep Compression" (ICLR 2016 Best Paper) -- estabeleceu o pipeline de tres estagios: poda (pruning), quantizacao treinada e codificacao Huffman. Esse pipeline alcancou compressao de **35x a 49x** sem perda de acuracia:

- AlexNet: de 240 MB para 6.9 MB (35x)
- Poda reduz conexoes em 9-13x
- Quantizacao reduz de 32 bits para 5 bits por conexao
- Huffman coding fornece compressao adicional significativa

**Fonte:** [Deep Compression - Han et al. 2016](https://arxiv.org/abs/1510.00149)

### 1.2 DeepCABAC (Fraunhofer HHI)

O **DeepCABAC** aplica Context-Adaptive Binary Arithmetic Coding (CABAC) -- originalmente projetado para o padrao H.264/AVC de video -- aos parametros de redes neurais.

**Resultados:**
- VGG16 ImageNet: compressao de **63.6x** sem perda de acuracia (de ~553 MB para 8.7 MB)
- BERT: razao de compressao de 9% do tamanho original com F1-score de 86%
- Consistentemente supera tecnicas anteriores de codificacao para compressao neural

**Caracteristicas tecnicas:**
- Esquema de quantizacao que minimiza funcao rate-distortion
- Considera o impacto da quantizacao na acuracia da rede
- Modelagem de estatisticas tipicas de pesos neurais
- Suporta taxas sub-1-bit por peso amortizado

**Fonte:** [DeepCABAC - Wiedemann et al. 2019](https://arxiv.org/abs/1907.11900)

### 1.3 Padrao ISO/IEC 15938-17 (NNC - Neural Network Coding)

O DeepCABAC tornou-se o motor principal de codificacao do primeiro padrao internacional de compressao de redes neurais:

- **ISO/IEC 15938-17:2022** (primeira edicao)
- **ISO/IEC 15938-17:2024** (segunda edicao vigente)
- Compressao para **menos de 5%** do tamanho original sem degradacao de inferencia
- Em cenarios de treinamento distribuido: atualizacoes representadas em **1% ou menos** do modelo base
- Implementacao de referencia: **NNCodec** (Fraunhofer HHI, codigo aberto)

**Pipeline de compressao NNC:**
1. Pre-processamento (poda, esparsificacao, decomposicao de baixo rank)
2. Quantizacao eficiente (uniforme, dependente, adaptativa)
3. Codificacao aritmetica via DeepCABAC
4. Ferramentas adicionais: batch norm folding, escalonamento local, acesso aleatorio

**Fonte:** [Fraunhofer HHI - Neural Network Coding](https://www.hhi.fraunhofer.de/en/departments/ai/research-groups/efficient-deep-learning/research-topics/neural-network-compression.html)

### 1.4 CERWU: Rate-Constrained Quantization + Entropy Coding (2025)

Trabalho recente combina quantizacao com restricao de taxa e codificacao por entropia:

- **20-40% de reducao na taxa de bits** comparado ao NNCodec na mesma acuracia
- Abordagem fundamentada em teoria da informacao que subsume poda e quantizacao dependente de saliencia como casos-limite
- Usa DeepCABAC para modelar estatisticas de pesos e atingir taxas fracionarias
- Regularizacao Hessiana previne acumulacao de pesos que degrada compressibilidade

**Fonte:** [CERWU - arxiv 2505.18758](https://arxiv.org/html/2505.18758)

---

## 2. A Lacuna de Entropia na Quantizacao Atual

### 2.1 Medicoes Empiricas de Entropia

Esta e possivelmente a descoberta **mais importante** desta pesquisa. Dados empiricos de multiplas fontes convergem:

#### ECQ (Entropy Coded Quantization) -- Medicoes de Shannon:

| Modelo | Bits Alocados | Entropia Real (Shannon) | Lacuna |
|--------|:---:|:---:|:---:|
| Qwen2.5-1.5B | 4 bits | **1.12 bits** | 2.88 bits (72%) |
| Qwen2.5-0.5B | 4 bits | **1.15 bits** | 2.85 bits (71%) |
| GPT-2 | 4 bits | **1.17 bits** | 2.83 bits (71%) |
| SmolLM-135M | 4 bits | **1.54 bits** | 2.46 bits (62%) |

**Fonte:** [ECQ - GitHub drxddy/ecq](https://github.com/drxddy/ecq)

#### MLX Feature Request -- Medicoes em Llama/Qwen:

| Bits Alocados | Entropia Medida | Lacuna |
|:---:|:---:|:---:|
| 8 bits | ~4-5 bits | 37-50% |
| 4 bits | ~2.17 bits | 46% |
| 3 bits | ~1.8 bits | 40% |
| 2 bits | ~1.5 bits | 25% |

**Fonte:** [MLX Issue #3043](https://github.com/ml-explore/mlx/issues/3043)

#### BBQ (Bell Box Quantization) -- Entropia por Tipo de Quantizacao:

| Modelo | Bits | Entropia BBQ | Entropia QuEST |
|--------|:---:|:---:|:---:|
| LLaMA-95M | 4 bits | 3.93 bits | 3.61 bits |
| LLaMA-125M | 4 bits | 3.93 bits | 3.61 bits |
| LLaMA-300M | 2 bits | 1.98 bits | ~1.93 bits |

**Nota:** BBQ maximiza entropia propositalmente para melhorar acuracia. A entropia de ~3.93 para 4 bits e alta, indicando uso eficiente da capacidade de representacao. A lacuna aqui e menor (~0.07 bits) -- mas isso reflete uma filosofia oposta: maximizar entropia para acuracia vs. minimizar entropia para compressao.

**Fonte:** [BBQ - arxiv 2603.01599](https://arxiv.org/html/2603.01599)

### 2.2 Interpretacao da Lacuna de Entropia

A distribuicao de pesos quantizados e altamente **nao-uniforme** -- tipicamente em formato de sino (bell-curved), concentrada ao redor do zero. Isso significa que:

1. **Poucos simbolos dominam a distribuicao** (pesos proximos de zero)
2. **Simbolos raros desperdicam bits** (pesos distantes de zero recebem os mesmos bits dos frequentes)
3. **Codificacao de comprimento variavel** pode explorar essa assimetria

Para pesos de 4 bits com entropia real de ~1.12 bits:
- Taxa de compressao teorica: 4/1.12 = **3.57x**
- Compressao pratica medida (ECQ): **3.27x** (proximo do limite teorico)

### 2.3 Entropia em BFloat16 (Nao-Quantizado)

O DFloat11 mediu a entropia dos componentes de BFloat16:

| Componente | Bits Alocados | Entropia Real |
|-----------|:---:|:---:|
| Expoente | 8 bits | ~2.6 bits |
| Sinal | 1 bit | ~1 bit |
| Mantissa | 7 bits | ~7 bits |

Resultado: compressao lossless de ~30% (de 16 para ~11 bits efetivos) via Huffman coding nos expoentes. Bits comprimidos variam de 10.87 a 11.49 entre diferentes modelos.

**Fonte:** [DFloat11 - arxiv 2504.11651](https://arxiv.org/abs/2504.11651) (NeurIPS 2025)

---

## 3. Asymmetric Numeral Systems (ANS) para Deploy Neural

### 3.1 Visao Geral do ANS

ANS (Jarek Duda, 2014) combina a **razao de compressao** da codificacao aritmetica com o **custo computacional** da codificacao Huffman. Dois variantes principais:

- **rANS (Range ANS):** Operacoes aritmeticas simples, altamente paralelizavel via SIMD
- **tANS (Table ANS):** Decodificacao via lookup table, menos cache-friendly

ANS foi adotado por: Facebook (Zstandard), Apple (LZFSE), Google (Draco).

### 3.2 Velocidade de Decodificacao

#### Hypersonic rANS (CPU, single-thread):

| Implementacao | Throughput | Requisitos |
|--------------|-----------|-----------|
| rANS32x64 16w (melhor) | **3,018 MiB/s** (~3.0 GB/s) | AVX2 |
| rANS32x32 (variantes) | ~2,500 MiB/s | AVX2 |
| Multi-thread (melhor) | **18,035 MiB/s** (~18 GB/s) | AVX2 + multi-core |

- ~1.42 ciclos de clock por byte
- 32 ou 64 streams interleaved
- AVX2 e essencial; AVX-512 nao demonstrou vantagem

**Fonte:** [hypersonic-rANS](https://github.com/rainerzufalldererste/hypersonic-rANS)

#### Comparativo:

| Metodo | Throughput Tipico | Notas |
|--------|:---:|------|
| rANS (SIMD) | ~1,500 MB/s por core | 32 streams interleaved |
| tANS | ~500 MB/s por core | Lookup table, menos cache-friendly |
| Huffman | ~800 MB/s por core | Mais simples, menos compressao |
| CABAC | ~10-15 Mb/s (software) | Serial, dificil de paralelizar |

### 3.3 ANS em GPU

- **Decodificacao massivamente paralela de ANS em GPU** e tema de pesquisa ativo
- Apos aceleracao GPU de inferencia neural, codificacao por entropia se torna gargalo dominante (73% do runtime)
- tANS com lookup tables estaticas reduz latencia em **77%** comparado a rANS em cenarios edge-cloud

### 3.4 rANS para Compressao de Features Neurais

Aplicacao pratica recente (2025):
- Compressao de features intermediarias para split computing de DNNs
- Tempo de codificacao: **0.705-0.739 ms**
- Tempo de decodificacao: **0.479-0.517 ms**
- Compressao: **7.2x** vs. serializacao binaria baseline
- **2.8x melhor** que DietGPU otimizado
- Mantem acuracia dentro de +/-0.2% do baseline

**Fonte:** [rANS Split Computing - arxiv 2511.11664](https://arxiv.org/abs/2511.11664)

### 3.5 Viabilidade para Inferencia em Tempo Real

**Conclusao: Sim, ANS e viavel para inferencia em tempo real.**

Para um modelo de 7B parametros em Q4 (~3.5 GB nao-comprimido):
- Com rANS a 3 GB/s: decodificacao completa em **~1.2 segundos**
- Na pratica, decodificacao e feita por bloco/camada durante inferencia
- Kernel fusionado (decode+GEMM) evita materializar pesos em memoria
- ECQ demonstra: 7B no M3 Pro passa de 42.9 tok/s (Q4 padrao) para **140.0 tok/s** com entropy coding

---

## 4. Compressores Gerais (Zstd, LZ4) em Pesos Quantizados

### 4.1 Zstandard (Zstd)

Zstandard combina compressao estilo LZ77 com Finite State Entropy (FSE, baseado em tANS):

**Resultados em modelos de IA (ZipNN):**

| Tipo de Modelo | Compressao Adicional | Notas |
|---------------|:---:|------|
| BFloat16 (limpo) | 34% reducao | 4.6x mais rapido que zstd puro |
| BFloat16 (geral) | 17% melhoria vs. zstd | 62% speedup em compress/decompress |
| GPTQ/AWQ quantizado | 85-91% do original | 9-15% de headroom |
| **GGUF quantizado** | **Nenhuma compressao** | Formato ja altamente empacotado |

**Fonte:** [ZipNN - arxiv 2411.05239](https://arxiv.org/html/2411.05239v2)

### 4.2 Por que GGUF Nao Comprime com Zstd

A descoberta critica do ZipNN: **modelos GGUF nao comprimem de forma alguma com compressores gerais**.

Razoes provaveis:
1. **Bit-packing agressivo:** Pesos de 4 bits sao empacotados contiguamente sem padding
2. **Hierarquia de escala quantizada:** Super-blocos de 256 pesos com sub-escalas quantizadas
3. **Sem redundancia de formato:** Diferente de BFloat16 (onde expoentes tem entropia baixa), os bits empacotados do GGUF parecem pseudo-aleatorios para compressores baseados em LZ77
4. **Granularidade fina de escala:** Fatores de escala ja codificam informacao local

**Implicacao crucial:** Compressores gerais falham em GGUF, mas isso **NAO** significa que nao ha entropia residual. Significa que a redundancia esta na **distribuicao estatistica dos simbolos**, nao em padroes repetitivos de bytes -- exatamente o dominio da codificacao por entropia.

### 4.3 GPTQ/AWQ vs. GGUF

- GPTQ e AWQ mantem headroom de 9-15% porque usam formatos padrao (int4 em tensores PyTorch)
- GGUF elimina redundancia de formato mas **nao aplica codificacao por entropia sobre os simbolos quantizados**
- A lacuna de entropia de ~1.12-2.17 bits em pesos de 4 bits persiste em GGUF

---

## 5. Modelagem de Contexto para Entropia de Pesos

### 5.1 Analogia com Compressao de Texto (PAQ/PPM)

Em compressores de texto de ponta (PAQ, PPM):
- **Modelos de contexto** predizem o proximo simbolo baseado nos anteriores
- **Context mixing** combina predicoes de multiplos modelos com pesos adaptativos
- Predicao bit-a-bit com codificacao aritmetica do residuo
- Ajuste de pesos via gradiente para favorecer modelos mais precisos

Analogia com pesos neurais:
- Pesos adjacentes em uma mesma camada/filtro provavelmente sao correlacionados
- Predicao de peso[i+1] a partir de peso[i] (e contexto da camada) poderia reduzir entropia do residuo
- O "erro de predicao" seria codificado por entropia em vez do peso bruto

### 5.2 Correlacao Sequencial em Matrizes de Peso

Evidencias de redundancia sequencial:
- Redes neurais sao tipicamente **sobre-parametrizadas** com redundancia significativa
- **Decomposicao de baixo rank** mostra que matrizes de peso possuem rank efetivo muito menor que rank completo
- Correlacoes espaciais entre pesos adjacentes existem e sao exploraveis via decomposicao tensorial (MPO bond dimensions)
- Esparsidade de 15-40% pode ser alcancada via perturbacao de pesos sem perda significativa

### 5.3 Neural Weight Compression (NWC) -- Abordagem por Autoencoder

O NWC (2025) treina codecs neurais diretamente sobre pesos pre-treinados:

- Processamento **coluna-a-coluna sequencial** com feedback de erro
- Autoencoder captura propriedades distribucionais dos pesos
- Perda com pesos de importancia para alocacao adaptativa de qualidade
- Em 4-6 bits: desempenho quase equivalente ao FP16

**Insight chave:** Para distribuicoes de cauda pesada (tipicas de pesos neurais), o autoencoder aprende melhor aproximacao que codecs baseados em Gaussiana, capturando comportamento nas caudas.

**Fonte:** [NWC - arxiv 2510.11234](https://arxiv.org/html/2510.11234v1)

### 5.4 EntroLLM -- Maximizando Compressibilidade via Quantizacao Mista

EntroLLM usa quantizacao a nivel de tensor (nao de bloco) para criar distribuicoes "mais pontudas" (spikier) com menor entropia:

- Escolha por camada entre quantizacao simetrica e assimetrica
- Distribuicoes mais concentradas -> menor entropia -> melhor compressao Huffman
- Reducao de bitwidth medio para **1.39 bits** em pesos de 4 bits
- Melhoria de **11.3x** sobre estado-da-arte

**Fonte:** [EntroLLM - arxiv 2505.02380](https://arxiv.org/html/2505.02380)

### 5.5 Potencial de Predicao de Contexto para Pesos

**Area subexplorada.** A pesquisa encontrou:
- Nenhum trabalho aplicando diretamente PPM/PAQ-style context mixing a pesos neurais
- DeepCABAC usa modelagem de contexto binario (heranca do H.264), mas limitada
- NWC usa autoencoder (predicao implicita), nao predicao sequencial explicita
- Delta encoding entre pesos adjacentes (Delta-DNN) existe para compressao de ativacoes, nao de pesos estaticos

**Oportunidade:** Aplicar modelagem de contexto estilo compressores de texto (predicao do proximo peso baseado em vizinhos da mesma camada/filtro/canal) pode reduzir ainda mais a entropia do residuo alem do que codificacao por entropia simples alcanca.

---

## 6. Especificidades do Formato GGUF

### 6.1 Estrutura do Arquivo GGUF

GGUF (Georgi Gerganov Universal Format) consiste em:
1. **Header** (24 bytes): magic number, versao, contadores
2. **Metadata** (key-value pairs): arquitetura, tokenizer, parametros de quantizacao
3. **Padding de alinhamento** (32 bytes)
4. **Dados dos tensores**: pesos quantizados + fatores de escala

### 6.2 Estrutura de Blocos de Quantizacao

#### Q4_0 (Formato Legado):
- Blocos de 32 pesos
- 1 fator de escala FP16 por bloco
- `peso' = escala x q` (simetrico)
- ~4.5 bits efetivos por peso (contando overhead de escala)

#### Q4_K (K-Quant):
- **Super-blocos de 256 pesos** (16 sub-blocos de 16 valores)
- 12 escalas de 6 bits + 4 escalas de 4 bits (quantizadas)
- 8 valores de correcao de offset de 6 bits
- Super-escala `d` (FP16) escala as escalas quantizadas
- Super-minimo `dmin` (FP16) escala os minimos quantizados
- ~4.5 bits efetivos por peso (Q4_K_S) a ~4.83 bits (Q4_K_M)

#### IQ4_XS (I-Quant):
- Super-blocos de 256 pesos
- Usa **importancia matrix** para alocacao adaptativa
- Quantizacao nao-linear
- ~4.32 bits efetivos por peso
- Supera Q4_K_M em qualidade com menor footprint

### 6.3 Codificacao por Entropia no GGUF Atual

**NAO EXISTE.** O formato GGUF atual:
- Nao aplica Huffman coding
- Nao aplica codificacao aritmetica
- Nao aplica ANS
- Nao aplica nenhuma forma de codificacao por entropia
- Compressao depende exclusivamente de:
  - Bit-packing dos valores quantizados
  - Quantizacao hierarquica de fatores de escala (K-Quants)
  - Mapeamento nao-linear (IQ4_NL)
  - Alocacao adaptativa via importancia matrix (I-Quants)

### 6.4 O que Seria Necessario para Adicionar Entropy Coding ao llama.cpp

Baseado na analise da arquitetura:

1. **Codificacao (offline, na quantizacao):**
   - Calcular distribuicao de frequencia dos simbolos por tensor/camada
   - Construir tabelas ANS (tANS) ou arvore Huffman
   - Codificar pesos com comprimento variavel
   - Armazenar tabelas de frequencia + stream codificado no GGUF

2. **Decodificacao (runtime, na inferencia):**
   - Decodificar blocos sob demanda durante carregamento ou inferencia
   - Opcao A: Decodificar para buffer intermediario (simples, mais memoria)
   - Opcao B: Kernel fusionado decode+GEMM (complexo, menor memoria)
   - Opcao C: Decodificacao lazy por camada com cache (compromisso)

3. **Mudancas no formato GGUF:**
   - Novo tipo de quantizacao (ex: Q4_K_E para entropy-coded)
   - Metadata adicional: tabelas de frequencia, tamanho dos streams
   - Ponteiros de offset para acesso aleatorio por tensor

4. **Desafios:**
   - Acesso aleatorio: decodificacao sequencial do ANS vs. necessidade de acesso por bloco
   - Latencia de primeira inferencia (decodificacao inicial)
   - Compatibilidade retroativa com GGUF existente
   - Complexidade de implementacao no ecossistema ggml

### 6.5 Headroom Estimado em GGUF

Combinando os dados de entropia com a estrutura GGUF:

| Formato GGUF | Bits/Peso | Entropia Estimada | Compressao Potencial |
|-------------|:---:|:---:|:---:|
| Q4_K_M | 4.83 | ~2.0-2.5 | **1.9-2.4x** |
| Q4_0 | 4.50 | ~1.5-2.2 | **2.0-3.0x** |
| IQ4_XS | 4.32 | ~2.0-2.5 | **1.7-2.2x** |
| Q3_K_M | 3.91 | ~1.8-2.2 | **1.8-2.2x** |
| Q2_K | 3.35 | ~1.5-1.8 | **1.9-2.2x** |
| IQ2_XS | 2.31 | ~1.5-1.7 | **1.4-1.5x** |

**Nota:** Os IQ-Quants com importancia matrix provavelmente tem entropia mais proxima dos bits alocados (distribuicao mais uniforme por design), portanto menor headroom.

---

## 7. DeepCABAC em Detalhe

### 7.1 Mecanismo Tecnico

DeepCABAC herda o CABAC do H.264/AVC:

1. **Binarizacao:** Pesos quantizados sao convertidos em sequencia binaria
2. **Modelagem de contexto:** Probabilidade de cada bit e estimada com base em bits anteriores
3. **Codificacao aritmetica adaptativa:** Codifica bits com comprimento proporcional ao conteudo de informacao
4. **Quantizacao rate-distortion:** Minimiza distorcao (erro na saida da rede) para um dado budget de bits

### 7.2 Resultados Detalhados

| Modelo | Tamanho Original | Taxa Compressao | Tamanho Final | Acuracia |
|--------|:---:|:---:|:---:|:---:|
| VGG16 (ImageNet) | ~553 MB | **63.6x** | 8.7 MB | Sem perda |
| BERT (QA) | ~440 MB | ~11x | ~40 MB | F1=86% |
| Compressao NNC (geral) | 100% | 20x+ | <5% | Mantida |

### 7.3 Conexao com Nosso Trabalho (DCT-Quantization)

**Relevancia direta:**

1. **Pipeline complementar:** DCT transforma pesos no dominio de frequencia -> quantizacao dos coeficientes -> codificacao por entropia do resultado. Exatamente o pipeline JPEG mas para pesos neurais.

2. **Compressao em cascata:**
   - DCT decorrelaciona pesos (remove redundancia espacial)
   - Quantizacao reduz precisao (compressao lossy)
   - Entropy coding remove redundancia estatistica (compressao lossless)
   - Cada estagio e independente e aditivo

3. **Transform coding para redes neurais** (Ulbricht et al., 2018) ja demonstrou:
   - DCT com tamanho de bloco igual ao tamanho do filtro (ex: 7x7 DCT para filtro 7x7)
   - 5-6 bits/coeficiente com impacto minimo na acuracia
   - Diferenca do JPEG: todos coeficientes quantizados igualmente (sem matriz de quantizacao com penalidade em altas frequencias)

4. **Oportunidade composta:** Se DCT reduz entropia dos coeficientes (via compactacao de energia), a codificacao por entropia subsequente pode ser ainda mais eficiente que aplicada diretamente aos pesos brutos.

**Fonte:** [Neural Network Compression using Transform Coding - arxiv 1805.07258](https://arxiv.org/pdf/1805.07258)

---

## 8. Resultados Praticos: ECQ e EntroLLM

### 8.1 ECQ (Entropy Coded Quantization) -- Resultados Concretos

Implementacao pratica com rANS em Apple Silicon:

| Modelo | Tamanho Q4 | Tamanho ECQ | Compressao | Tokens/s Q4 | Tokens/s ECQ |
|--------|:---:|:---:|:---:|:---:|:---:|
| 7B (M3 Pro) | 3.5 GB | 1.07 GB | **3.27x** | 42.9 | **140.0** |
| Qwen2.5-1.5B | -- | -- | 3.58x | -- | -- |
| Qwen2.5-0.5B | -- | -- | 3.47x | -- | -- |
| GPT-2 | -- | -- | 3.42x | -- | -- |

**Zero perda de qualidade** -- compressao puramente lossless.

A aceleracao de 3.27x nos tokens/s e resultado direto da reducao de bandwidth de memoria: menos bytes para ler = mais rapido em hardware limitado por memoria.

**Fonte:** [ECQ - GitHub drxddy/ecq](https://github.com/drxddy/ecq)

### 8.2 EntroLLM -- Resultados em Dispositivos Edge

| Modelo | Economia uint8 | Economia uint4 | Speedup Inferencia |
|--------|:---:|:---:|:---:|
| smolLM-1.7B | ate 30% | ate 65% | 31.9-146.6% |
| phi3-mini-4k (3.8B) | ate 30% | ate 65% | 31.9-146.6% |
| mistral-7B | ate 30% | ate 65% | 31.9-146.6% |

Tempo de decodificacao no Jetson P3450:
- 4-bit: 1.66 segundos
- 8-bit: 6.66 segundos

**Fonte:** [EntroLLM - arxiv 2505.02380](https://arxiv.org/abs/2505.02380)

### 8.3 DFloat11 -- Compressao Lossless de BFloat16

- Compressao de ~30% (16 bits -> ~11 bits efetivos)
- **Bit-for-bit identico** ao modelo original
- GPU kernel customizado para decompressao online
- Llama 3.1 405B (810 GB) roda em 8x80GB GPUs (antes impossivel)
- 2.3-46.2x maior throughput vs. offload para CPU
- **Aceito no NeurIPS 2025**

**Fonte:** [DFloat11 - arxiv 2504.11651](https://arxiv.org/abs/2504.11651)

---

## 9. Sintese e Implicacoes para o Projeto DCT-Quantization

### 9.1 Oportunidade Principal

A combinacao de **DCT + quantizacao + codificacao por entropia** forma um pipeline analogo ao JPEG para pesos neurais:

```
Pesos originais (FP16/BF16)
    |
    v
DCT por bloco (decorrelacao)
    |
    v
Quantizacao de coeficientes (compressao lossy, 4-6 bits)
    |
    v
Codificacao por entropia - rANS/Huffman (compressao lossless, ~1.5-2.5 bits efetivos)
    |
    v
Peso comprimido final
```

### 9.2 Ganhos Estimados

| Estagio | Bits/Peso | Reducao |
|---------|:---:|:---:|
| Original (BF16) | 16.0 | 1.0x |
| Quantizacao direta (Q4) | 4.0-4.8 | 3.3-4.0x |
| Entropy coding sobre Q4 | 1.5-2.5 | **6.4-10.7x** |
| DCT + Quant + Entropy (estimado) | 1.2-2.0 | **8.0-13.3x** |

A **DCT pode reduzir entropia alem da quantizacao direta** por compactar energia nos primeiros coeficientes, criando distribuicao ainda mais concentrada (e portanto com menor entropia) para a etapa de codificacao.

### 9.3 Recomendacoes Praticas

1. **Medir entropia real dos coeficientes DCT quantizados** dos nossos experimentos
2. **Implementar rANS** (nao tANS) -- superior em throughput com SIMD moderno
3. **Usar codificacao por tensor/camada** para permitir acesso aleatorio
4. **Comparar com ECQ como baseline** -- ja demonstra 3.27x em Q4 sem DCT
5. **Investigar modelagem de contexto simples:** predicao do coeficiente DCT atual baseado nos coeficientes anteriores do mesmo bloco pode reduzir entropia do residuo
6. **Considerar integrar com GGUF** adicionando novos tipos de quantizacao

### 9.4 Riscos e Limitacoes

- **CABAC e lento** em software (~10-15 Mb/s) -- preferir rANS (~3 GB/s)
- **IQ-Quants de GGUF** ja otimizam distribuicao para acuracia, potencialmente reduzindo headroom de entropia
- **Decodificacao sequencial de ANS** conflita com acesso aleatorio -- mitigavel com codificacao independente por bloco/tensor
- **Overhead das tabelas de frequencia** pode anular ganho em tensores pequenos
- **Complexidade de implementacao** em kernels GPU fusionados e significativa

---

## 10. Tabela de Trabalhos Relacionados

| Trabalho | Ano | Metodo | Compressao | Alvo |
|----------|:---:|--------|:---:|------|
| Deep Compression (Han) | 2016 | Poda+Quant+Huffman | 35-49x | CNN classicas |
| DeepCABAC (Fraunhofer) | 2019 | CABAC aritmetico | 63.6x | CNNs, BERT |
| NNC ISO/IEC 15938-17 | 2022/2024 | Pipeline completo | <5% tamanho | Padrao internacional |
| CERWU | 2025 | Rate-constrained+EC | 20-40% melhor que NNC | Redes pre-treinadas |
| ZipNN | 2024 | Zstd otimizado | ~30% BF16 / 0% GGUF | Armazenamento |
| DFloat11 | 2025 | Huffman em expoentes | ~30% BF16 lossless | GPU inference |
| ECQ | 2025 | rANS em Q4 | **3.27x sobre Q4** | Apple Silicon |
| EntroLLM | 2025 | Huffman+quant mista | **11.3x sobre SotA** | Edge devices |
| BBQ | 2026 | Max entropy quant | Melhora acuracia | QAT |
| NWC | 2025 | Autoencoder neural | Competitivo 4-6 bits | LLMs |
| Transform Coding (Ulbricht) | 2018 | DCT+Quant | 5-6 bits/coef | CNNs |

---

## 11. Conclusao

A codificacao por entropia e uma **oportunidade massiva e comprovada** para compressao adicional de pesos quantizados de LLMs. Os dados empiricos sao inequivocos:

- **Pesos de 4 bits tem entropia real de 1.12-2.17 bits** (45-72% desperdicado)
- **rANS pode comprimir 3.27x adicionalmente** sem perda (ECQ, testado)
- **Decodificacao a 3 GB/s** e viavel em CPU moderna com AVX2
- **GGUF nao aplica nenhuma codificacao por entropia** -- headroom intacto
- **DCT + entropy coding** e o pipeline natural, analogo ao JPEG

A integracao de codificacao por entropia no pipeline DCT-Quantization pode potencialmente reduzir o tamanho efetivo de modelos para **1.2-2.0 bits por peso** partindo de 4 bits quantizados, representando compressao total de **8-13x** sobre o modelo original em BFloat16.

---

## Fontes Principais

- [Deep Compression - Han et al. (ICLR 2016)](https://arxiv.org/abs/1510.00149)
- [DeepCABAC - Wiedemann et al. (2019)](https://arxiv.org/abs/1907.11900)
- [Fraunhofer HHI - Neural Network Coding](https://www.hhi.fraunhofer.de/en/departments/ai/research-groups/efficient-deep-learning/research-topics/neural-network-compression.html)
- [NNCodec - GitHub](https://github.com/fraunhoferhhi/nncodec)
- [ISO/IEC 15938-17:2024](https://www.iso.org/standard/85545.html)
- [CERWU - Rate-Constrained Quantization (2025)](https://arxiv.org/html/2505.18758)
- [ECQ - Entropy Coded Quantization](https://github.com/drxddy/ecq)
- [EntroLLM (2025)](https://arxiv.org/abs/2505.02380)
- [DFloat11 (NeurIPS 2025)](https://arxiv.org/abs/2504.11651)
- [BBQ - Bell Box Quantization (2026)](https://arxiv.org/html/2603.01599)
- [NWC - Neural Weight Compression (2025)](https://arxiv.org/html/2510.11234v1)
- [ZipNN (2024)](https://arxiv.org/html/2411.05239v2)
- [hypersonic-rANS](https://github.com/rainerzufalldererste/hypersonic-rANS)
- [rANS Split Computing (2025)](https://arxiv.org/abs/2511.11664)
- [MLX rANS Feature Request](https://github.com/ml-explore/mlx/issues/3043)
- [Transform Coding for NNs (2018)](https://arxiv.org/pdf/1805.07258)
- [MEC-Quant (2025)](https://arxiv.org/html/2509.15514v1)
- [GGUF Quantization Overview](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9)
- [PAQ Compression](https://mattmahoney.net/dc/paq.html)
