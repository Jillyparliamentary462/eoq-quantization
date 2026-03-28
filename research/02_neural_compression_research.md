# Pesquisa: Compressao Neural e Suas Aplicacoes a Quantizacao de Pesos de LLMs

**Data:** 28 de marco de 2026
**Objetivo:** Mapear o estado da arte em compressao neural (imagens/video) e identificar tecnicas transferiveis para compressao de pesos de redes neurais, com foco no projeto DCT.

---

## Sumario Executivo

Esta pesquisa revela uma convergencia profunda entre compressao neural de imagens/video e compressao de pesos de redes neurais. O paper **"Neural Weight Compression for Language Models" (NWC, outubro 2025)** demonstra explicitamente que a arquitetura autoencoder + modelo de entropia + codificacao aritmetica -- usada em compressao de imagens -- pode ser aplicada diretamente a pesos de LLMs, atingindo estado da arte em 4-6 bits. Alem disso, metodos como AQLM, QuIP#, QTIP, VPTQ e LLVQ ja utilizam conceitos de quantizacao vetorial oriundos da teoria de compressao. A existencia do padrao ISO/IEC 15938-17 (NNCodec/DeepCABAC) para compressao de redes neurais confirma que esta e uma area madura e em rapida evolucao.

---

## 1. Compressao Neural de Imagens: Fundamentos

### 1.1 Balle et al. (2017) -- "End-to-end Optimized Image Compression"

**Como funciona:**
O modelo consiste em tres componentes: (1) uma **transformada de analise nao-linear** (encoder), (2) um **quantizador uniforme**, e (3) uma **transformada de sintese nao-linear** (decoder). A arquitetura usa camadas convolucionais com a nao-linearidade **GDN (Generalized Divisive Normalization)**, inspirada em modelos de neuronios do sistema visual biologico.

**Inovacoes-chave:**
- Treinamento end-to-end por gradiente estocastico, usando um proxy continuo para a funcao de perda descontinua do quantizador (ruido uniforme durante treino: y_hat = y + epsilon, epsilon ~ U[-1/2, 1/2])
- Otimizacao conjunta de rate-distortion: `L = R_y + lambda * D(x, x_hat)`
  - R_y = soma_i(-log2(p_yi(y_i))) (taxa em bits)
  - D = erro quadratico medio (MSE) ou MS-SSIM
- Sob certas condicoes, pode ser interpretado como um VAE (Variational Autoencoder)
- Supera JPEG e JPEG 2000 em rate-distortion em todas as taxas de bits

**Referencia:** Balle, Laparra, Simoncelli. "End-to-end Optimized Image Compression." ICLR 2017.
**Link:** https://arxiv.org/abs/1611.01704

### 1.2 Scale Hyperprior (Balle et al., 2018)

**Como funciona:**
Adiciona um **hyperprior** ao modelo basico -- essencialmente um segundo autoencoder que opera sobre a representacao latente. A ideia e capturar dependencias espaciais que o modelo basico nao consegue modelar.

**Arquitetura (4 componentes):**
1. **Transformada de analise (g_a):** Imagem -> representacao latente y (4 camadas conv + GDN, ~1.5-3.5M parametros)
2. **Hyperencoder (h_a):** y -> z (representacao compacta das estatisticas de y)
3. **Hyperdecoder (h_s):** z_hat -> parametros sigma_i (escalas da distribuicao Gaussiana de y)
4. **Transformada de sintese (g_s):** y_hat -> imagem reconstruida (conv transposta + IGDN)

**Modelo de entropia:**
- A distribuicao de cada elemento y_i e modelada como Gaussiana com media zero e escala sigma_i
- sigma_i vem do hyperdecoder, condicionado na informacao lateral z
- z e quantizado e codificado com modelo de entropia fatorizado (nao-parametrico)
- Codificacao aritmetica (ou rANS) comprime y_hat usando as distribuicoes estimadas

**Por que funciona melhor que JPEG/H.265:**
- Modelo de entropia **adaptativo e aprendido** (vs. tabelas fixas do JPEG)
- Transformada **nao-linear e aprendida** (vs. DCT linear e fixa)
- Otimizacao **end-to-end** para rate-distortion (vs. pipeline separado)
- Hyperprior captura **dependencias espaciais** das estatisticas latentes

**Referencia:** Balle, Minnen, Singh, Hwang, Johnston. "Variational Image Compression with a Scale Hyperprior." ICLR 2018.
**Link:** https://arxiv.org/abs/1802.01436

### 1.3 Modelos Avancados de Entropia

**Autoregressive + Hyperprior (Minnen et al., 2018):**
- Combina hyperprior com contexto autoregressivo (vizinhos ja decodificados)
- Parametros mu_i e sigma_i condicionados em ambos hyperprior e contexto causal
- Ganho significativo em compressao, porem decodificacao sequencial (lenta)

**Variantes de paralelismo:**
- **Checkerboard:** Decodifica em 2 passos (ancora + nao-ancora), melhor paralelismo
- **Channel-conditional:** Grupos de canais de tamanho desigual, decodifica menores primeiro

**Implementacao de referencia:** CompressAI (PyTorch) -- biblioteca open-source com todas as arquiteturas implementadas.
**Link:** https://github.com/InterDigitalInc/CompressAI

---

## 2. Compressao Neural de Video

### 2.1 SSF -- Scale-Space Flow (Agustsson et al., CVPR 2020)

Generaliza fluxo optico adicionando um parametro de escala para lidar com desoclusoes e movimentos rapidos. Arquitetura simples: 4 camadas CNN para encoder/decoder, sem redes de fluxo optico pre-treinadas.

**Pipeline:**
1. Estima e codifica o scale-space flow (campo de fluxo + escala)
2. Warpa o frame anterior usando o fluxo decodificado -> predicao
3. Codifica o residual (diferenca entre original e predicao)
4. Decodifica residual + predicao -> reconstrucao final

### 2.2 DCVC -- Deep Contextual Video Compression (Microsoft, 2021-2025)

**Inovacao paradigmatica:** Substitui codificacao de residuos (pixel-wise) por **codificacao condicional** -- o contexto extraido de frames anteriores e usado como condicao para encoder, decoder e modelo de entropia.

**Evolucao da familia:**
| Versao | Ano | Inovacao |
|--------|-----|----------|
| DCVC | 2021 | Paradigma condicional (substituiu residual coding) |
| DCVC-TCM | 2022 | Mineracao de contexto temporal multi-escala |
| DCVC-HEM | 2022 | Modelo de entropia hibrido espaco-temporal |
| DCVC-DC | 2023 | Contextos diversos em dimensoes temporal e espacial |
| DCVC-RT | 2024 | **125 fps encode / 113 fps decode para 1080p, -21% bitrate vs. H.266/VTM** |

**Relevancia para DCT project:** DCVC-RT e o primeiro codec neural a atingir tempo real em 1080p e 4K, demonstrando que codecs neurais ja sao praticos. A abordagem condicional (em vez de residual) e uma inspiracao para modelar dependencias entre camadas de um LLM.

**Link:** https://github.com/microsoft/DCVC

---

## 3. Compressao Neural Aplicada a PESOS de Redes Neurais

### 3.1 NWC -- Neural Weight Compression (outubro 2025) *** PAPER CENTRAL ***

Este e o paper mais relevante para nosso projeto. Aplica explicitamente a arquitetura de compressao neural de imagens a pesos de LLMs.

**Arquitetura:**
```
Pesos originais -> Preprocessing -> Encoder Neural (analise) -> Quantizacao ->
Entropy Coding (aritmetico) -> Bitstream comprimido

Bitstream -> Entropy Decoding -> Decoder Neural (sintese) -> Pesos reconstruidos
```

**Tres componentes tecnicos:**

1. **Chunk-and-Normalize (Preprocessamento):**
   - Pesos particionados column-wise em chunks de tamanho fixo (d=16)
   - Cada coluna normalizada para desvio padrao unitario
   - Overhead: ~0.004 bits/parametro (desprezivel)
   - Resolve heterogeneidade de formas de tensores

2. **Importance-Aware Training:**
   - Metrica de distorcao baseada em Hessiana (aproxima impacto downstream)
   - Loss: `L_rate + lambda * lambda_I(i) * MSE(w, w_hat)`
   - Niveis de importancia variam por chunk
   - Amostragem aleatoria de importancia durante treino -> compressao de qualidade variavel
   - Na inferencia, niveis atribuidos com base nas diagonais da Hessiana (dados de calibracao)

3. **Compensacao de Erro em Tempo de Inferencia:**
   - **Intra-layer feedback:** Atualiza colunas nao-comprimidas para compensar erros anteriores
   - **Inter-layer fine-tuning:** Ajustes por bloco para erros de outras camadas

**Resultados-chave:**
- Estado da arte em tradeoff accuracy-compressao a **4-6 bits**
- Supera GPTQ, AWQ, QuIP# em taxas medias
- Generaliza para modelos de visao (CLIP, SigLIP, DINOv2)
- Encoder/decoder: MLPs de 4 camadas, largura 512
- Treinado nos pesos do Llama 3-8B (~6.8M exemplos), ~11.5h em uma GPU A6000
- **Kurtosis reduzida de 20.48 para 0.00** (vs. Hadamard que nao elimina totalmente)

**Insight crucial para nosso projeto:** A transformada aprendida (encoder neural) suprime outliers e caudas pesadas **muito melhor** que a transformada de Hadamard usada em QuIP#/QTIP. Isso sugere que uma DCT ou transformada aprendida poderia ser superior ao Hadamard para preprocessamento de pesos.

**Latencia competitiva:**
- Encoding: 1.64ms (vs. GPTQ 0.69ms, QTIP 19.84ms)
- Decoding: 1.17ms via GPU-accelerated entropy decoding

**Referencia:** https://arxiv.org/abs/2510.11234

### 3.2 NNCodec / DeepCABAC -- Padrao ISO/IEC 15938-17

**O que e:** Primeiro padrao internacional para compressao de redes neurais (ISO/IEC 15938-17:2022, tambem conhecido como MPEG-7 parte 17).

**Componentes:**
- **DeepCABAC:** Codificador aritmetico context-adaptive que adapta modelos de probabilidade binaria on-the-fly as estatisticas dos pesos
- Quantizacao eficiente + codificacao de entropia
- Comprime redes neurais para **menos de 5% do tamanho original** sem degradar performance

**Resultado notavel:** O comprimento medio de codeword do NNCodec frequentemente fica **abaixo do limite de entropia de Shannon** (graças a modelagem de contexto adaptativa).

**2a Edicao:** Adicionou sparsificacao estruturada e adaptacao temporal no DeepCABAC.

**Referencia:** https://github.com/fraunhoferhhi/nncodec

### 3.3 Rate-Constrained Quantization + Entropy Coding (maio 2025)

**Abordagem:** Combina quantizacao com consciencia de taxa (rate-aware) usando o metodo Optimal Brain Surgeon adaptado com regularizacao de entropia.

**Framework:** Minimiza `D(W,W_hat) + lambda * R(W_hat)` onde D e distorcao na saida da camada e R e informacao em bits.

**Resultado:** 20-40% reducao de taxa em relacao ao NNCodec padrao com mesma performance.

**Insight:** "O codificador de entropia teoricamente otimo atinge dentro de ~1 bit do limite teorico." Na pratica, codificadores como DeepCABAC atingem taxas de compressao que correspondem ao conteudo de informacao previsto.

**Referencia:** https://arxiv.org/abs/2505.18758

### 3.4 EntroLLM -- Entropy Encoded Weight Compression (maio 2025)

**Para edge devices:** Combina quantizacao mista (unsigned/assimetrica por tensor) com codificacao Huffman.

**Resultados:**
- Pesos 8-bit comprimem para **5.58 bits efetivos**
- Pesos 4-bit comprimem para **1.39 bits efetivos**
- Ate 30% economia vs. modelos uint8, 65% vs. uint4
- Decodificacao paralela: 1.66s (uint4) no NVIDIA Jetson

**Referencia:** https://arxiv.org/abs/2505.02380

---

## 4. Codificacao de Entropia: ANS e Aritmetica para Pesos

### 4.1 Asymmetric Numeral Systems (ANS)

**Criador:** Jaroslaw (Jarek) Duda, Universidade Jaguelonica (2014).

**Por que e relevante:**
- Combina a **taxa de compressao** da codificacao aritmetica com a **velocidade** da codificacao Huffman
- Estado unico (um numero natural) em vez de intervalo (dois numeros na aritmetica)
- ~50% mais rapido para decodificar que Huffman para alfabeto de 256 simbolos
- E o metodo padrao em propostas recentes de compressao baseada em ML

**Variantes:**
| Variante | Caracteristica | Trade-off |
|----------|---------------|-----------|
| **rANS** (Range ANS) | Usa multiplicacao, aceita modelos adaptativos | Mais flexivel, ligeiramente mais lento |
| **tANS** (Table ANS) | Lookup tables, sem multiplicacao | 77% menos latencia que rANS, porem 12.6% pior compressao; tabelas grandes |

**Uso em compressao neural:**
- CompressAI usa rANS para codificacao de entropia
- NWC usa codificacao aritmetica acelerada por GPU (1.17ms decode)
- NNCodec usa DeepCABAC (codificacao aritmetica context-adaptive)

**Implicacoes para velocidade de decompressao:**
- Codificacao de entropia pode ser o **gargalo dominante** (73% do runtime em alguns sistemas)
- Solucoes: tANS com lookup tables, decodificacao paralela por GPU, segmentacao do bitstream
- Para pesos de LLM, a decompressao e feita **uma unica vez** no carregamento -- portanto latencia e aceitavel

**Referencia:** https://arxiv.org/abs/1311.2540

### 4.2 Limites Teoricos de Compressao de Pesos

**Fenomeno de concentracao de expoentes (outubro 2025):**
- Pesos de redes neurais seguem distribuicoes **alpha-estaveis** (pela convergencia de SGD via Teorema Central Limite Generalizado)
- Expoentes de ponto flutuante exibem **entropia baixa** e se concentram em faixas estreitas
- Limite teorico de compressao: ~**FP4.67** (2.67 bits expoente + 1 bit sinal + ~1 bit mantissa)
- Na pratica, GPUs nao suportam FP4.67, entao FP8 com compressao de entropia (ECF8) e o compromisso

**ECF8 (formato pratico):**
- Huffman encoding dos expoentes + lookup tables hierarquicas + decodificacao otimizada para GPU
- Reducao de memoria: 9.8-26.9%
- Aceleracao de throughput: 11.3-177.1%
- **Compressao lossless** (fidelidade bit-exata)

**Referencia:** https://arxiv.org/abs/2510.02676

---

## 5. Quantizacao Vetorial (VQ) para Pesos de LLMs

### 5.1 AQLM -- Additive Quantization for LLMs (janeiro 2024)

**Como funciona:**
- Cada linha da matriz de pesos e dividida em sub-vetores (grupos)
- Cada grupo e aproximado como a **soma de M codewords** de codebooks diferentes
- Cada codebook tem 2^B codewords em precisao nativa
- Custo por grupo: M x B bits

**Otimizacao:**
- **Fase 1:** Beam search para selecao de codigos (reformulado como problema de Markov Random Field)
- **Fase 2:** Atualizacao de codebooks para minimizar erro
- Minimiza erro na **saida da camada** (instance-aware), nao nos pesos diretamente
- Fine-tuning global apos quantizacao

**Resultados (2 bits, WikiText2):**
- Llama 2 7B: 6.93 perplexidade
- Llama 2 13B: 5.70 perplexidade
- Llama 2 70B: 3.94 perplexidade

**Referencia:** https://arxiv.org/abs/2401.06118

### 5.2 QuIP# -- Hadamard Incoherence + Lattice Codebooks (fevereiro 2024)

**Tres inovacoes:**

1. **Hadamard Incoherence Processing:**
   - Transformada de Hadamard aleatorizada torna pesos ~sub-Gaussianos
   - Forma principiada de supressao de outliers

2. **Codebook E8 (lattice 8D):**
   - E8 tem o empacotamento otimo de esferas em 8 dimensoes
   - E8P e **1000x menor** que um codebook naif 8D (1KiB vs. 1MiB) gracas a simetria
   - Decodificacao em **<4 instrucoes por peso**

3. **Residual Vector Quantization (RVQ):**
   - Quantiza o residual iterativamente, diminuindo erro exponencialmente
   - Permite escalar para 3 e 4 bits usando E8 sucessivamente

**Resultado marcante:** Com 2 bits, o Llama 2 70B cabe em uma **unica GPU consumer de 24GB**.

**Referencia:** https://arxiv.org/abs/2402.04396

### 5.3 QTIP -- Quantization with Trellises (NeurIPS 2024 Spotlight)

**Problema resolvido:** VQ tem custo exponencial na dimensao. QTIP usa **Trellis Coded Quantization (TCQ)** para custo linear.

**Inovacoes:**
- **Bitshift trellis:** Estrutura implicitamente definida (zero armazenamento), decodificacao paralela em GPU
- **Codebooks computados:** Gera valores Gaussianos com 3 instrucoes ALU + pequeno lookup (sem armazenar codebook)
- Incoherence processing -> pesos ~ i.i.d. Gaussiano -> **reduz quantizacao a source coding Gaussiano**

**Resultados:**
- Supera QuIP# e AQLM **sem** fine-tuning
- Dimensao 256 (vs. 8 do QuIP#): MSE 0.071 vs. 0.089
- Mesma velocidade de inferencia que QuIP# (ganhos de qualidade "essencialmente gratis")
- 3x throughput vs. modelos nao-quantizados em H100

**Referencia:** https://arxiv.org/abs/2406.11235

### 5.4 VPTQ -- Vector Post-Training Quantization (Microsoft, EMNLP 2024)

**Abordagem:** Formula o problema de VQ para LLMs usando **otimizacao de segunda ordem** (Hessiana).

- Decomposicao do problema de otimizacao -> algoritmo de inicializacao de codebook
- Suporta quantizacao residual + outlier quantization
- Comprime Llama 3 405B para **1-2 bits** sem retraining

**Resultados vs. estado da arte:**
- Reducao de perplexidade: 0.01-7.34 em 2 bits
- Melhoria de accuracy: 0.79-22% em tarefas QA
- Usa apenas 10.4-18.6% do tempo de algoritmos concorrentes
- 1.6-1.8x throughput de inferencia

**Referencia:** https://arxiv.org/abs/2409.17066

### 5.5 LLVQ -- Leech Lattice Vector Quantization (marco 2025)

**Estado da arte atual em 2 bits.**

- Usa o **Leech lattice** (24 dimensoes) -- empacotamento otimo em 24D
- Nao armazena codebook: usa propriedades algebricas para calcular nearest neighbors on-the-fly
- Codigo de Golay para indexacao eficiente
- Busca angular sobre unioes de shells

**Supera AQLM, QuIP#, QTIP** em perplexidade e tarefas downstream.

**Referencia:** https://arxiv.org/abs/2603.11021

### 5.6 GLVQ -- Grouped Lattice VQ (NeurIPS 2025)

Atribui a cada grupo de pesos um **codebook de lattice customizado** com matriz de geracao aprendivel. Combina a eficiencia de lattices com a adaptabilidade de codebooks aprendidos.

**Referencia:** https://arxiv.org/abs/2510.20984

---

## 6. O Conceito de Hyperprior para Pesos de LLMs

### 6.1 Como funciona em compressao de imagens

Em compressao neural de imagens, o hyperprior funciona assim:
1. A representacao latente y tem estatisticas que variam espacialmente
2. Um segundo autoencoder (hyperencoder) comprime essas estatisticas em z
3. z e enviado como **side information** (informacao lateral) -- tipicamente poucos % do bitstream total
4. O decoder usa z para condicionar o modelo de entropia, tornando-o **adaptativo por regiao**

### 6.2 Transferencia para pesos de LLMs

**Analogia direta:**
| Imagens | Pesos de LLM |
|---------|-------------|
| Pixels de uma regiao | Pesos de um bloco/camada |
| Estatisticas espaciais variaveis | Estatisticas por camada/cabeca/canal variaveis |
| Hyperprior z codifica escalas/medias locais | Pequena rede codifica distribuicoes por camada |
| Side information (~5-10% do bitstream) | Metadados de calibracao (~0.1-1% do modelo) |

**Implementacoes existentes:**

1. **NWC (2025):** Usa um encoder-decoder neural (MLPs 4 camadas) que funciona como um hyperprior aprendido para pesos. O encoder captura estatisticas dos chunks de pesos e o modelo de entropia se adapta.

2. **AQLM/QuIP#:** Usam informacao por camada (Hessiana, calibracao) como uma forma implicita de "side information" para guiar a quantizacao.

3. **NNCodec/DeepCABAC:** O codificador aritmetico context-adaptive adapta suas probabilidades on-the-fly, funcionando como um hyperprior implicito.

### 6.3 Oportunidade para o projeto DCT

**Proposta concreta:** Implementar um hyperprior leve para pesos de LLM:
- Uma pequena rede (ou ate parametros explicitios) que armazena, por camada ou por bloco:
  - Media e escala da distribuicao de pesos
  - Parametros da distribuicao (ex: alpha da alpha-estavel)
  - Informacao de importancia (diagonais da Hessiana, comprimidas)
- Este hyperprior condiciona o modelo de entropia para compressao mais eficiente
- Custo: ~0.01-0.1 bits/parametro adicional (desprezivel)
- Ganho esperado: 5-20% melhor compressao no mesmo nivel de accuracy

---

## 7. Conexao com DCT para Compressao de Pesos

### 7.1 DCT na compressao neural classica

No JPEG, a DCT transforma blocos 8x8 de pixels para o dominio de frequencia, onde:
- Coeficientes de baixa frequencia concentram a maior parte da energia
- Coeficientes de alta frequencia podem ser quantizados mais agressivamente
- Tabelas de quantizacao sao fixas (nao aprendidas)

### 7.2 DCT aplicada a pesos de redes neurais

**Trabalho existente (CNNpack, NeurIPS 2016):**
- Kernels convolucionais transformados para dominio de frequencia via DCT
- Seguido de k-means clustering, L1 shrinkage, quantizacao e Huffman coding
- Reducao de memoria e complexidade computacional

### 7.3 Vantagens teoricas da DCT para pesos

1. **Compactacao de energia:** Se pesos de uma camada tem estrutura espacial/correlacional, a DCT concentra energia em poucos coeficientes
2. **Quantizacao adaptativa por frequencia:** Coeficientes de alta frequencia (detalhes finos) podem receber menos bits
3. **Nao requer treino:** Ao contrario de transformadas aprendidas, a DCT e fixa e rapida
4. **Base ortonormal:** Preserva normas e distancias (reconstrucao otima em MSE)

### 7.4 Desvantagens e como mitigar

1. **Pesos nao sao imagens:** A estrutura espacial pode ser menor
   - **Mitigacao:** Testar em matrizes de peso reshape para 2D; muitas matrizes de atencao tem estrutura
2. **DCT e linear:** Nao captura dependencias nao-lineares
   - **Mitigacao:** Combinar com modelo de entropia aprendido (como no NWC)
3. **Hadamard pode ser superior:** Em pratica, a Hadamard e usada mais que DCT em LLMs
   - **Contra-argumento:** NWC mostra que transformadas aprendidas superam Hadamard; DCT pode estar no meio-termo (melhor que Hadamard, mais rapida que aprendida)

---

## 8. Insights Acionaveis para o Projeto DCT

### 8.1 Arquitetura recomendada (inspirada em NWC + JPEG)

```
Pesos W [n x m]
    |
    v
Reshape para blocos 2D (ex: 16x16 ou 32x32)
    |
    v
DCT-2D por bloco (transformada fixa, rapida)
    |
    v
Quantizacao adaptativa por frequencia
  (tabela de quantizacao aprendida por camada -- como "hyperprior")
    |
    v
Modelo de entropia (Gaussiana/Laplaciana condicionada em estatisticas por camada)
    |
    v
Codificacao ANS (rANS para flexibilidade, tANS se velocidade e critica)
    |
    v
Bitstream comprimido + side information (tabelas de quantizacao + estatisticas)
```

### 8.2 Prioridades de implementacao

1. **ALTA:** Implementar DCT-2D em blocos de pesos e medir compactacao de energia
2. **ALTA:** Comparar DCT vs. Hadamard vs. nenhuma transformada em termos de kurtosis e entropia dos coeficientes
3. **MEDIA:** Implementar modelo de entropia simples (Gaussiana/Laplaciana por canal) + rANS
4. **MEDIA:** Adicionar "hyperprior" leve (parametros por camada que condicionam o modelo de entropia)
5. **BAIXA:** Combinar com RVQ (quantizacao residual) para taxas ultra-baixas (2 bits)
6. **BAIXA:** Explorar codebooks de lattice (E8 ou Leech) no dominio DCT

### 8.3 Metricas de avaliacao

- **Bits por parametro (bpp):** taxa de compressao real apos entropy coding
- **Perplexidade WikiText2:** degradacao de qualidade
- **Accuracy em benchmarks:** MMLU, ARC, HellaSwag
- **Tempo de decompressao:** essencial para carregamento do modelo
- **Distancia ao limite de Shannon:** entropia empirica dos pesos quantizados vs. taxa real

### 8.4 Vantagem competitiva potencial

O insight central e: **nenhum metodo atual combina DCT + modelo de entropia aprendido + codificacao ANS para pesos de LLMs.** Os metodos existentes usam:
- Hadamard (QuIP#, QTIP) -- fixo, rapido, mas subotimo
- Transformadas aprendidas (NWC) -- otimas, mas requerem treino do codec
- Nenhuma transformada (AQLM, GPTQ) -- mais simples, mas perde compactacao

A DCT ocupa um **nicho unico**: e fixa (sem treino), mais rica que Hadamard (captura estrutura de frequencia), e universalmente eficiente. Combinada com um modelo de entropia condicionado em estatisticas por camada (hyperprior leve), pode atingir compressao competitiva sem o custo de treinar um codec.

---

## 9. Tabela Comparativa de Metodos

| Metodo | Ano | Bits | Transformada | Entropy Coding | VQ | Hyperprior |
|--------|-----|------|-------------|----------------|-----|------------|
| GPTQ | 2023 | 3-4 | Nenhuma | Nao | Nao | Nao |
| AWQ | 2023 | 4 | Nenhuma | Nao | Nao | Nao |
| QuIP# | 2024 | 2-4 | Hadamard | Nao | E8 lattice | Nao (implicito) |
| AQLM | 2024 | 2-4 | Nenhuma | Nao | Additive VQ | Nao (implicito) |
| QTIP | 2024 | 2-4 | Hadamard | Implicito (trellis) | TCQ | Nao |
| VPTQ | 2024 | 1-4 | Nenhuma | Nao | VQ + 2a ordem | Nao |
| NNCodec | 2022 | Variavel | Nenhuma | DeepCABAC | Nao | Implicito (context-adaptive) |
| NWC | 2025 | 4-6 | **Aprendida** | **Aritmetica** | Nao | **Sim (aprendido)** |
| LLVQ | 2025 | 2 | Hadamard | Nao | Leech lattice | Nao |
| EntroLLM | 2025 | 1.4-5.6 | Nenhuma | Huffman | Nao | Nao |
| ECF8 | 2025 | ~5 | Nenhuma | Huffman | Nao | Nao (lossless) |
| **DCT (proposta)** | 2026 | 2-6 | **DCT** | **rANS/tANS** | Opcional | **Sim (leve)** |

---

## 10. Referencias Principais

### Compressao Neural de Imagens
1. Balle et al. "End-to-end Optimized Image Compression." ICLR 2017. https://arxiv.org/abs/1611.01704
2. Balle et al. "Variational Image Compression with a Scale Hyperprior." ICLR 2018. https://arxiv.org/abs/1802.01436
3. Minnen et al. "Joint Autoregressive and Hierarchical Priors for Learned Image Compression." NeurIPS 2018. https://arxiv.org/abs/1809.02736

### Compressao Neural de Video
4. Agustsson et al. "Scale-Space Flow for End-to-End Optimized Video Compression." CVPR 2020.
5. Li et al. "Deep Contextual Video Compression." NeurIPS 2021. https://github.com/microsoft/DCVC

### Compressao de Pesos com Tecnicas Neurais
6. NWC: "Neural Weight Compression for Language Models." 2025. https://arxiv.org/abs/2510.11234
7. NNCodec (ISO/IEC 15938-17). https://github.com/fraunhoferhhi/nncodec
8. "Reducing Storage of Pretrained Neural Networks by Rate-Constrained Quantization and Entropy Coding." 2025. https://arxiv.org/abs/2505.18758
9. EntroLLM. 2025. https://arxiv.org/abs/2505.02380
10. ECF8: "To Compress or Not? Pushing the Frontier of Lossless GenAI Model Weights Compression." 2025. https://arxiv.org/abs/2510.02676

### Quantizacao Vetorial para LLMs
11. AQLM: "Extreme Compression of LLMs via Additive Quantization." ICML 2024. https://arxiv.org/abs/2401.06118
12. QuIP#: "Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks." ICML 2024. https://arxiv.org/abs/2402.04396
13. QTIP: "Quantization with Trellises and Incoherence Processing." NeurIPS 2024. https://arxiv.org/abs/2406.11235
14. VPTQ: "Extreme Low-bit Vector Post-Training Quantization for LLMs." EMNLP 2024. https://arxiv.org/abs/2409.17066
15. LLVQ: "Leech Lattice Vector Quantization for Efficient LLM Compression." 2025. https://arxiv.org/abs/2603.11021
16. GLVQ: "Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression." NeurIPS 2025. https://arxiv.org/abs/2510.20984

### Codificacao de Entropia
17. Duda. "Asymmetric Numeral Systems." 2014. https://arxiv.org/abs/1311.2540

### Ferramentas
18. CompressAI (PyTorch). https://github.com/InterDigitalInc/CompressAI

---

*Este documento foi gerado como pesquisa para o projeto DCT de quantizacao de pesos de LLMs.*
