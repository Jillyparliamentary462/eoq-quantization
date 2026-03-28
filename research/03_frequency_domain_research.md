# Pesquisa: Dominio de Frequencia para Compressao de Redes Neurais

**Data da pesquisa:** 2026-03-28
**Escopo:** DCT, wavelets, Fourier, JPEG-style compression, matrizes estruturadas, teoria de matrizes aleatorias, correlacao espacial em matrizes de pesos de redes neurais.

---

## Indice

1. [DCT Aplicada a Matrizes de Pesos de Redes Neurais](#1-dct-aplicada-a-matrizes-de-pesos-de-redes-neurais)
2. [Compressao Baseada em Wavelets](#2-compressao-baseada-em-wavelets)
3. [Analise de Fourier de Matrizes de Pesos](#3-analise-de-fourier-de-matrizes-de-pesos)
4. [Compressao Estilo JPEG para Redes Neurais](#4-compressao-estilo-jpeg-para-redes-neurais)
5. [Matrizes Estruturadas e Conexao com Frequencia](#5-matrizes-estruturadas-e-conexao-com-frequencia)
6. [Teoria de Matrizes Aleatorias Aplicada a Pesos](#6-teoria-de-matrizes-aleatorias-aplicada-a-pesos)
7. [Correlacao Espacial em Matrizes de Pesos](#7-correlacao-espacial-em-matrizes-de-pesos)
8. [Sintese e Implicacoes para o Projeto](#8-sintese-e-implicacoes-para-o-projeto)

---

## 1. DCT Aplicada a Matrizes de Pesos de Redes Neurais

### 1.1 Redes Harmonicas (Harmonic Networks)

O trabalho mais consolidado na aplicacao da DCT a redes neurais sao as **Harmonic Convolutional Networks** (Ulicny et al., 2020/2022). Nessa abordagem:

- Filtros convolucionais sao aprendidos como combinacoes de filtros espectrais pre-definidos pela DCT.
- Os "blocos harmonicos" substituem camadas convolucionais tradicionais.
- A DCT permite **ordenar os parametros por significancia**: frequencias baixas (mais relevantes) para frequencias altas (menos importantes).
- **Compressao por truncamento**: removendo coeficientes de alta frequencia, o modelo pode ser comprimido com degradacao minima de desempenho.
- Resultados: WRN com lambda=3 tem ~33% menos parametros que o baseline; com lambda=2, o modelo fica ~3x menor.
- **Observacao critica**: camadas mais profundas tendem a favorecer informacao de baixa frequencia sobre alta frequencia ao aprender representacoes.

**Fonte:** [Harmonic Convolutional Networks based on Discrete Cosine Transform](https://arxiv.org/abs/2001.06570)

### 1.2 DCT-Conv: Codificacao de Filtros no Dominio DCT

O DCT-Conv (2020) propoe que filtros em camadas convolucionais resultem da IDCT (inversa) sobre pesos treinados:

- Os tensores de pesos treinados sao transformados em filtros via IDCT.
- Alguns pesos sao fixados em zero, reduzindo o numero de parametros treinaveis.
- O operador DCT e diferenciavel e implementavel por multiplicacao elemento-a-elemento, facilitando a integracao em CNNs.

**Fonte:** [DCT-Conv: Coding filters in convolutional networks](https://arxiv.org/pdf/2001.08517)

### 1.3 Compressao Espectral via DFT (Post-Hoc)

Um trabalho recente (2025) propoe compressao pos-treinamento usando a Transformada Discreta de Fourier:

- Apos o treinamento, cada vetor de pesos e transformado via DFT.
- Filtragem por estrategia escolhida: passa-baixa, limiar de magnitude, ou mascara espectral.
- Reconstrucao via DFT inversa.
- **Resultados**: compressao de 10-15x com perda negligenciavel em MNIST, CIFAR-10 e ResNet.
- Confirmacao em hardware de reducao de uso de memoria e melhoria de latencia de inferencia.
- Desempenho **competitivo comparado com DCT e wavelets**.
- **Vantagem-chave**: nao requer retreinamento; e leve e adequado para IA embarcada.

**Fonte:** [Spectral Neural Network Compression via DFT](https://www.researchsquare.com/article/rs-7254889/v1)

### 1.4 Compressao por Transform Coding (Laude et al., 2018)

Um dos primeiros trabalhos a aplicar diretamente a DCT como codec de compressao para pesos de redes neurais:

- Aplica transform coding para camadas convolucionais e densas.
- O tamanho do bloco DCT e definido conforme o tamanho do filtro (ex: DCT 7x7 para filtros 7x7).
- Clustering e usado para biases e normalizacoes.
- **Resultados**: fatores de compressao medios entre **7.9-9.3x** com reducao de acuracia de apenas 1-2%.
- Em hardware MNIST: compressao de ate 42x com perda menor que 1% de acuracia, largura de banda efetiva 3x maior e energia 19x menor.

**Fonte:** [Neural Network Compression using Transform Coding and Clustering](https://arxiv.org/abs/1805.07258)

### 1.5 DCT para Vision Transformers

Metodos de compressao baseados em DCT foram formulados para Vision Transformers:

- Truncamento de componentes de alta frequencia reduz efetivamente a carga computacional.
- Resulta em matrizes de pesos menores que mantem acuracia consideravel.

**Fonte:** [DCT Based Decorrelated Attention for Vision Transformers](https://arxiv.org/html/2405.13901v3)

---

## 2. Compressao Baseada em Wavelets

### 2.1 Wavelets Aprendiveis para Compressao de Pesos

Wolter, Lin e Yao (2020) propuseram a aplicacao da **Transformada Rapida de Wavelet** para compressao de camadas lineares:

- Tanto as bases wavelet quanto os coeficientes correspondentes sao **aprendidos** para representar eficientemente camadas lineares de RNNs.
- A otimizacao de wavelets adiciona flexibilidade de base sem um grande numero de pesos extras.
- **Resultados**: RNNs comprimidas por wavelets tem significativamente menos parametros e ainda assim competem com o estado-da-arte em benchmarks sinteticos e do mundo real.
- Implementacao publica em PyTorch disponivel.

**Fonte:** [Neural Network Compression via Learnable Wavelet Transforms](https://arxiv.org/abs/2004.09569) | [Codigo](https://github.com/v0lta/Wavelet-network-compression)

### 2.2 Wavelet Compressed Convolution (WCC)

Para compressao de mapas de ativacao (nao diretamente pesos):

- A transformada Haar-wavelet e usada para comprimir mapas de ativacao em convolucoes 1x1.
- Abordagem amigavel ao hardware (hardware-friendly).

**Fonte:** [Wavelet Feature Maps Compression for Image-to-Image CNNs (NeurIPS 2022)](https://papers.neurips.cc/paper_files/paper/2022/file/81f19c0e9f3e06c831630ab6662fd8ea-Paper-Conference.pdf)

### 2.3 Wavelets vs DCT vs DFT

O trabalho de compressao espectral via DFT (secao 1.3) inclui comparacao direta, indicando que DFT apresenta desempenho competitivo com DCT e wavelets. Isso sugere que as tres transformadas sao viáveis, mas com trade-offs diferentes:

| Transformada | Vantagens | Desvantagens |
|---|---|---|
| **DCT** | Excelente compactacao de energia; sem numeros complexos; bloco definido | Requer escolha de tamanho de bloco |
| **DFT** | Teoria de limiar rigorosa; pos-treinamento | Numeros complexos (2x armazenamento) |
| **Wavelet** | Multi-resolucao; bases aprendiveis | Mais parametros; complexidade de implementacao |

---

## 3. Analise de Fourier de Matrizes de Pesos

### 3.1 O Principio de Frequencia (F-Principle) / Vies Espectral

Uma descoberta fundamental sobre como redes neurais aprendem:

- **DNNs ajustam funcoes-alvo de baixa para alta frequencia durante o treinamento.** Isso e chamado de "Frequency Principle" ou "Spectral Bias".
- Redes com ativacao ReLU frequentemente falham em representar componentes de alta frequencia de sinais (vies de baixa frequencia).
- **Implicacoes para compressao**: como o ruido e dominado por altas frequencias, com early-stopping, uma rede com vies espectral pode evitar aprender ruido de alta frequencia.
- Aumentar a profundidade da rede melhora significativamente a capacidade de ajustar frequencias mais altas.

**Fontes:**
- [On the Spectral Bias of Neural Networks](https://arxiv.org/abs/1806.08734)
- [Frequency Principle: Fourier Analysis Sheds Light on DNNs](https://arxiv.org/abs/1901.06523)
- [Wikipedia: Frequency principle/spectral bias](https://en.wikipedia.org/wiki/Frequency_principle/spectral_bias)

### 3.2 A Funcao que a Rede Computa vs Seus Pesos

Ha uma distincao crucial:

- O **F-Principle** descreve a funcao que a rede computa (saida em funcao da entrada), nao diretamente a estrutura dos pesos.
- A funcao computada tende a ser suave (dominada por baixas frequencias).
- Isso **nao implica necessariamente** que as matrizes de pesos em si tenham estrutura de baixa frequencia.
- Porem, existe uma conexao indireta: se a rede computa uma funcao suave, os pesos devem estar organizados de forma coerente, o que pode manifestar alguma estrutura espectral.

### 3.3 Fourier Features para Superar o Vies Espectral

Trabalhos como Tancik et al. (NeurIPS 2020) mostram que:

- Passar entradas por um mapeamento de Fourier permite que MLPs aprendam funcoes de alta frequencia.
- Isso transforma o kernel tangente neural efetivo em um kernel estacionario com largura de banda ajustavel.
- Operadores Neurais de Fourier (FNO) demonstram capacidade excepcional em capturar informacao de baixa frequencia.

**Fonte:** [Fourier Features Let Networks Learn High Frequency Functions](https://dl.acm.org/doi/abs/10.5555/3495724.3496356)

---

## 4. Compressao Estilo JPEG para Redes Neurais

### 4.1 Trabalhos Que Tentaram Esta Abordagem

**Sim, a abordagem JPEG-style (DCT em blocos + quantizacao + codificacao de entropia) ja foi explorada:**

**Laude et al. (2018)** -- O trabalho mais proximo de JPEG para pesos neurais:
- Transform coding baseado em DCT aplicado a camadas convolucionais e densas.
- Quantizacao dos coeficientes DCT.
- Resultados de compressao 7.9-9.3x.

**Compressao Adaptativa de Pesos (2017, Design Automation Conference):**
- Aplica codificacao JPEG diretamente a pesos de redes neurais.
- Explora localidade espacial e suavidade da matriz de pesos.
- Controle adaptativo do fator de quantizacao baseado na sensibilidade ao erro (gradiente) de cada peso.
- Blocos com maior sensibilidade sao comprimidos menos para maior acuracia.
- **Resultados**: ate 42x compressao com menos de 1% perda de acuracia (MNIST), largura de banda efetiva 3x maior, energia 19x menor.

**Fonte:** [Adaptive weight compression for memory-efficient neural networks](https://ieeexplore.ieee.org/document/7926982/)

### 4.2 Por Que JPEG-Style Nao Se Tornou Mainstream

Apesar dos resultados promissores, existem razoes pela qual esta abordagem nao dominou:

1. **Estrutura de dados diferente**: Pesos de redes neurais nao tem os mesmos padroes espaciais ou estrutura que dados de imagem. Imagens tem forte correlacao espacial local; pesos nem sempre tem.

2. **Falta de flexibilidade**: A DCT e uma transformada fixa -- nao e necessariamente a transformada otima para todos os tipos de pesos. A Transformada de Karhunen-Loeve (KLT) seria a otima, mas requer transmissao do kernel de transformada para cada bloco, exigindo grande quantidade de informacao lateral.

3. **Alternativas mais eficazes surgiram**: Quantizacao (AWQ, GPTQ), pruning, destilacao de conhecimento, e abordagens de baixo rank (LoRA/SVD) mostraram-se mais praticas para o pipeline existente de ML.

4. **Custo de decodificacao**: A decompressao em tempo de inferencia adiciona overhead computacional. Metodos como quantizacao uniforme sao triviais de decodificar.

5. **Distribuicao dos pesos**: Pesos tendem a ter distribuicoes de cauda pesada (Laplaciana ou power-law), nao Gaussiana. Codecs otimizados para Gaussiana (como JPEG) mostram desvantagens comparados com codecs projetados para dados de cauda pesada.

### 4.3 Neural Weight Compression (NWC) -- A Evolucao Moderna

O trabalho NWC (2025) representa a evolucao moderna da ideia JPEG-style:

- Formula compressao de pesos como um problema de **codec neural aprendido** (nao fixa a transformada).
- Pesos sao particionados em chunks de 16 elementos, normalizados.
- Treinamento com awareness de importancia via diagonais do Hessiano.
- Quantizacao com restricao de entropia, aproximando-se do **limite de Shannon**.
- **Resultado**: a transformada aprendida reduz a curtose dos pesos de 20.48 para perto de 0, suprimindo outliers.
- Desempenho superior a 4-6 bits/parametro em Llama, Mixtral e encoders de visao.

**Fonte:** [Neural Weight Compression for Language Models](https://arxiv.org/html/2510.11234)

### 4.4 DFloat11 -- Compressao Lossless por Codificacao de Entropia

Uma abordagem puramente de codificacao de entropia (sem transformada):

- Explora a **baixa entropia** na representacao BFloat16 dos pesos de LLMs.
- A distribuicao de expoentes e altamente desbalanceada: apenas ~40 dos 256 valores possiveis de 8 bits sao usados.
- Aplica codificacao de Huffman nos bits de expoente, combinada com design algoritmico hardware-aware.
- **Resultado**: 30% de reducao de tamanho com saida **bit-a-bit identica** ao modelo original.
- Aceito no NeurIPS 2025.

**Fonte:** [DFloat11: Lossless LLM Compression](https://arxiv.org/abs/2504.11651)

---

## 5. Matrizes Estruturadas e Conexao com Frequencia

### 5.1 Framework de Low Displacement Rank (LDR)

Matrizes estruturadas como Toeplitz, Hankel, Vandermonde, Cauchy e circulantes pertencem a classe de matrizes com **baixo rank de deslocamento (LDR)**:

- Uma matriz e representada por dois operadores de deslocamento e um residuo de baixo rank.
- Essas matrizes tem representacoes eficientes no dominio da frequencia.
- Multiplicacao por essas matrizes custa O(n log n) ao inves de O(n^2).

**Fonte:** [Theoretical Properties for Neural Networks with Weight Matrices of Low Displacement Rank](https://arxiv.org/abs/1703.00144)

### 5.2 CirCNN: Matrizes Bloco-Circulantes

O CirCNN (2017) e a abordagem mais desenvolvida usando matrizes circulantes:

- Representa pesos usando matrizes bloco-circulantes.
- Usa **FFT** para multiplicacao rapida: complexidade reduzida de O(n^2) para O(n log n).
- Armazenamento reduzido de O(n^2) para O(n) -- cada sub-matriz circulante e representada por um vetor primitivo de comprimento k.
- Evita os problemas de pruning (estrutura irregular, complexidade de treinamento, falta de garantia de taxa de compressao).
- Combina com quantizacao power-of-two para substituir multiplicacoes por operacoes de shift e add.

**Fonte:** [CirCNN: Accelerating and Compressing Deep Neural Networks Using Block-Circulant Weight Matrices](https://arxiv.org/abs/1708.08917)

### 5.3 Matrizes Bloco-Circulantes com DCT-DST em Transformers

Um trabalho diretamente relevante (2023) para o nosso projeto:

- Propoe usar matrizes g-circulantes em bloco para substituir matrizes densas nas **camadas feedforward** de Transformers.
- Usa o algoritmo **DCT-DST** (nao FFT!) para multiplicacao eficiente.
- A multiplicacao matriz-vetor via DCT-DST e definida pelo produto de Kronecker entre a matriz DCT-DST e uma matriz ortogonal.
- **Resultados**: reducao de ate **41% dos parametros** do modelo com leve degradacao de acuracia.
- Testado em traducao Portugues-Ingles.

**Fonte:** [Real block-circulant matrices and DCT-DST algorithm for transformer neural network](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2023.1260187/full)

### 5.4 SURMs: Matrizes Estruturadas para Fine-Tuning Eficiente

Structured Unrestricted-Rank Matrices (SURMs) foram propostas como alternativa ao LoRA:

- Comparam camadas LoRA, circulantes, Toeplitz simetrico e Toeplitz com o mesmo numero de parametros treinaveis.
- SURMs alcancam **ganhos de 7% em acuracia sobre LoRA** em diversas tarefas de classificacao de imagens em cenarios de baixo recurso.
- Matrizes LDR mostram desempenho similar ao baseline enquanto camadas LoRA lutam para ajustar os dados.

**Fonte:** [Structured Unrestricted-Rank Matrices for Parameter Efficient Fine-tuning](https://arxiv.org/html/2406.17740v1)

### 5.5 Sao Matrizes de Pesos de Transformers Proximas de Estruturas Especiais?

A evidencia sugere:

- **Matrizes de pesos de Transformers nao sao exatamente circulantes ou Toeplitz**, mas podem ser bem aproximadas por essas estruturas, especialmente nas camadas feedforward.
- A eficacia do LoRA demonstra que as **atualizacoes de pesos** tem baixo rank intrinseco.
- Matrizes pre-treinadas de LLMs tem "dimensao intrinseca" baixa -- podem aprender eficientemente apesar de projecao aleatoria para subespacos menores.
- Isso motiva tanto abordagens de baixo rank (SVD/LoRA) quanto abordagens de matrizes estruturadas (circulante/Toeplitz).

---

## 6. Teoria de Matrizes Aleatorias Aplicada a Pesos

### 6.1 Marchenko-Pastur e a Estrutura Bulk+Spike

A **lei de Marchenko-Pastur (MP)** descreve o comportamento assintotico do espectro de matrizes de Wishart (XX^T para X aleatorio):

**Antes do treinamento**: os valores singulares conformam-se precisamente as predicoes de matrizes aleatorias (distribuicao MP).

**Apos o treinamento**: a maior parte do espectro **permanece aleatoria** e continua a corresponder a predicoes MP. Porem, **desvios significativos ocorrem apenas nas partes associadas aos maiores valores singulares** (os "spikes").

Isso cria a famosa estrutura **"bulk+tail"**:
- **Bulk**: a maioria dos valores singulares segue a distribuicao MP -- representam "ruido" ou informacao nao aprendida.
- **Spikes/Tail**: os maiores valores singulares desviam-se do bulk -- aqui esta a **informacao aprendida pela rede**.

**Implicacao direta para compressao**: A informacao aprendida esta concentrada em poucos valores singulares dominantes. O "bulk" pode ser potencialmente descartado ou comprimido agressivamente.

**Fonte:** [Random matrix theory analysis of neural network weight matrices](https://openreview.net/pdf?id=41kpc2Nzwc)

### 6.2 WeightWatcher e a Teoria de Auto-Regularizacao de Cauda Pesada (HTSR)

Martin e Mahoney desenvolveram a **teoria HTSR (Heavy-Tailed Self-Regularization)** e a ferramenta WeightWatcher:

- Quando redes neurais treinam eficazmente, os valores singulares de suas matrizes de pesos seguem **distribuicoes de cauda pesada tipo power-law**.
- O parametro **alpha** (inclinacao da cauda da ESD em escala log-log) e um indicador de qualidade:
  - alpha proximo de 2.0: camada bem treinada.
  - alpha muito alto: camada sub-treinada ou com problemas.
- A **compressibilidade da distribuicao de eigenvalues** indica quao bem a rede foi treinada.
- Metricas baseadas em power-law superam metricas baseadas em norma porque capturam correlacoes de informacao em multiplas escalas.
- **Metricas disponiveis**: rand_distance, alpha, alpha_weighted, stable_rank, num_spikes (numero de spikes fora da regiao MP bulk).

**Fontes:**
- [WeightWatcher: Data-Free Diagnostics for Deep Learning](https://weightwatcher.ai/)
- [Predicting trends in the quality of state-of-the-art neural networks (Nature Communications)](https://www.nature.com/articles/s41467-021-24025-8)
- [Traditional and Heavy-Tailed Self Regularization in Neural Network Models](https://arxiv.org/abs/1901.08276)

### 6.3 Dinamica Espectral: De SGD a Espectros

Pesquisa recente (2024-2025) sobre dinamica espectral de pesos durante treinamento:

- Valores singulares ao quadrado seguem **movimento Browniano de Dyson** com repulsao de eigenvalues.
- Distribuicoes estacionarias sao **densidades tipo gamma com caudas power-law**.
- Isso fornece a primeira explicacao teorica para a estrutura "bulk+tail" observada empiricamente.
- **Aplicacao**: inicializacao spectral-aware, taxas de aprendizado adaptativas, e **pruning informado pelo espectro** que preserva contribuidores da estrutura bulk+tail enquanto comprime agressivamente parametros do bulk.

**Fonte:** [From SGD to Spectra: A Theory of Neural Network Weight Dynamics](https://arxiv.org/html/2507.12709)

### 6.4 Grokking e Minimizacao de Rank

O fenomeno de "grokking" (generalizacao tardia) esta conectado a estrutura espectral:

- A transicao para generalizacao coincide com a **descoberta de solucoes de baixo rank** em todas as matrizes de pesos.
- Weight decay afeta tanto grokking quanto minimizacao de rank.
- Ha uma correlacao consistente entre matrizes de baixo rank e generalizacao.
- Este vies de baixo rank e observado desde tarefas pequenas (grokking) ate grandes (classificacao de imagens com ConvNets, geracao com UNets, reconhecimento de fala com LSTMs, modelagem de linguagem com Transformers).

**Fonte:** [Grokking, Rank Minimization and Generalization in Deep Learning (ICLR 2024)](https://openreview.net/forum?id=6NHnsjsYXH)

### 6.5 O Que Isso Significa para Compressao no Dominio de Frequencia

A teoria de matrizes aleatorias revela que:

1. **Matrizes de pesos sao altamente compressiveis** -- a maior parte do espectro e "ruido" que segue MP.
2. A informacao util esta em poucos valores singulares dominantes (spikes).
3. Compressao por SVD/low-rank e naturalmente motivada por esta estrutura.
4. **Para compressao DCT**: a questao-chave e se a DCT consegue separar eficientemente os spikes do bulk. Se a estrutura for predominantemente de baixo rank (poucos vetores singulares importantes), entao SVD e mais natural. Se houver correlacao espacial nas matrizes de pesos, a DCT pode ser complementar.

---

## 7. Correlacao Espacial em Matrizes de Pesos

### 7.1 A Questao Central

A eficacia da DCT para compressao depende fundamentalmente de: **entradas proximas na matriz de pesos sao correlacionadas?**

- **Se sim**: a DCT vai concentrar energia nos coeficientes de baixa frequencia (como faz com imagens).
- **Se nao**: a DCT nao vai ajudar significativamente -- os coeficientes serao distribuidos uniformemente entre frequencias.

### 7.2 Evidencias de Estrutura Espacial

**A favor da existencia de correlacao espacial:**

1. **Distill "Visualizing Weights" (Olah et al., 2020)**: Pesos expandidos (resultado da multiplicacao de matrizes de pesos adjacentes) frequentemente exibem **estruturas espaciais suaves tipo "orbital de eletron"**. Esta suavidade e "tipica da maioria dos pesos expandidos de multiplas camadas", sugerindo "rica estrutura espacial na escala de multiplas camadas".

2. **Weight Banding (Distill, 2020)**: Em camadas convolucionais finais de modelos de visao (InceptionV1, ResNet50, VGG19), pesos exibem **faixas horizontais** -- forte correlacao posicional. Pesos em posicoes verticais similares tem valores similares entre canais.

3. **Filtros convolucionais**: Por design, filtros convolucionais sao pequenos (3x3, 5x5, 7x7) e representam padroes espaciais locais (bordas, texturas). Esses filtros intrinsecamente tem correlacao espacial.

4. **Correlacao entre pesos (Yak et al., 2020)**: A correlacao de pesos pode ser definida como a **similaridade de cosseno media** entre vetores de pesos de neuronios (para camadas FC) ou entre matrizes de filtros (para camadas convolucionais). Pesos altamente correlacionados indicam redundancia compressivel.

**Fonte:** [Visualizing Weights](https://distill.pub/2020/circuits/visualizing-weights/) | [Weight Banding](https://distill.pub/2020/circuits/weight-banding/)

### 7.3 Evidencias Contra / Complicacoes

**Fatores que complicam a correlacao espacial:**

1. **Matrizes densas (FC/Transformers)**: Ao contrario de filtros convolucionais que tem estrutura espacial intrinseca, as matrizes de pesos de camadas densas e de atencao em Transformers nao tem uma "vizinhanca espacial" natural. A organizacao de linhas e colunas e determinada pela ordem dos neuronios, que e arbitraria.

2. **Espectro de matrizes aleatorias**: A maior parte do espectro de pesos treinados segue a distribuicao MP (efetivamente aleatoria). Elementos aleatorios nao tem correlacao espacial.

3. **Caudas pesadas**: A distribuicao de pesos segue leis de potencia, nao Gaussiana. Outliers extremos podem estar espalhados sem padrao espacial.

4. **Permutation invariance**: Permutar neuronios dentro de uma camada oculta (e ajustar conexoes correspondentes) nao muda a funcao computada pela rede. Isso implica que a "posicao" de entradas na matriz nao tem significado intrinseco para camadas FC.

### 7.4 Implicacao Critica para DCT

**Para camadas convolucionais**: A DCT e uma aposta razoavel. Filtros tem estrutura espacial, e a DCT de filtros funciona bem (evidenciado pelas Harmonic Networks e pelo trabalho de Laude et al.).

**Para matrizes densas/Transformers**: A situacao e menos clara. A DCT 2D aplicada diretamente a uma matriz de pesos densa pode nao concentrar energia eficientemente porque:
- A organizacao de linhas/colunas e arbitraria.
- Nao ha garantia de correlacao entre entradas proximas.
- A estrutura dominante e de baixo rank (melhor capturada por SVD), nao de baixa frequencia.

**Possivel solucao**: Aplicar a DCT apos reordenamento ou dentro de blocos estruturados. Ou usar a DCT ao longo de cada linha/coluna individualmente (1D), nao como DCT 2D da matriz inteira.

---

## 8. Sintese e Implicacoes para o Projeto

### 8.1 Resumo do Estado da Arte

| Abordagem | Compressao Tipica | Requer Retreinamento | Maturidade |
|---|---|---|---|
| DCT + Quantizacao (Laude 2018) | 7.9-9.3x | Nao | Demonstrada |
| JPEG-style adaptativo (2017) | ate 42x (MNIST) | Nao | Demonstrada |
| DFT pos-treinamento (2025) | 10-15x | Nao | Recente |
| Wavelets aprendiveis (2020) | Competitivo c/ SOTA | Sim | Demonstrada |
| Harmonic Networks (2022) | 33%-66% parametros | Sim (treina no dominio DCT) | Madura |
| CirCNN (2017) | O(n) vs O(n^2) | Sim | Madura |
| Block-circulant DCT-DST (2023) | 41% reducao | Sim | Demonstrada |
| NWC codec aprendido (2025) | SOTA a 4-6 bits | Sim (treina o codec) | Recente |
| DFloat11 lossless (2025) | 30% (lossless) | Nao | Madura (NeurIPS 2025) |
| SVD/Low-Rank | Variavel | Variavel | Muito madura |

### 8.2 Insights Chave para o Projeto DCT-Quantization

1. **A abordagem DCT para pesos ja foi validada** -- nao e uma ideia nova, mas e sub-explorada. Os resultados de Laude et al. (7.9-9.3x) e da compressao adaptativa (ate 42x) sao encorajadores.

2. **A compactacao de energia pela DCT funciona melhor para filtros convolucionais** (onde ha correlacao espacial intrinseca) do que para matrizes densas (onde a organizacao e arbitraria).

3. **Complementaridade com SVD**: A estrutura dominante em matrizes de pesos e de baixo rank (poucos valores singulares dominantes). A DCT pode ser complementar ao SVD:
   - SVD captura a estrutura de baixo rank (informacao aprendida nos spikes).
   - DCT pode capturar correlacoes locais residuais apos remocao dos componentes de baixo rank.

4. **Quantizacao adaptativa e essencial**: Blocos com maior sensibilidade (gradiente maior) devem ser comprimidos menos. Isso e analogo as matrizes de quantizacao do JPEG, mas adaptadas por camada ou por bloco.

5. **Distribuicao de pesos nao e Gaussiana**: Pesos seguem distribuicoes de cauda pesada. Codecs otimizados para Gaussiana sao subotimos. Considerar distribuicoes Laplacianas ou codificacao de entropia adaptativa.

6. **DCT 1D por vetor pode ser mais promissora que DCT 2D da matriz**: Aplicar a DCT a cada linha ou coluna da matriz de pesos (tratando como sinais 1D) evita o problema da falta de correlacao espacial 2D em matrizes densas.

7. **O tamanho do bloco importa**: Para filtros convolucionais, o bloco DCT deve corresponder ao tamanho do filtro. Para matrizes densas, experimentar com tamanhos de bloco variados (8, 16, 32, 64).

8. **Matrizes circulantes oferecem um caminho alternativo**: Em vez de comprimir matrizes densas existentes, treinar diretamente com matrizes bloco-circulantes que inherentemente sao diagonalizaveis pela DFT/DCT.

### 8.3 Lacunas na Literatura e Oportunidades

1. **Ninguem combinou sistematicamente DCT + quantizacao + entropia coding + awareness de sensibilidade especificamente para LLMs/Transformers modernos.** A maioria dos trabalhos foca em CNNs ou em redes pequenas.

2. **Falta analise empirica da correlacao espacial em matrizes de atencao e FFN de Transformers modernos.** Visualizar os coeficientes DCT de matrizes de pesos reais (GPT, Llama, etc.) seria contribuicao valiosa.

3. **A combinacao SVD (para capturar low-rank) + DCT (para capturar correlacoes residuais) + entropia coding nao foi explorada.**

4. **Nao ha estudo comparativo rigoroso entre DCT, DFT, wavelets e KLT para compressao de matrizes de pesos de Transformers**, embora o trabalho de compressao espectral via DFT (2025) comece a abordar isso.

5. **A aplicacao de DCT com tamanhos de bloco adaptativos** (analogos ao HEVC que usa blocos de 4x4 a 32x32) a matrizes de pesos nao foi explorada.

### 8.4 Recomendacoes de Proximos Passos

1. **Analise empirica**: Aplicar DCT 1D e 2D a matrizes de pesos de modelos Transformer reais (Llama, GPT-2) e visualizar a distribuicao de energia nos coeficientes. Isso responde diretamente a pergunta: "a DCT compacta energia em pesos de Transformers?"

2. **Benchmark comparativo**: Comparar DCT vs DFT vs wavelets vs SVD para compressao de pesos com mesma taxa de bits, medindo acuracia preservada.

3. **Pipeline JPEG-style completo**: Implementar bloco-DCT + quantizacao adaptativa (por sensibilidade/gradiente) + codificacao de entropia para matrizes de pesos de Transformers.

4. **Investigar reordenamento**: Antes de aplicar DCT, investigar se reordenar linhas/colunas da matriz (por similaridade) melhora a compactacao de energia.

---

## Referencias Completas

### DCT e Redes Harmonicas
- [Harmonic Convolutional Networks based on Discrete Cosine Transform](https://arxiv.org/abs/2001.06570)
- [DCT-Conv: Coding filters in convolutional networks](https://arxiv.org/pdf/2001.08517)
- [Discrete Cosine Transform Based Decorrelated Attention for Vision Transformers](https://arxiv.org/html/2405.13901v3)
- [Neural Network Compression using Transform Coding and Clustering](https://arxiv.org/abs/1805.07258)
- [Adaptive weight compression for memory-efficient neural networks](https://ieeexplore.ieee.org/document/7926982/)

### Compressao Espectral e Fourier
- [Spectral Neural Network Compression via Discrete Fourier Transform](https://www.researchsquare.com/article/rs-7254889/v1)
- [On the Spectral Bias of Neural Networks](https://arxiv.org/abs/1806.08734)
- [Frequency Principle: Fourier Analysis Sheds Light on DNNs](https://arxiv.org/abs/1901.06523)
- [Overview frequency principle/spectral bias in deep learning](https://arxiv.org/abs/2201.07395)
- [Fourier Features Let Networks Learn High Frequency Functions](https://dl.acm.org/doi/abs/10.5555/3495724.3496356)

### Wavelets
- [Neural Network Compression via Learnable Wavelet Transforms](https://arxiv.org/abs/2004.09569)
- [Wavelet Feature Maps Compression for Image-to-Image CNNs](https://papers.neurips.cc/paper_files/paper/2022/file/81f19c0e9f3e06c831630ab6662fd8ea-Paper-Conference.pdf)

### Matrizes Estruturadas
- [CirCNN: Accelerating and Compressing DNNs Using Block-Circulant Weight Matrices](https://arxiv.org/abs/1708.08917)
- [Real block-circulant matrices and DCT-DST algorithm for transformer neural network](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2023.1260187/full)
- [Structured Unrestricted-Rank Matrices for Parameter Efficient Fine-tuning](https://arxiv.org/html/2406.17740v1)
- [Theoretical Properties for Neural Networks with Weight Matrices of Low Displacement Rank](https://arxiv.org/abs/1703.00144)
- [Structured Matrices and Their Application in Neural Networks: A Survey](https://link.springer.com/article/10.1007/s00354-023-00226-1)
- [Learning Compressed Transforms with Low Displacement Rank](https://pmc.ncbi.nlm.nih.gov/articles/PMC6534145/)

### Teoria de Matrizes Aleatorias
- [Random matrix theory analysis of neural network weight matrices](https://openreview.net/pdf?id=41kpc2Nzwc)
- [WeightWatcher: Data-Free Diagnostics for Deep Learning](https://weightwatcher.ai/)
- [Predicting trends in the quality of state-of-the-art neural networks (Nature Comms)](https://www.nature.com/articles/s41467-021-24025-8)
- [Traditional and Heavy-Tailed Self Regularization in Neural Network Models](https://arxiv.org/abs/1901.08276)
- [From SGD to Spectra: A Theory of Neural Network Weight Dynamics](https://arxiv.org/html/2507.12709)
- [Grokking, Rank Minimization and Generalization in Deep Learning](https://openreview.net/forum?id=6NHnsjsYXH)
- [Approaching Deep Learning through the Spectral Dynamics of Weights](https://arxiv.org/html/2408.11804v1)

### Correlacao Espacial e Visualizacao de Pesos
- [Visualizing Weights (Distill)](https://distill.pub/2020/circuits/visualizing-weights/)
- [Weight Banding (Distill)](https://distill.pub/2020/circuits/weight-banding/)
- [How does Weight Correlation Affect the Generalisation Ability of DNNs](https://arxiv.org/abs/2010.05983)

### Compressao de LLMs e Metodos Modernos
- [Neural Weight Compression for Language Models](https://arxiv.org/html/2510.11234)
- [DFloat11: Lossless LLM Compression](https://arxiv.org/abs/2504.11651)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Low-Rank Prehab: Preparing Neural Networks for SVD Compression](https://arxiv.org/abs/2512.01980)
- [A survey of model compression techniques: past, present, and future](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1518965/full)

### Distribuicao de Pesos e Entropia
- [Deep neural networks with dependent weights: heavy tails, sparsity and compressibility](https://arxiv.org/abs/2205.08187)
- [Heavy Tails in SGD and Compressibility of Overparametrized Neural Networks](http://www.cs.utoronto.ca/~erdogdu/papers/comp-heavy.pdf)
