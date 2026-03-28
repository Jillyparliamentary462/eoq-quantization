# Pesquisa: Dequantizacao Neural e Tecnicas Relacionadas

**Data:** 2026-03-28
**Escopo:** Levantamento abrangente sobre dequantizacao aprendida, quantizacao nao-linear, hiperrredes para compressao, e tecnicas transferiveis de compressao de audio neural.

---

## 1. Dequantizacao Aprendida / Dequantizador Neural

### 1.1 Conceito Fundamental

A ideia central e substituir a operacao tradicional de table lookup na dequantizacao por uma pequena rede neural que aprende um mapeamento nao-linear dos codigos quantizados para os valores reconstruidos. Em vez de armazenar um codebook fixo com N entradas, uma MLP compacta poderia gerar os valores de reconstrucao de forma parametrica.

### 1.2 Quantizacao com Formatos Aprendidos de Baixo Bit

O trabalho "Quantized GEMM with Learned Low-Bit Formats" demonstra tecnicas algoritmicas e de hardware onde os mapeamentos de quantizacao sao derivados por metodos data-driven e adaptativos. Esses metodos exploram estatisticas locais ou objetivos de otimizacao para alocar bitwidths e determinar centroides de quantizacao.

**Referencia:** [Quantized GEMM with Learned Low-Bit Formats](https://www.emergentmind.com/topics/quantized-gemm-with-learned-low-bit-formats)

### 1.3 Codebooks Aprendidos Conjuntamente

O trabalho "Network Memory Footprint Compression Through Jointly Learnable Codebooks and Mappings" (OpenReview) propoe o aprendizado conjunto do codebook e dos mapeamentos de pesos, junto com uma nova definicao de atualizacao de gradiente que permite busca proximal dos codebooks e seus mapeamentos.

**Resultados-chave:** Redes neurais podem ser comprimidas 10-100x com perda minima ou ate negativa de acuracia, validado em tarefas de visao, linguagem e regressao.

**Referencia:** [Jointly Learnable Codebooks and Mappings](https://openreview.net/forum?id=1RrOtCmuKr&noteId=Gscr8RywTf)

### 1.4 VQ4ALL: Codebook Universal

VQ4ALL (2024) e uma abordagem baseada em quantizacao vetorial que compartilha um codebook universal entre multiplas redes neurais. A tecnica:

- Usa estimacao de densidade por kernel (KDE) para extrair um codebook universal
- Constroi progressivamente diferentes redes de baixo bit atualizando atribuicoes diferenciaveis
- Alcanca taxas de compressao superiores a 16x mantendo alta acuracia
- O codebook universal pode ser armazenado em ROM, reduzindo area de silicio e eliminando necessidade de recarregamento de codebook

**Implicacao para dequantizacao neural:** Se um codebook universal funciona para multiplas redes, uma MLP que aprende esse mapeamento poderia ser ainda mais compacta e flexivel.

**Referencia:** [VQ4ALL](https://arxiv.org/html/2412.06875v1)

### 1.5 LCQ: Codebook de Baixo Rank

LCQ (Low-Rank Codebook based Quantization, 2024) introduz codebooks de rank superior a um para LLMs. A maioria dos metodos existentes usa codebook rank-one, o que resulta em perda substancial de acuracia em altas taxas de compressao. LCQ tem capacidade de representacao mais forte, incluindo metodos rank-one como casos especiais.

**Referencia:** [LCQ](https://arxiv.org/abs/2405.20973)

---

## 2. Metodos de Quantizacao Nao-Linear

### 2.1 NormalFloat (NF4) no bitsandbytes

NF4 e um formato de dados de 4 bits introduzido no paper QLoRA, projetado com base na observacao de que pesos de redes neurais pre-treinadas frequentemente seguem distribuicao normal centrada em zero.

**Design do mapeamento:**
- Os 16 niveis representaveis sao escolhidos como quantis da distribuicao normal padrao N(0,1)
- Cada valor representa uma proporcao igual da distribuicao N(0,1)
- Mais precisao e alocada para valores proximos de zero (onde a maior parte dos pesos normalmente distribuidos reside)
- Menos precisao para valores de maior magnitude nas caudas

**Valores especificos do NF4:** -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0

**Normalizacao:** Usa normalizacao blockwise absolute-maximum (absmax) por blocos de pesos.

**Implementacao:** A funcao `create_normal_map` em `functional.py` do bitsandbytes gera o mapeamento.

**Referencia:** [NF4 Quantization](https://www.emergentmind.com/topics/4-bit-normalfloat-nf4-quantization), [bitsandbytes NF4 Analysis](https://id2thomas.medium.com/ml-bitsandbytes-nf4-quantize-dequantize-analysis-1ad91d9912c9)

### 2.2 Algoritmo Lloyd-Max e Alem

O quantizador Lloyd-Max e o quantizador escalar otimo que minimiza o MSQE (Mean Squared Quantization Error) para um numero fixo de niveis de saida. Suas condicoes necessarias de otimalidade sao:

- **Condicao de ponto medio:** Cada nivel de decisao entre dois bins e o ponto medio dos niveis de saida adjacentes
- **Condicao de centroide:** O nivel de saida em cada bin e a expectativa condicional da entrada dentro daquele bin

**KDE-Lloyd-Max (KDE-LM):** Extensao que usa estimacao de densidade por kernel a partir dos pesos originais para construir quantizadores mais eficientes, realizando calculos em um numero muito menor de dados.

**Relacao com k-means:** O algoritmo Linde-Buzo-Gray (LBG) generaliza Lloyd-Max usando k-means clustering para particionar o espaco de entrada em regioes de Voronoi.

**Referencia:** [KDE-based Non-uniform Quantizer](https://www.mdpi.com/2076-3417/9/12/2559)

### 2.3 Quantization Networks (CVPR 2019)

Primeiro trabalho a formular quantizacao como funcao de mapeamento nao-linear diferenciavel. A funcao de quantizacao e formada como **combinacao linear de varias funcoes Sigmoid com vieses e escalas aprendiveis**.

- Permite aprendizado end-to-end sem problemas de gradient mismatch
- A quantizacao e alcancada via relaxacao continua da inclinacao (steepness) das funcoes Sigmoid durante o treinamento
- Solucao geral para quantizacao de qualquer bit tanto de pesos quanto de ativacoes

**Referencia:** [Quantization Networks - CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Quantization_Networks_CVPR_2019_paper.pdf)

### 2.4 Learnable Companding Quantization (LCQ, CVPR 2021)

Propoe funcoes de companding aprendiveis que controlam de forma flexivel e nao-uniforme os niveis de quantizacao de pesos e ativacoes.

- Otimiza conjuntamente pesos do modelo e funcoes de companding aprendiveis
- ResNet-50 de 2 bits no ImageNet alcanca top-1 accuracy de 75.1% (gap de apenas 1.7% para full-precision)
- Inclui tecnica de normalizacao de pesos para treinamento mais estavel

**Relevancia direta:** As funcoes de companding sao efetivamente um "dequantizador neural" simples -- uma funcao nao-linear aprendida que mapeia indices para valores.

**Referencia:** [Learnable Companding Quantization](https://openaccess.thecvf.com/content/CVPR2021/papers/Yamamoto_Learnable_Companding_Quantization_for_Accurate_Low-Bit_Neural_Networks_CVPR_2021_paper.pdf)

### 2.5 Learnable Non-Uniform Step Sizes (nuLSQ, ACCV 2024)

Otimiza tamanhos de passo individuais para quantizar pesos e ativacoes, superando abordagens uniformes em multiplos datasets e arquiteturas.

**Referencia:** [nuLSQ](https://link.springer.com/chapter/10.1007/978-981-96-0966-6_4)

### 2.6 AQLM: Quantizacao Aditiva para LLMs

AQLM (ICML 2024) aplica Multi-Codebook Quantization (MCQ) para compressao extrema de LLMs:

- Divide cada vetor em subgrupos aproximados usando vetores aprendidos (codewords)
- Primeiro esquema Pareto-otimo para compressao abaixo de 3 bits por parametro
- Melhora significativamente sobre todos os esquemas conhecidos no regime de 2 bits
- Pode igualar ou superar baseline em ponto flutuante em velocidade, reduzindo footprint de memoria em ate 8x

**Referencia:** [AQLM](https://arxiv.org/abs/2401.06118), [AQLM Explained](https://towardsdatascience.com/the-aqlm-quantization-algorithm-explained-8cf33e4a783e/)

### 2.7 QuIP#: Codebooks de Lattice E8

QuIP# (ICML 2024) combina tres tecnicas para compressao extrema:

1. **Processamento de incoerencia** via transformada randomizada de Hadamard (RHT) -- pesos transformados seguem distribuicao aproximadamente Gaussiana esferica
2. **Codebooks baseados no lattice E8** -- quantizacao vetorial usando o lattice E8, que alcanca o empacotamento otimo de esferas em 8 dimensoes. O codebook E8P requer apenas 1KiB e pode ser decodificado com menos de 5 instrucoes por peso
3. **Fine-tuning** para melhorar fidelidade ao modelo original

**Desempenho:** 3x mais rapido que inferencia FP16 e 5x mais rapido que a abordagem de VQ direto do AQLM.

**Referencia:** [QuIP#](https://arxiv.org/abs/2402.04396)

---

## 3. Dequantizacao Dependente de Contexto

### 3.1 Quantizacao Mista Ciente de Contexto

Classe de metodologias que otimizam eficiencia de memoria e computacao atribuindo bit-widths discretos a pesos, ativacoes ou elementos de cache de forma que se adapta a sensibilidade contextual variavel de diferentes camadas, chunks ou tokens.

**Exemplos:**
- **KVmix:** Preserva dinamicamente precisao total para uma janela recente em declinio ("Recent Pivotal Context") por camada, quantizando tokens mais antigos usando alocacao de bit-width derivada de scores de importancia
- **Cocktail:** Divide contexto em chunks de tamanho fixo, atribui bit-width por chunk baseado em similaridade com a query
- **ADQ (Adaptive Distribution-aware Quantization):** Constroi quantizadores cujos parametros sao dinamicamente adaptados a distribuicao atual de pesos/ativacoes
- **ASQ (Adaptive Step Size Quantization):** Ajusta tamanhos de passo dinamicamente

**Referencia:** [Context-Aware Mixed-Precision Quantization](https://www.emergentmind.com/topics/context-aware-mixed-precision-quantization)

### 3.2 Quantizacao Dinamica Instance-Aware

Adapta a precisao da rede em tempo de execucao baseada em caracteristicas da entrada, alocando maiores recursos computacionais apenas para entradas mais desafiadoras.

### 3.3 RAVN: Quantizacao Vetorial com Aprendizado por Reforco

RAVN (CVPR 2024 Workshop) usa quantizacao vetorial (VQ) e aprendizado por reforco (RL) para ajuste adaptativo de bit-width em Post-Training Quantization (PTQ). O RL permite quantizacao dinamica, marcando uma mudanca significativa de metodos estaticos.

**Referencia:** [RAVN](https://openaccess.thecvf.com/content/CVPR2024W/EVW/papers/Jha_RAVN_Reinforcement_Aided_Adaptive_Vector_Quantization_of_Deep_Neural_Networks_CVPRW_2024_paper.pdf)

---

## 4. Compressao de Redes Neurais usando Hiperrredes

### 4.1 Conceito de Hiperrredes (HyperNetworks)

Hiperrredes sao redes neurais que geram pesos para outra rede neural (a rede alvo). Uma rede pequena (a "hiperrede") gera os pesos para uma rede maior (a "rede principal").

**Paper original:** Ha, Dai, Le - "HyperNetworks" (ICLR 2017). A hiperrede recebe um conjunto de entradas contendo informacoes sobre a estrutura dos pesos e gera o peso para cada camada.

**Referencia:** [HyperNetworks - ICLR 2017](https://openreview.net/pdf?id=rkpACe1lx)

### 4.2 Compressao via Hiperrredes

Hiperrredes menores sao treinadas para gerar redes alvo maiores, reduzindo o footprint de memoria e requisitos computacionais. A hiperrede pode ser vista como um **sistema de codificacao/decodificacao** para comprimir pesos de redes neurais em uma representacao latente simples, onde a hiperrede descomprime os codigos latentes comprimidos de volta em pesos para a rede alvo.

**Aplicacoes documentadas:**
- Eficiencia de parametros (pesos aprendiveis da hiperrede menores que da DNN padrao)
- Aprendizado continuo
- Inferencia causal
- Transfer learning
- Poda de pesos
- Quantificacao de incerteza

**Referencia:** [Brief Review of Hypernetworks](https://arxiv.org/pdf/2306.06955), [Springer Review](https://link.springer.com/article/10.1007/s10462-024-10862-8)

### 4.3 Decoder-Only Random Hypernetworks (D'OH, ACCV 2024)

Arquitetura onde um vetor de codigo latente treinavel e pesos de projecao aleatorios fixos geram pesos da rede alvo. Isso fornece uma forma natural de variar o footprint de memoria alterando diretamente a dimensao do codigo latente.

**Relevancia para dequantizacao neural:** A ideia de gerar pesos a partir de codigos compactos via uma rede decodificadora e exatamente o paradigma de um "dequantizador neural".

**Referencia:** [D'OH - ACCV 2024](https://openaccess.thecvf.com/content/ACCV2024/papers/Gordon_DOH_Decoder-Only_Random_Hypernetworks_for_Implicit_Neural_Representations_ACCV_2024_paper.pdf)

### 4.4 Modelagem Generativa de Pesos

Abordagens recentes (2024-2025):
- **Metodos baseados em autoencoder:** Aprendem embedding latente compacta de pesos a partir de um "model zoo", depois amostram do espaco latente para gerar novos pesos
- **Abordagens autoregressivas baseadas em tokens:** VQ-VAE+Transformer representam parametros do modelo como sequencias de codigos discretos, permitindo geracao coerente entre arquiteturas arbitrarias
- **Modelos de difusao:** Operam no espaco completo de pesos ou em espaco latente encapsulado por VAE

**Referencia:** [Generative Modeling of Neural Network Weights](https://www.emergentmind.com/topics/generative-modeling-of-neural-network-weights)

---

## 5. Inferencia Amortizada em Compressao

### 5.1 VAEs e Inferencia Amortizada

Em autoencoders variacionais, o encoder funciona como abordagem amortizada para otimizar conjuntamente entre pontos de dados, com os mesmos parametros reutilizados para multiplos pontos de dados. O encoder comprime dados em um espaco latente (bottleneck), aprendendo uma compressao eficiente dos dados em espaco de menor dimensao.

### 5.2 VAE para Compressao de Modelos

O paper "Variational Autoencoder-based Neural Network Model Compression" (2024) propoe usar VAE como metodo de compressao de modelos onde:

- O **encoder** reduz os parametros do modelo para o espaco latente como representacao do modelo comprimido
- O **decoder** e responsavel por reconstruir e gerar um modelo completo para uso
- O espaco latente intermediario serve como representacao comprimida, esperando-se alcancr taxas de compressao mais altas

**Referencia:** [VAE-based NN Model Compression](https://arxiv.org/html/2408.14513v1)

### 5.3 Inferencia Amortizada Semi-Amortizada

O VAE vanilla realiza inferencia por um unico forward pass pela rede encoder (inferencia amortizada), que permite inferencia rapida mas com custo de degradacao de acuracia (erro de amortizacao). Trabalhos como "Semi-Amortized Variational Autoencoders" (ICML 2018) e "Iterative Amortized Inference" (ICML 2018) propoem combinar inferencia amortizada com passos iterativos de refinamento.

**Aplicacao potencial:** Um encoder amortizado poderia encontrar codigos otimos para qualquer matriz de pesos em um unico forward pass, com refinamento iterativo opcional para melhorar a qualidade.

**Referencia:** [Semi-Amortized VAEs](https://proceedings.mlr.press/v80/kim18e/kim18e.pdf), [Iterative Amortized Inference](https://proceedings.mlr.press/v80/marino18a/marino18a.pdf)

### 5.4 Compressao Neural com Decodificacao em Tempo de Inferencia

O paper "Efficient Neural Compression with Inference-time Decoding" (2024) explora decodificacao eficiente em tempo de inferencia para compressao neural, combinando modelos de entropia aprendidos com decodificacao pratica.

**Referencia:** [Efficient Neural Compression](https://arxiv.org/html/2406.06237v1)

### 5.5 Codificacao de Entropia Aprendida para Pesos

Trabalhos recentes (2024-2025) combinam quantizacao de redes neurais com codificacao de entropia para minimizar footprint de memoria. Abordagens combinam mixed precision, quantizacao de zero-point e codificacao de entropia (como tANS) para empurrar a compressao alem dos limites tradicionais.

**Referencia:** [ICLR 2025 - Neural Weight Compression](https://proceedings.iclr.cc/paper_files/paper/2025/file/f429c8a1472dc4dd2f8dd2f5b7ed8917-Paper-Conference.pdf)

---

## 6. A Restricao de Velocidade

### 6.1 Table Lookup vs. MLP Pequena: Analise de Performance

**Table lookup (LUT):**
- Instrucoes tbl/pshuf em CPUs permitem table lookup extremamente rapido
- T-MAC (Microsoft, 2024) demonstra que LUTs podem substituir completamente a dequantizacao em multiplicacao de matrizes mista, com scaling linear de FLOPs e latencia em relacao ao numero de bits
- FLUTE (EMNLP 2024) alcanca 91.3-121.7 tokens/s para Llama3-8B 4-bit em GPUs A6000/A100

**MLP pequena (ex: 2 camadas, 32 unidades ocultas):**
- Uma MLP com input 4-16 dimensoes, hidden 32, output 1 requer ~(16x32 + 32x1) = ~544 multiplicacoes + ativacoes
- Em contraste, um table lookup requer 1 acesso a memoria
- A diferenca e de 2-3 ordens de magnitude em operacoes

### 6.2 Otimizacao SIMD de MLPs Pequenas

- Em CPUs modernas com AVX-512 ou NEON, uma MLP pequena pode processar 16+ elementos em paralelo via instrucoes SIMD
- Contudo, o overhead de ativacoes nao-lineares (ReLU, sigmoid) e memory fetches reduz o ganho
- Para MLPs muito pequenas, o overhead de setup de instrucoes SIMD pode dominar

### 6.3 Tensor Cores e Matrizes Pequenas

**Requisitos de alinhamento para Tensor Cores NVIDIA:**
- Dimensoes devem ser multiplos de 16 bytes (8 para FP16)
- No A100, multiplos de potencias de 2 ate 128 bytes (64 para FP16) melhoram eficiencia
- K divisivel por 8 e necessario para uso eficiente

**Problema para MLPs minusculas:** Uma MLP com 32 unidades ocultas teria matrizes 32x32 ou menores. Tensor Cores em GPUs modernas processam tiles minimos de 16x16 (ou 8x8 para FP16), entao matrizes muito pequenas teriam baixa utilizacao. Para M muito menor que 64, a dimensao M e preenchida com zeros, desperdicando recursos.

**Referencia:** [NVIDIA Matrix Multiplication Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)

### 6.4 Kernel Fusion como Alternativa

A abordagem mais pratica seria **fundir o dequantizador com a operacao GEMM:**
- 8 elementos dequantizados com apenas 7 instrucoes no melhor caso
- Dequantizacao de primeiro nivel fundida no epilogo da GEMM com custo negligivel
- Fused dequant + GEMM usando SplitK alcanca 64-124% de speedup medio (pico de 295%) no H100

**Conclusao critica:** Um dequantizador neural baseado em MLP seria ordens de magnitude mais lento que table lookup direto. A abordagem viavel seria usar o dequantizador neural apenas durante o treinamento/otimizacao, e materializar o codebook resultante como LUT para inferencia.

**Referencia:** [T-MAC](https://github.com/microsoft/t-mac), [FLUTE](https://github.com/HanGuo97/flute)

### 6.5 T-MAC e FLUTE: Estado da Arte em LUT-based Inference

**T-MAC (Microsoft):**
- Biblioteca de kernels para multiplicacao de matrizes mixed-precision (int1/2/3/4 x int8/fp16/fp32) sem dequantizacao
- Pesos de 1 bit agrupados (ex: grupos de 4), todas as somas parciais possiveis pre-computadas em LUT
- Scaling linear de FLOPs com numero de bits

**FLUTE (EMNLP 2024):**
- Motor flexivel de lookup table para LLMs quantizados com LUT
- Reestruturacao offline da matriz de pesos quantizados para minimizar manipulacoes de bits
- Vetorizacao e duplicacao da lookup table para mitigar restricoes de bandwidth de shared memory
- Suporte a bit-widths nao-uniformes (ex: 3 bits) com quantizacao nao-uniforme via LUT

---

## 7. Trabalho Relacionado em Compressao de Audio

### 7.1 SoundStream (Google, 2021)

Codec de audio neural end-to-end composto por:
- **Encoder** convolucional que gera representacao latente
- **Quantizador Vetorial Residual (RVQ)** que produz representacao comprimida
- **Decoder** convolucional que reconstroi o sinal no dominio do tempo

**Tecnicas transferiveis:**
- **Residual Vector Quantization (RVQ):** Multiplas camadas de quantizacao onde cada camada quantiza o residuo da anterior. Poderia ser aplicado a compressao de pesos.
- **Quantizer Dropout:** Durante treinamento, camadas de quantizacao sao aleatoriamente removidas para simular bitrate variavel. O decoder aprende a funcionar bem em qualquer bitrate, criando um modelo "escalavel".

**Referencia:** [SoundStream](https://arxiv.org/abs/2107.03312)

### 7.2 EnCodec (Meta, 2022)

Codec de audio neural de alta fidelidade com arquitetura encoder-decoder convolucional streaming:
- Usa RVQ no output flutuante do encoder neural
- Discriminador adversarial multi-escala baseado em STFT (MS-STFT) para reduzir artefatos
- Combina losses adversariais e de reconstrucao
- Opera em tempo real em um unico nucleo de CPU

**Desempenho:** SOTA em fala e musica monoforica a 1.5-12 kbps (24 kHz) e musica estereo a 6-24 kbps (48 kHz).

**Referencia:** [EnCodec](https://github.com/facebookresearch/encodec)

### 7.3 Lyra V2 (Google, 2022)

Baseado na estrutura SoundStream, com encoder e decoder neurais:
- Integracao de RVQ permite mudar bitrate a qualquer momento selecionando numero de quantizadores
- Mais quantizadores = audio de maior qualidade (custo: bitrate maior)
- Avancos de text-to-speech e speech enhancement transferidos para o treinamento

**Referencia:** [Lyra V2](https://opensource.googleblog.com/2022/09/lyra-v2-a-better-faster-and-more-versatile-speech-codec.html)

### 7.4 Codec2 + WaveNet

Em 2017, pesquisadores do Google substituiram o decoder do Codec2 por uma rede neural WaveNet. A rede neural foi capaz de **extrapolar caracteristicas da voz nao descritas no bitstream do Codec2**, resultando em melhor qualidade de audio.

**Analogia direta para pesos:** Assim como o WaveNet pode reconstruir audio de alta qualidade a partir de features comprimidas do Codec2, um "decoder neural de pesos" poderia reconstruir pesos de alta precisao a partir de codigos comprimidos, extrapolando informacoes nao capturadas pela quantizacao.

### 7.5 Tecnicas Transferiveis para Descompressao de Pesos

| Tecnica de Audio | Aplicacao em Pesos |
|---|---|
| Residual VQ (RVQ) | Quantizacao multi-estagio de pesos: cada estagio corrige o erro residual do anterior |
| Quantizer Dropout | Treinar decoder que funcione com diferentes niveis de compressao |
| Discriminador adversarial | Treinar dequantizador para produzir pesos "realistas" que mantenham performance da rede |
| Encoder-decoder streaming | Dequantizacao on-the-fly durante inferencia, sem materializar todos os pesos |
| Treinamento end-to-end | Otimizar quantizador e dequantizador conjuntamente com a tarefa final |

---

## 8. Sintese e Direcoes Promissoras

### 8.1 O Que Ja Existe

1. **Codebooks aprendidos** sao amplamente usados (AQLM, QuIP#, VQ4ALL, LCQ)
2. **Mapeamentos nao-lineares aprendiveis** foram demonstrados (Quantization Networks, Learnable Companding)
3. **Hiperrredes** ja geram pesos a partir de codigos latentes (D'OH, modelagem generativa de pesos)
4. **Inferencia amortizada** via VAE para compressao de modelos existe (VAE-based NN Compression)
5. **LUTs para inferencia** sao altamente otimizadas (T-MAC, FLUTE)

### 8.2 O Gap: Dequantizador Neural em Tempo de Inferencia

O principal gap identificado e: **ninguem usa uma MLP como dequantizador em tempo real durante inferencia para pesos de rede neural**. As razoes sao:

1. **Velocidade:** Uma MLP de 2 camadas com 32 unidades e ~500x mais lenta que um table lookup
2. **Paralelismo:** Tensor Cores nao sao eficientes para matrizes muito pequenas
3. **Overhead:** O custo por peso dequantizado e ordens de magnitude maior

### 8.3 Abordagem Hibrida Viavel

A abordagem mais promissora seria:

1. **Treinar** um dequantizador neural (MLP) que aprende o mapeamento otimo codigos -> valores
2. **Materializar** o codebook resultante como LUT apos o treinamento
3. **Usar a LUT** durante inferencia para velocidade maxima

Isso combina o poder de representacao de uma rede neural (durante otimizacao) com a velocidade de table lookup (durante inferencia). A rede neural serve como "regularizador" que encontra um codebook otimo considerando o contexto global da tarefa.

**Variante avancada:** Treinar codebooks condicionais (por camada, por posicao) que sao materializados como LUTs diferentes para diferentes contextos, inspirado nas tecnicas de quantizacao ciente de contexto.

### 8.4 Quando Faria Sentido uma MLP Real em Inferencia

Uma MLP como dequantizador poderia fazer sentido se:

1. O **numero de codigos possiveis for muito grande** (ex: quantizacao vetorial com codebook de milhoes de entradas), onde uma LUT seria proibitiva em memoria
2. O custo for **amortizado sobre muitos pesos** simultaneamente (batch dequantization)
3. A MLP puder ser **fundida no kernel GEMM** como operacao de epilogo
4. A tarefa exigir **dequantizacao adaptativa** (ex: diferentes mapeamentos para diferentes entradas/contextos), impossivel com LUT estatica

### 8.5 Conexao com Compressao de Audio Neural

A analogia mais forte e com o **decoder do SoundStream/EnCodec**: um pequeno decoder convolucional reconstroi sinal de alta qualidade a partir de codigos RVQ comprimidos. Da mesma forma, um "decoder de pesos" poderia:

- Receber codigos quantizados (indices VQ/RVQ)
- Usar informacao de contexto (posicao na camada, estatisticas locais)
- Gerar pesos reconstruidos de alta fidelidade
- Ser treinado end-to-end com a tarefa da rede principal

---

## 9. Referencias Principais

### Quantizacao Aprendida e Codebooks
- [VQ4ALL: Universal Codebook (2024)](https://arxiv.org/html/2412.06875v1)
- [LCQ: Low-Rank Codebook (2024)](https://arxiv.org/abs/2405.20973)
- [AQLM: Additive Quantization for LLMs (ICML 2024)](https://arxiv.org/abs/2401.06118)
- [QuIP#: Lattice Codebooks (ICML 2024)](https://arxiv.org/abs/2402.04396)
- [Jointly Learnable Codebooks and Mappings](https://openreview.net/forum?id=1RrOtCmuKr&noteId=Gscr8RywTf)

### Quantizacao Nao-Linear
- [Quantization Networks (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Quantization_Networks_CVPR_2019_paper.pdf)
- [Learnable Companding Quantization (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Yamamoto_Learnable_Companding_Quantization_for_Accurate_Low-Bit_Neural_Networks_CVPR_2021_paper.pdf)
- [NF4 in bitsandbytes](https://www.emergentmind.com/topics/4-bit-normalfloat-nf4-quantization)
- [nuLSQ: Non-Uniform Step Sizes (ACCV 2024)](https://link.springer.com/chapter/10.1007/978-981-96-0966-6_4)
- [KDE-based Non-uniform Quantizer](https://www.mdpi.com/2076-3417/9/12/2559)

### Hiperrredes e Geracao de Pesos
- [HyperNetworks (ICLR 2017)](https://openreview.net/pdf?id=rkpACe1lx)
- [Brief Review of Hypernetworks (2023)](https://arxiv.org/pdf/2306.06955)
- [D'OH: Decoder-Only Random Hypernetworks (ACCV 2024)](https://openaccess.thecvf.com/content/ACCV2024/papers/Gordon_DOH_Decoder-Only_Random_Hypernetworks_for_Implicit_Neural_Representations_ACCV_2024_paper.pdf)
- [Generative Modeling of Neural Network Weights](https://www.emergentmind.com/topics/generative-modeling-of-neural-network-weights)

### Inferencia Amortizada
- [VAE-based NN Model Compression (2024)](https://arxiv.org/html/2408.14513v1)
- [Semi-Amortized VAEs (ICML 2018)](https://proceedings.mlr.press/v80/kim18e/kim18e.pdf)
- [Iterative Amortized Inference (ICML 2018)](https://proceedings.mlr.press/v80/marino18a/marino18a.pdf)
- [Efficient Neural Compression with Inference-time Decoding (2024)](https://arxiv.org/html/2406.06237v1)

### Velocidade e Implementacao
- [T-MAC: CPU LUT-based Inference (Microsoft)](https://github.com/microsoft/t-mac)
- [FLUTE: Flexible LUT Engine (EMNLP 2024)](https://github.com/HanGuo97/flute)
- [NVIDIA Matrix Multiplication Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
- [RAVN: RL-based Adaptive VQ (CVPR 2024 Workshop)](https://openaccess.thecvf.com/content/CVPR2024W/EVW/papers/Jha_RAVN_Reinforcement_Aided_Adaptive_Vector_Quantization_of_Deep_Neural_Networks_CVPRW_2024_paper.pdf)

### Compressao de Audio Neural
- [SoundStream (Google, 2021)](https://arxiv.org/abs/2107.03312)
- [EnCodec (Meta, 2022)](https://github.com/facebookresearch/encodec)
- [Lyra V2 (Google, 2022)](https://opensource.googleblog.com/2022/09/lyra-v2-a-better-faster-and-more-versatile-speech-codec.html)
- [Codec2 + WaveNet](https://en.wikipedia.org/wiki/Lyra_(codec))
