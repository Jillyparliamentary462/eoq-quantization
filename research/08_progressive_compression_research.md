# Pesquisa: Compressao Progressiva e Tecnicas Relacionadas para Redes Neurais

**Data:** 2026-03-28
**Escopo:** Carregamento progressivo, representacoes Matryoshka, auto-compressao, LOD para redes neurais, codificacao por bit-planes, streaming de modelos e teoria taxa-distorcao.

---

## Indice

1. [Carregamento/Inferencia Progressiva de Redes Neurais](#1-carregamentoinferencia-progressiva-de-redes-neurais)
2. [Representacoes Matryoshka](#2-representacoes-matryoshka)
3. [Redes Neurais Auto-Compressiveis](#3-redes-neurais-auto-compressiveis)
4. [Level of Detail (LOD) para Redes Neurais](#4-level-of-detail-lod-para-redes-neurais)
5. [Codificacao por Bit-Planes para Redes Neurais](#5-codificacao-por-bit-planes-para-redes-neurais)
6. [Streaming e Carregamento Parcial de Modelos](#6-streaming-e-carregamento-parcial-de-modelos)
7. [Teoria Taxa-Distorcao Aplicada a Redes Neurais](#7-teoria-taxa-distorcao-aplicada-a-redes-neurais)
8. [Sintese e Conexoes entre os Temas](#8-sintese-e-conexoes-entre-os-temas)

---

## 1. Carregamento/Inferencia Progressiva de Redes Neurais

### 1.1 Redes Progressivas (Progressive Neural Networks)

O conceito de "progressive neural networks" foi introduzido por Rusu et al. (2016) no contexto de aprendizado sequencial de tarefas. A rede expande sua estrutura incrementalmente, adicionando colunas dedicadas para cada nova tarefa enquanto mantem conexoes laterais para colunas anteriores (somente leitura). O objetivo original era evitar o esquecimento catastrofico em aprendizado continuo, nao necessariamente compressao progressiva de pesos.

**Referencia:** [Progressive Neural Networks - arXiv 1606.04671](https://arxiv.org/abs/1606.04671)

### 1.2 Inferencia Coarse-to-Fine (Grosseira para Fina)

Varios trabalhos implementam arquiteturas que realizam classificacao em multiplos estagios com complexidade crescente:

- **CF-CNN (Coarse-to-Fine CNN):** Usa uma CNN menor para classificacao inicial grosseira, seguida de classificacao fina baseada no resultado inicial. Isso reduz o custo computacional para amostras "faceis".
- **Redes Progressivas para Classificacao de Imagens:** Estruturas com unidades de rede ativadas sequencialmente com complexidade crescente. Combinando unidades sequenciais com saidas antecipadas baseadas em confianca, conseguem escalabilidade de complexidade de mais de 10x mantendo acuracia competitiva no CIFAR-10 e ImageNet.

**Referencia:** [Coarse-to-Fine DNN Inference - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0167739X24000736)

### 1.3 Redes Anytime (Predicao a Qualquer Momento)

Um preditor "anytime" produz um resultado grosseiro rapidamente e continua refinando-o ate esgotar o orcamento computacional de tempo de teste. Caracteristicas-chave:

- **Quanto mais orcamento gasto, melhor a predicao.** O preditor se ajusta automaticamente ao orcamento disponivel.
- **Perdas auxiliares adaptativas:** Uma abordagem geral que adiciona cabecas de predicao intermediarias com perdas ponderadas. Os pesos sao inversamente proporcionais a media de cada perda, balanceando as perdas para terem a mesma escala.
- **Escalabilidade:** Uma sequencia de ANNs exponencialmente mais profundas pode atingir resultados anytime quase-otimos em qualquer orcamento, ao custo de uma fracao constante de orcamento adicional consumido.

**Referencias:**
- [Learning Anytime Predictions - arXiv 1708.06832](https://arxiv.org/abs/1708.06832)
- [Anytime Neural Network - OpenReview](https://openreview.net/forum?id=SJa1Nk10b)

### 1.4 Quantizacao Progressiva

A quantizacao progressiva refere-se a uma familia de tecnicas onde a quantizacao eh conduzida em multiplos estagios cuidadosamente agendados, cada estagio tipicamente empregando uma discretizacao mais fina ou de menor precisao que o anterior. Objetivos:

- Mitigar acumulo de erro induzido por quantizacao
- Adaptar a resolucao do quantizador a sensibilidade dos dados/modelo
- Habilitar adaptacao eficiente a restricoes dinamicas de recursos
- Suportar representacoes escaláveis em qualidade ou sucessivamente refinaveis

**Metodo notavel - PFCR (Progressive Fine-to-Coarse Reconstruction):** Reconstrucao progressiva de fina para grossa para Vision Transformers, combinando unidades de granularidade fina em blocos mais grossos com reotimizacao iterativa.

**Referencia:** [Progressive Quantization Overview - EmergentMind](https://www.emergentmind.com/topics/progressive-quantization)

### 1.5 Any-Precision Deep Neural Networks

Uma contribuicao particularmente relevante para compressao progressiva de pesos:

- **Conceito:** Uma unica rede treinada que pode ser executada com precisoes numericas diferentes em tempo de inferencia. O mesmo modelo em execucao pode ser ajustado flexivelmente para diferentes larguras de bits, **truncando os bits menos significativos**.
- **Mecanismo:** Treina-se conjuntamente o modelo em um subconjunto selecionado de larguras de bits. Quando todas as camadas sao configuradas para baixa precisao, os modelos alcancam acuracia comparavel a modelos dedicados treinados na mesma precisao.
- **Vantagem pratica:** Quando a demanda de trade-off eficiencia/acuracia varia ou muda dinamicamente em tempo de execucao, eh inviavel re-treinar modelos; a abordagem any-precision resolve isso com um unico modelo.

**Referencia:** [Any-Precision Deep Neural Networks - AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17286)

> **Relevancia direta para compressao DCT progressiva:** Este trabalho demonstra que pesos podem ser organizados de forma que bits mais significativos contenham a informacao mais critica, permitindo truncamento progressivo -- analogamente a como coeficientes DCT de baixa frequencia capturam a maior parte da energia de um sinal.

---

## 2. Representacoes Matryoshka

### 2.1 Matryoshka Representation Learning (MRL)

**Autores:** Kusupati, Bhatt, Rege, Wallingford, Sinha, Ramanujan, Howard-Snyder, Chen, Kakade, Jain, Farhadi (NeurIPS 2022)

**Conceito central:** Codificar informacao em diferentes granularidades dentro de um unico embedding, criando uma estrutura aninhada (como bonecas russas). O embedding de dimensao d=8 esta contido dentro de d=16, que esta dentro de d=32, ate o embedding completo (ex: d=2048).

**Mecanismo de treinamento:**
- Funcao de perda multi-escala: o treinamento MRL computa a perda multiplas vezes -- uma para cada tamanho de dimensao alvo.
- O modelo eh forcado a produzir boas representacoes nao apenas na dimensao completa, mas em multiplos tamanhos menores.
- As dimensoes iniciais acabam codificando as features mais amplamente uteis, enquanto dimensoes posteriores lidam com detalhes mais finos.

**Resultados:**
- **14x menor** tamanho de embeddings para classificacao ImageNet-1K mantendo acuracia
- **14x de aceleracoes** reais para tarefas de recuperacao em larga escala
- **2% de ganho** de acuracia em classificacao few-shot de cauda longa
- Funciona com ViT, ResNet, ALIGN, BERT

**Referencia:** [Matryoshka Representation Learning - arXiv 2205.13147](https://arxiv.org/abs/2205.13147)

### 2.2 Conexao com Quantizacao Progressiva

O framework Matryoshka foi recentemente estendido para incorporar quantizacao:

- **Quantization Aware Matryoshka Adaptation (QAMA):** Aprende embeddings aninhados que reduzem graciosamente para subconjuntos dimensionais menores e utiliza operacoes bitwise (XOR, NOT, POPCOUNT) para recuperacao eficiente. O treinamento end-to-end usa perda consciente de quantizacao.
- **Combinacao MRL + Quantizacao Escalar:** Partindo de Float32 com 384 dimensoes para Int8 ja reduz armazenamento em 63.7%. MRL para 128 dimensoes da 56.6% de reducao. MRL 128 dimensoes + quantizacao escalar atinge **77.9% de reducao**.

**Referencia:** [Quantization Aware Matryoshka Adaptation - ACM CIKM 2024](https://dl.acm.org/doi/10.1145/3746252.3761077)

### 2.3 Nested Dropout e Representacoes Ordenadas

O nested dropout eh uma variante que remove conjuntos coerentes aninhados de unidades escondidas. Isso forca a rede a organizar features por importancia:

- Unidades anteriores contem informacao mais critica
- Equivalencia exata com PCA quando aplicado adequadamente
- **Variational Nested Dropout (VND):** Extensao que aprende a taxa de dropout otima durante treinamento

**Referencia:** [Learning Ordered Representations with Nested Dropout - ResearchGate](https://www.researchgate.net/publication/260062234_Learning_Ordered_Representations_with_Nested_Dropout)

> **Insight-chave:** MRL e nested dropout demonstram que eh possivel e pratico organizar representacoes neurais de forma que subconjuntos menores (dimensoes iniciais) contenham informacao progressivamente mais importante -- um principio diretamente aplicavel a compressao progressiva de pesos.

---

## 3. Redes Neurais Auto-Compressiveis

### 3.1 Self-Distillation (Auto-Destilacao)

Diferente da destilacao de conhecimento tradicional (teacher-student separados), a auto-destilacao transfere conhecimento **dentro da mesma rede** -- das camadas mais profundas para as mais rasas.

**Mecanismo:**
- Modulos de atencao e classificadores rasos sao adicionados em diferentes profundidades da rede
- Conhecimento eh destilado do classificador mais profundo para os mais rasos
- Todos os modulos auxiliares sao removidos durante a inferencia (sem custo adicional de parametros ou computacao)

**Resultados:**
- Melhoria media de 3.49% no CIFAR100 e 2.32% no ImageNet
- Melhoria variando de 0.61% (ResNeXt) a 4.07% (VGG19)
- Permite inferencia dinamica com diferentes profundidades

**Referencia:** [Be Your Own Teacher - arXiv 1905.08094](https://arxiv.org/abs/1905.08094)

### 3.2 Destilacao de Conhecimento como Auto-Compressao

Na destilacao de conhecimento classica (Hinton 2015), um modelo grande (teacher) transfere seu "conhecimento" -- as distribuicoes de probabilidade suaves das saidas -- para um modelo menor (student). Variantes modernas incluem:

- **Destilacao offline vs. online:** Teacher fixo vs. co-treinado
- **Auto-destilacao:** O modelo ensina a si mesmo via suas proprias camadas
- **Multiplos teachers:** Ensemble de modelos como teacher
- **Destilacao cross-modal:** Teacher e student em modalidades diferentes

### 3.3 EPSD: Early Pruning with Self-Distillation

Combinacao de poda precoce com auto-destilacao para compressao mais eficiente. A auto-destilacao complementa a poda ao preservar e transferir conhecimento critico das camadas profundas durante o processo de compressao.

**Referencia:** [EPSD - arXiv 2402.00084](https://arxiv.org/abs/2402.00084)

> **Relevancia:** A auto-destilacao demonstra que redes neurais contem informacao redundante intrinsecamente organizada por "importancia" (camadas mais profundas = mais informacao). Isso paralela a ideia de que pesos podem ser organizados de MSB para LSB em termos de importancia.

---

## 4. Level of Detail (LOD) para Redes Neurais

### 4.1 Redes com Saida Antecipada (Early Exit Networks)

As redes early-exit estendem DNNs classicas adicionando cabecas de predicao auxiliares em pontos intermediarios, permitindo que o modelo "saia antecipadamente" se criterios de confianca forem satisfeitos.

**Politicas de saida:**
- **Baseadas em regras (estaticas):** Comparacao de entropia, limiar de probabilidade softmax. Baixa complexidade de treinamento, mas menor robustez.
- **Aprendiveis (dinamicas):** Controladores de selecao de saida, tecnicas de aprendizado por reforco. Alta complexidade de treinamento, mas excelente adaptabilidade.

**SLEXNet:** Combina redes early-exit com "slimming" (afinamento), permitindo adaptacao simultanea de profundidade E largura em tempo de execucao.

**Referencias:**
- [Early-Exit Deep Neural Networks Survey - ACM Computing Surveys](https://dl.acm.org/doi/full/10.1145/3698767)
- [Adaptive Inference through Early-Exit Networks](https://arxiv.org/pdf/2106.05022)

### 4.2 Adaptive Computation Time (ACT)

Proposto por Alex Graves, o ACT permite que redes neurais recorrentes aprendam **quantos passos computacionais realizar** entre entrada e saida.

**Mecanismo do halting score:**
- Uma unidade sigmoidal (halting unit) decide se deve parar ou continuar a computacao
- O processamento para quando a soma das saidas das halting units se aproxima de 1.0
- Um "ponder cost" eh adicionado a perda total, penalizando computacao excessiva

**Referencia:** [Adaptive Computation Time - arXiv 1603.08983](https://arxiv.org/abs/1603.08983)

### 4.3 Slimmable Neural Networks

Uma unica rede treinada que pode ser executada com diferentes larguras (numero de canais por camada) em tempo de execucao.

**Mecanismo:** Batch normalization switchavel permite que a rede ajuste sua largura dinamicamente conforme restricoes de hardware.

**Universally Slimmable Networks (US-Nets):**
- Extensao que permite execucao em largura **arbitraria**
- Tecnicas: "sandwich rule" (treinar na maior e menor largura, mais amostras aleatorias) e "inplace distillation" (destilacao in-loco do modelo mais largo para o mais estreito)
- Acuracia similar ou superior a modelos treinados individualmente (MobileNet v1/v2, ShuffleNet, ResNet-50)

**Referencia:** [Slimmable Neural Networks - arXiv 1812.08928](https://arxiv.org/abs/1812.08928)

### 4.4 Once-for-All (OFA) Networks

Treina uma unica rede que suporta diversas configuracoes arquiteturais: profundidade, largura, tamanho de kernel e resolucao elasticos.

**Progressive Shrinking Algorithm:** Um metodo generalizado de poda que reduz o tamanho do modelo em multiplas dimensoes simultaneamente, prevenindo interferencia entre sub-redes de diferentes tamanhos.

**Resultados:** Supera metodos NAS estado-da-arte (ate 4.0% de melhoria top-1 ImageNet sobre MobileNetV3) reduzindo ordens de magnitude em horas-GPU e emissoes de CO2.

**Referencia:** [Once-for-All - arXiv 1908.09791](https://arxiv.org/abs/1908.09791)

### 4.5 Neural Geometric Level of Detail (NGLOD)

Aplica o conceito de LOD da computacao grafica a funcoes de distancia com sinal (SDFs) neurais, usando um volume de features baseado em octree que se adapta a formas com multiplos niveis discretos de LOD. Camadas iniciais fornecem estimativas rapidas de forma; camadas posteriores adicionam detalhes.

**Referencia:** [NGLOD - NVIDIA Research](https://nv-tlabs.github.io/nglod/)

> **Insight-chave para compressao progressiva de pesos:** Os conceitos de early-exit, slimmable networks e OFA demonstram que uma unica rede pode operar em multiplos pontos de trade-off acuracia/eficiencia. O principio pode ser estendido: em vez de ajustar a **arquitetura**, podemos ajustar a **precisao dos pesos** progressivamente.

---

## 5. Codificacao por Bit-Planes para Redes Neurais

### 5.1 Conceito de Bit-Planes

Um bit-plane de um sinal discreto digital eh o conjunto de bits correspondentes a uma posicao de bit dada nos numeros binarios que representam o sinal. Para dados de 16 bits, existem 16 bit-planes: o primeiro contem o bit mais significativo (MSB) e o 16o contem o menos significativo (LSB).

### 5.2 Codificacao Progressiva por Bit-Planes (Classica)

No contexto de compressao de imagem (SPIHT, EZW), o bitstream embutido eh gerado codificando primeiro o MSB de todos os coeficientes wavelet, seguido pelo proximo MSB, e assim por diante. O numero de bit-planes comunicados eh determinado pela taxa de dados ou requisitos de qualidade da aplicacao. Compressao sem perda eh alcancada transmitindo todos os bit-planes.

### 5.3 Treinamento Bit-a-Bit de Pesos Neurais

**Trabalho-chave: "Bit-wise Training of Neural Network Weights" (2022)**

Um algoritmo que **aprende os bits individuais** representando os pesos de uma rede neural.

**Descobertas cruciais:**
- **Os primeiros 3 MSBs contribuem a maior parte para alta acuracia**, enquanto o restante fornece regularizacao intrinseca.
- **Mais de 90% da rede pode armazenar codigos arbitrarios** sem afetar a acuracia -- os bits de menor significancia podem conter ruido aleatorio, arquivos binarios ou ate pesos de outras redes.
- Naturalmente descobre redes esparsas sem restricoes ou tecnicas de regularizacao adicionais.
- Resultados melhores que treinamento padrao para redes totalmente conectadas, e desempenho similar para convolucionais e residuais.

**Referencia:** [Bit-wise Training - arXiv 2202.09571](https://arxiv.org/abs/2202.09571)

### 5.4 DPICT: Compressao Progressiva com Trit-Planes

Extensao do conceito de bit-planes para compressao progressiva de imagens usando "trit-planes" (ternarios) com aprendizado profundo. Combina a abordagem classica de transmissao progressiva com codificacao neural.

**Referencia:** [DPICT - CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_DPICT_Deep_Progressive_Image_Compression_Using_Trit-Planes_CVPR_2022_paper.pdf)

> **Achado crucial:** A descoberta de que os 3 MSBs dos pesos contem a grande maioria da informacao util eh **diretamente relevante** para compressao progressiva no dominio DCT. Sugere que uma abordagem onde coeficientes DCT de baixa frequencia (analogos aos MSBs em termos de importancia de informacao) sao transmitidos primeiro pode ser altamente eficaz.

---

## 6. Streaming e Carregamento Parcial de Modelos

### 6.1 Estrategias de Carregamento Parcial de LLMs

Como a inferencia eh sequencial pelas camadas (camada 0, depois 1, depois 2...), teoricamente apenas uma camada precisa estar na VRAM por vez.

**Tres abordagens principais:**

1. **llama.cpp com GPU Offloading:** Flag `-ngl` para especificar camadas GPU vs. CPU. Particao estatica simples.
2. **HuggingFace Accelerate:** Colocacao automatica de camadas com offload para disco. Carregamento sincronico -- cria latencia.
3. **oLLM Layer Streaming:** Prefetching assincrono: enquanto uma camada processa na GPU, a proxima esta sendo carregada. Reduz latencia efetiva significativamente.

### 6.2 Benchmarks de Performance

Testes em APU AMD Strix Halo (128GB memoria unificada) mostram resultados contra-intuitivos:

| Configuracao | Tokens/segundo |
|---|---|
| GPU completa (16 camadas) | 1.92 |
| 50% offloaded (8 camadas CPU) | 2.26 |
| 75% offloaded (12 camadas CPU) | 3.36 |

O carregamento parcial superou GPU completa em 75% devido a arquitetura de memoria unificada (operacao de tabela de paginas, nao copia de dados).

### 6.3 Latencia por Camada por Fonte

| Fonte | Latencia por camada |
|---|---|
| NVMe SSD | 108-162ms |
| DDR5 RAM | 10-15ms |
| PCIe 4.0 x16 | 25-37ms |

### 6.4 Prefetching Avancado

- **PRESERVE:** Prefetching de pesos e KV-cache de HBM para cache L2, sobrepondo prefetching com comunicacao para esconder latencia.
- **ProMoE:** Preditor aprendido que faz prefetch de experts em modelos Mixture-of-Experts.
- **Pre-gated MoE:** Resultados de gate da camada anterior determinam experts necessarios antecipadamente.

**Referencias:**
- [Partial LLM Loading - TinyComputers](https://tinycomputers.io/posts/partial-llm-loading-running-models-too-big-for-vram.html)
- [PRESERVE - arXiv 2501.08192](https://arxiv.org/html/2501.08192v2)

> **Conexao com compressao progressiva:** Se os pesos forem armazenados em formato progressivo (MSBs primeiro, ou coeficientes DCT de baixa frequencia primeiro), o prefetching pode ser estendido para "prefetch de resolucao" -- carregar uma versao grosseira da proxima camada enquanto a camada atual processa na resolucao completa, e depois refinar.

---

## 7. Teoria Taxa-Distorcao Aplicada a Redes Neurais

### 7.1 Framework Teorico

A teoria taxa-distorcao de Shannon fornece um limite inferior no trade-off fundamental entre taxa (numero de bits para descrever o modelo) e distorcao (diferenca entre modelo comprimido e original).

**Resultado principal para modelos lineares (Gao et al., ICML 2019):**

Para regressao linear f_w(x) = w^T x:

```
R(D) >= (1/2) log det(Sigma_W) - Sum((1/2) log(D_i))
```

onde D_i satisfaz condicoes de "water-filling" ponderado baseadas na covariancia de entrada.

**Generalizacao para redes nao-lineares:**

Para redes ReLU de uma camada oculta, minimizar `(w - w_hat)^T I_w (w - w_hat)` eh **otimo**, onde I_w eh a "matriz de importancia de pesos":

```
I_w = E_X[nabla_w f_w(X) (nabla_w f_w(X))^T]
```

### 7.2 Regras Praticas Derivadas da Teoria

Duas "regras de ouro" para compressao:
1. **Condicao de ortogonalidade:** `w_hat^T I_w (w - w_hat) = 0`
2. **Minimizar perturbacao ponderada de parametros:** `(w - w_hat)^T I_w (w - w_hat)`

Tanto poda quanto quantizacao naturalmente satisfazem essas condicoes quando otimizadas corretamente.

### 7.3 Lacuna Teoria-Pratica

O compressor otimo para modelos lineares eh "intratavel na pratica" devido a distribuicoes continuas. Nao ha quantificacao explicita de quao proximo metodos praticos estao dos limites teoricos em termos absolutos, mas aproximacoes baseadas em gradiente superam significativamente objetivos L2 de baseline.

**Referencia:** [Rate Distortion For Model Compression - ICML 2019](https://proceedings.mlr.press/v97/gao19c/gao19c.pdf)

### 7.4 Avancos Recentes em Taxa-Distorcao para Compressao Neural

**Quantizacao Restrita por Taxa + Codificacao de Entropia (2025):**
- Estende objetivos de perda por camada incorporando estimativa quadratica de taxa
- Solucoes localmente exatas via Optimal Brain Surgeon (OBS)
- **20-40% de reducao em taxa de bits** versus NNCodec na mesma performance

**Referencia:** [Rate-Constrained Quantization - arXiv 2505.18758](https://arxiv.org/abs/2505.18758)

### 7.5 Estado da Arte: Quantos Bits por Peso Sao Necessarios?

#### Metodos Praticos Atuais

| Metodo | Bits/peso | Qualidade vs FP16 | Abordagem |
|---|---|---|---|
| GPTQ | 3-4 bits | ~92-95% | Hessiana segunda-ordem |
| GGUF Q4_K_M | ~4.5 bits | ~94% | Quantizacao por bloco |
| AWQ | 4 bits | ~93-95% | Ativacao-aware |
| QuIP | 2 bits | Primeiro viavel | Incoerencia + rounding adaptativo |
| QuIP# | 2-3 bits | Sub-redes 3-bit > 4-bit | Hadamard + lattice E8 |
| AQLM | 2-4 bits | Estado-da-arte em 2-bit | VQ com codebooks 8D |
| BitNet b1.58 | 1.58 bits | ~Llama 2 FP16 | Ternario {-1,0,+1}, treinado |

#### Limites Teoricos

- **1 bit (binario):** Limite absoluto inferior. BNNs com pesos {+1, -1} funcionam mas com degradacao significativa para LLMs.
- **1.58 bits (ternario):** Log_2(3) = 1.585 bits. O BitNet b1.58 demonstra que modelos treinados do zero neste regime igualam FP16 em certos benchmarks. Speedups de 2.37x-6.17x em CPUs x86, reducoes de energia de 71.9-82.2%.
- **~2-3 bits:** Fronteira atual de metodos post-training (QuIP#, AQLM). Qualidade aceitavel mas nao sem perda.
- **~4 bits:** "Sweet spot" pratico atual onde a degradacao eh minima (< 5% perplexidade).
- **~6 bits:** "Joelho da curva" onde quantizacao uniforme comeca a degradar significativamente.

#### Codificacao de Entropia e Bits Fracionarios

A codificacao de entropia permite **bits fracionarios por peso**, em alguns casos alcancando taxas abaixo de 1 bit por peso (com ajuda de esparsidade). A distribuicao natural de pesos treinados concentra-se em torno de 0, o que favorece quantizadores centrados em zero e reduz a entropia dos pesos quantizados.

### 7.6 DeepCABAC e o Padrao ISO/MPEG-NNR

**DeepCABAC** eh um codificador aritmetico binario context-adaptativo para compressao de redes neurais:

- Quantiza cada parametro minimizando uma funcao taxa-distorcao ponderada
- Comprime os valores quantizados em bitstream com redundancias minimas
- **VGG16 ImageNet comprimido 63.6x** (de ~500MB para 8.7MB) sem perda de acuracia

**Padrao ISO/IEC 15938-17:2022 (MPEG-NNR):**
- Padrao internacional para representacao comprimida de redes neurais
- Usa DeepCABAC como motor de codificacao principal
- Comprime modelos para **5-20% do tamanho original** sem perda de performance
- Em alguns casos, abaixo de **3% do tamanho original**
- Segunda edicao em desenvolvimento (2025-2026)

**Referencia:** [Neural Network Coding - Fraunhofer HHI](https://www.hhi.fraunhofer.de/en/departments/ai/research-groups/efficient-deep-learning/research-topics/neural-network-compression.html)

### 7.7 Deep Compression Pipeline (Han et al.)

O pipeline classico de tres estagios de Song Han:

1. **Poda:** Remove conexoes nao-importantes (9-13x reducao)
2. **Quantizacao treinada:** Agrupa pesos e retina (32 bits -> 5 bits)
3. **Codificacao Huffman:** Explora distribuicao enviesada dos pesos efetivos

**Resultado:** AlexNet reduzido de 240MB para 6.9MB (35x) sem perda de acuracia. Melhor artigo ICLR 2016.

**Referencia:** [Deep Compression - arXiv 1510.00149](https://arxiv.org/abs/1510.00149)

---

## 8. Sintese e Conexoes entre os Temas

### 8.1 O Principio Unificador: Informacao Hierarquica

Todos os temas pesquisados convergem para um principio: **a informacao em redes neurais eh inerentemente hierarquica e pode ser organizada por importancia.**

| Dominio | Hierarquia | Analogia |
|---|---|---|
| Bit-planes | MSB -> LSB | Os 3 MSBs contem ~90% da informacao util |
| Matryoshka | Dim 8 -> 16 -> ... -> 2048 | Dimensoes iniciais = features mais importantes |
| Early-exit | Camada 1 -> 2 -> ... -> N | Camadas iniciais = predicoes grosseiras |
| DCT/Wavelet | Baixa freq -> Alta freq | Coeficientes baixos = maior energia |
| Quantizacao | 2 bits -> 4 -> 8 -> 16 -> 32 | Mais bits = mais precisao, retornos decrescentes |

### 8.2 Compressao Progressiva de Pesos: Estado da Arte e Lacunas

**O que ja existe:**
- Any-Precision Networks: truncamento de bits LSB com modelo unico
- Bit-wise training: demonstra dominancia dos MSBs
- Matryoshka: representacoes aninhadas por dimensao
- MPEG-NNR: codificacao aritmetica otimizada com DeepCABAC
- Quantizacao progressiva: multiplos estagios de refinamento

**O que aparentemente NAO existe (lacuna identificada):**
- **Compressao progressiva de pesos no dominio de frequencia (DCT/wavelet):** Nao encontramos trabalho que aplique codificacao progressiva tipo JPEG/SPIHT diretamente nos pesos da rede neural -- i.e., transformar blocos de pesos para dominio DCT, quantizar os coeficientes progressivamente (DC primeiro, AC depois), e transmitir/armazenar em bitstream progressivo.
- **Streaming de pesos com refinamento progressivo de resolucao:** Carregar versao grosseira de todos os pesos, depois refinar progressivamente enquanto a inferencia ja comeca -- combinando streaming de camadas com refinamento de precisao.
- **Framework unificado taxa-distorcao para compressao progressiva:** Embora a teoria taxa-distorcao seja aplicada a compressao de redes, nao ha framework especifico para o cenario progressivo (successive refinement) de pesos.

### 8.3 Oportunidade de Pesquisa: Compressao DCT Progressiva de Pesos

A combinacao das tecnicas pesquisadas sugere uma abordagem novel:

1. **Transformacao:** Aplicar DCT (ou wavelet) a blocos de pesos
2. **Ordenacao:** Coeficientes ordenados por importancia (energia/frequencia)
3. **Quantizacao progressiva:** Coeficientes mais importantes quantizados com mais bits
4. **Bitstream embutido:** Formato que permite decodificacao parcial com qualidade crescente
5. **Inferencia progressiva:** Comecar inferencia com pesos grosseiros, refinar conforme mais dados chegam

**Evidencias de viabilidade:**
- Bit-wise training mostra que 3 MSBs bastam para >90% da acuracia
- Matryoshka prova que representacoes aninhadas funcionam na pratica
- PLONQ demonstra quantizacao aninhada com refinamento progressivo para imagens
- DeepCABAC/MPEG-NNR validam codificacao aritmetica eficiente para pesos
- Any-Precision Networks mostram que truncamento progressivo de bits eh viavel

### 8.4 Dominio de Frequencia para Pesos: Trabalho Existente

**CNNpack (NeurIPS 2016):** Comprime camadas convolucionais usando K-means clustering no espaco DCT. Mais de 85% dos pesos podem ser zerados. A rede comprimida realiza convolucao diretamente no dominio de frequencia sem DCT inversa.

**Compressao de NN via Wavelet Learnable:** A transformada wavelet rapida comprime camadas lineares. Bases wavelet e coeficientes sao aprendidos conjuntamente. RNNs comprimidas competem com estado-da-arte com muito menos parametros.

**Referencia:** [Neural Network Compression via Learnable Wavelet Transforms - arXiv 2004.09569](https://arxiv.org/abs/2004.09569)

### 8.5 Tabela Resumo de Tecnicas Relevantes

| Tecnica | Ano | Progressivo? | Dominio | Bits/peso | Detalhe-chave |
|---|---|---|---|---|---|
| Deep Compression | 2015 | Nao | Espacial | ~5 | Pipeline poda+quant+Huffman |
| Progressive NNs | 2016 | Sim* | Arquitetura | N/A | *Progressao de tarefas, nao pesos |
| CNNpack (DCT) | 2016 | Nao | Frequencia | Variavel | K-means no dominio DCT |
| ACT (Graves) | 2016 | Sim | Computacao | N/A | Profundidade adaptativa |
| Nested Dropout | 2014 | Sim | Features | N/A | Ordenacao por importancia |
| Slimmable Networks | 2018 | Sim | Largura | N/A | Largura adaptativa |
| MRL (Matryoshka) | 2022 | Sim | Embedding | N/A | Dimensao adaptativa |
| DeepCABAC/NNR | 2019-22 | Nao | Bits | <1 (c/ esparsidade) | Padrao ISO, CABAC |
| Bit-wise Training | 2022 | Sim | Bits | 3 MSBs | 90% da rede pode ser ruido |
| Any-Precision | 2021 | Sim | Bits | 2-8 | Truncamento LSB |
| QuIP# | 2024 | Nao | Vetorial | 2-3 | Lattice E8 + Hadamard |
| BitNet b1.58 | 2024 | Nao | Ternario | 1.58 | Treinado, nao PTQ |
| PLONQ | 2021 | Sim | Latente | Variavel | Quantizacao aninhada para imagens |
| OFA | 2020 | Sim | Arquitetura | N/A | Shrinking progressivo |
| PFCR | 2025 | Sim | Quantizacao | Variavel | Fine-to-coarse para ViTs |

---

## Fontes Principais

### Carregamento Progressivo e Inferencia
- [Progressive Neural Networks - arXiv 1606.04671](https://arxiv.org/abs/1606.04671)
- [Coarse-to-Fine DNN Inference](https://www.sciencedirect.com/science/article/pii/S0167739X24000736)
- [Learning Anytime Predictions - arXiv 1708.06832](https://arxiv.org/abs/1708.06832)
- [Any-Precision Deep Neural Networks - AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/17286)
- [Progressive Quantization Overview](https://www.emergentmind.com/topics/progressive-quantization)

### Representacoes Matryoshka
- [Matryoshka Representation Learning - NeurIPS 2022](https://arxiv.org/abs/2205.13147)
- [Matryoshka Embedding Models - HuggingFace](https://huggingface.co/blog/matryoshka)
- [Quantization Aware Matryoshka Adaptation - ACM](https://dl.acm.org/doi/10.1145/3746252.3761077)
- [Matryoshka Embeddings Guide - Supermemory](https://supermemory.ai/blog/matryoshka-representation-learning-the-ultimate-guide-how-we-use-it/)

### Auto-Compressao e Self-Distillation
- [Be Your Own Teacher - arXiv 1905.08094](https://arxiv.org/abs/1905.08094)
- [Self-Distillation IEEE TPAMI](https://ieeexplore.ieee.org/document/9381661/)
- [EPSD - arXiv 2402.00084](https://arxiv.org/abs/2402.00084)

### LOD e Computacao Adaptativa
- [Early-Exit Survey - ACM Computing Surveys](https://dl.acm.org/doi/full/10.1145/3698767)
- [Adaptive Computation Time - arXiv 1603.08983](https://arxiv.org/abs/1603.08983)
- [Slimmable Neural Networks - arXiv 1812.08928](https://arxiv.org/abs/1812.08928)
- [Once-for-All - arXiv 1908.09791](https://arxiv.org/abs/1908.09791)
- [NGLOD - NVIDIA](https://nv-tlabs.github.io/nglod/)

### Bit-Planes e Codificacao de Pesos
- [Bit-wise Training - arXiv 2202.09571](https://arxiv.org/abs/2202.09571)
- [DPICT Trit-Planes - CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_DPICT_Deep_Progressive_Image_Compression_Using_Trit-Planes_CVPR_2022_paper.pdf)
- [Nested Dropout](https://www.researchgate.net/publication/260062234_Learning_Ordered_Representations_with_Nested_Dropout)

### Streaming e Carregamento Parcial
- [Partial LLM Loading - TinyComputers](https://tinycomputers.io/posts/partial-llm-loading-running-models-too-big-for-vram.html)
- [PRESERVE Prefetching - arXiv 2501.08192](https://arxiv.org/html/2501.08192v2)

### Taxa-Distorcao e Limites Teoricos
- [Rate Distortion For Model Compression - ICML 2019](https://proceedings.mlr.press/v97/gao19c/gao19c.pdf)
- [Rate-Constrained Quantization - arXiv 2505.18758](https://arxiv.org/abs/2505.18758)
- [Deep Compression - arXiv 1510.00149](https://arxiv.org/abs/1510.00149)
- [DeepCABAC - arXiv 1905.08318](https://arxiv.org/abs/1905.08318)
- [MPEG-NNR Standard - ISO](https://www.iso.org/standard/78480.html)
- [QuIP - arXiv 2307.13304](https://arxiv.org/abs/2307.13304)
- [QuIP# - arXiv 2402.04396](https://arxiv.org/abs/2402.04396)
- [AQLM - arXiv 2401.06118](https://arxiv.org/html/2401.06118v2)
- [BitNet b1.58 - Wikipedia](https://en.wikipedia.org/wiki/1.58-bit_large_language_model)

### Dominio de Frequencia
- [CNNpack DCT Compression - NeurIPS 2016](https://proceedings.neurips.cc/paper/2016/file/3636638817772e42b59d74cff571fbb3-Reviews.html)
- [Wavelet Compression for NNs - arXiv 2004.09569](https://arxiv.org/abs/2004.09569)
- [Quantization Survey - arXiv 2103.13630](https://arxiv.org/pdf/2103.13630)
- [Neural Network Quantization White Paper - arXiv 2106.08295](https://arxiv.org/pdf/2106.08295)
