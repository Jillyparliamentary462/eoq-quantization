# Pesquisa: Compressao de LLMs Baseada em SVD

**Data da pesquisa:** 2026-03-28
**Escopo:** Decomposicao em valores singulares (SVD) para compressao de Large Language Models, combinacao com quantizacao, decomposicoes tensoriais, e analise comparativa com metodos de quantizacao tradicionais.

---

## 1. Compressao Baseada em SVD para LLMs - Estado da Arte (2023-2026)

### 1.1 ASVD: Activation-aware Singular Value Decomposition (2023)

**Referencia:** Yuan et al., "ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models", arXiv:2312.05821

**Problema central:** A decomposicao SVD direta das matrizes de peso ignora a distribuicao das ativacoes, que frequentemente contem outliers significativos. Esses outliers causam grandes erros de reconstrucao quando a matriz de peso e truncada ingenuamente.

**Abordagem tecnica:**
- Transforma a matriz de peso com base na distribuicao das ativacoes, absorvendo os outliers na matriz de peso transformada antes da decomposicao
- Processo iterativo de calibracao otimiza a decomposicao por camada, respeitando a sensibilidade variavel de diferentes camadas do LLM
- Metodo training-free (sem necessidade de re-treinamento)

**Resultados empiricos (LLaMA-7B, WikiText-2 perplexity):**

| Razao de Parametros | Perplexidade | Degradacao vs Original (5.68) |
|---------------------|-------------|-------------------------------|
| 0.95 (5% compressao) | 5.78 | +0.10 |
| 0.90 (10% compressao) | 6.09 | +0.41 |
| 0.85 (15% compressao) | 6.80 | +1.12 |
| 0.80 (20% compressao) | 8.89 | +3.21 |

**Resultados (LLaMA-2-13B):** Degradacao muito menor em modelos maiores:
- 0.95 ratio: 4.94 (original 4.88, delta +0.06)
- 0.90 ratio: 5.12 (delta +0.24)
- 0.85 ratio: 5.54 (delta +0.66)

**Conclusao:** ASVD alcanca 10-30% de compressao de modelo e 50% de reducao no KV cache sem perda significativa de desempenho. Porem, a perda escala rapidamente apos 15-20% de compressao.

### 1.2 SVD-LLM: Truncation-aware SVD (ICLR 2025)

**Referencia:** Wang et al., "SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression", ICLR 2025, arXiv:2403.07378

**Inovacoes principais:**
1. **Data whitening consciente de truncamento:** Garante um mapeamento direto entre valores singulares e perda de compressao, resolvendo o problema de que truncar valores singulares menores nem sempre resulta em menor perda
2. **Atualizacao de parametros com aproximacao low-rank sequencial:** Compensa a degradacao de precisao apos truncamento SVD

**Resultados empiricos (WikiText-2 perplexity, LLaMA-7B):**

| Compressao | ASVD | SVD-LLM (W only) | SVD-LLM (completo) |
|-----------|------|-------------------|---------------------|
| 20% | 11.14 | 7.94 | 7.73 |
| 40% | 1407 | 13.73 | 9.27 |
| 60% | 57057 | 66.62 | 15.00 |
| 80% | 80425 | 1349 | 31.79 |

**Observacao critica:** A diferenca entre metodos e dramatica em altas taxas de compressao. ASVD colapsa completamente apos 40% de compressao, enquanto SVD-LLM mantem perplexidade aceitavel.

**Comparacao com quantizacao (1-bit):**
- BiLLM (1-bit quant): 47.67 perplexidade
- PB-LLM (1-bit quant): 104.83 perplexidade
- SVD-LLM + 2-bit quant: 9.83 perplexidade (~1.3 GB)
- OneBit (requer treinamento): 10.20 perplexidade

**Conclusao importante:** SVD-LLM combinado com quantizacao de 2 bits supera metodos de quantizacao de 1 bit, mesmo metodos que requerem treinamento extensivo.

### 1.3 SVD-LLM V2 (NAACL 2025)

**Referencia:** "SVD-LLM V2: Optimizing Singular Value Truncation for Large Language Model Compression", NAACL 2025, arXiv:2503.12340

**Inovacoes:**
1. **Alocacao dinamica de razao de compressao:** Usa a perda teorica de truncamento de cada matriz para alocar razoes de compressao unicas por camada
2. **Truncamento otimizado para perda:** Garante que os valores singulares truncados resultem em perda mais baixa e estavel

**Resultados (WikiText-2 perplexity, 20% compressao):**

| Modelo | FWSVD | ASVD | SVD-LLM V1 | SVD-LLM V2 | Original |
|--------|-------|------|------------|------------|----------|
| LLaMA-7B | 1727 | 11.14 | 7.94 | **7.12** | 5.68 |
| LLaMA-3 8B | 4782 | 17.55 | 11.82 | **8.01** | 6.14 |

**Resultados (C4 perplexity, 20% compressao):**

| Modelo | ASVD | SVD-LLM V1 | SVD-LLM V2 | Original |
|--------|------|------------|------------|----------|
| LLaMA-7B | 15.93 | 15.84 | **10.47** | 7.34 |
| LLaMA-3 8B | 28.41 | 20.05 | **11.72** | 9.47 |

- Melhoria de ate 42% na perplexidade e 9% na precisao zero-shot vs SVD-LLM V1
- Quando combinado com quantizacao de 2 bits: 69% menor perplexidade que BiLLM
- Velocidade de compressao: 18 minutos para LLaMA-7B (vs 5.5 horas GPU para ASVD)

### 1.4 Zero Sum SVD (ZS-SVD, fevereiro 2026)

**Referencia:** "Zero Sum SVD: Balancing Loss Sensitivity for Low Rank LLM Compression", arXiv:2602.02848

**Inovacao principal:** Selecao global de componentes singulares usando uma "regra de soma zero" que mantem a mudanca cumulativa predita na perda proxima de zero, evitando a necessidade de otimizacao cara de alocacao de rank por camada.

**Resultados (LLaMA-7B, WikiText-2):**

| Compressao | ZS-SVD | SVD-LLM | ASVD | Dobi-SVD |
|------------|--------|---------|------|----------|
| 30% | **8.2** | 9.5 | 95.3 | - |
| 60% | **6.96** (c/ remap) | 13.11 | - | 13.54 |

**Speedups:** 5.63x em 40% compressao, 5.86x em 60% compressao (RTX A5000).

### 1.5 Outros Metodos SVD Recentes (2025-2026)

**ARA (Adaptive Rank Allocation, outubro 2025):** Alocacao adaptativa de rank por camada. LLaMA2-7B com 80% compressao: perplexidade de 6.42 vs 8.38 com compressao uniforme.

**NSVD (Nested Activation-aware Decomposition, marco 2025):** Melhora ASVD em 14.7%, 18.3% e 13.5% sob 30%, 40% e 50% de compressao, respectivamente.

**ERC-SVD (Error-Controlled SVD, maio 2025):** Comprime seletivamente apenas as ultimas camadas do modelo para mitigar propagacao de erros.

**DipSVD (Dual-importance Protected SVD, junho 2025):** Protege componentes de dupla importancia durante compressao.

**FLAR-SVD (Fast and Latency-Aware SVD, CVPR 2025 Workshop):** Otimiza para latencia real em hardware.

**FlashSVD (agosto 2025):** Framework de inferencia streaming que funde projecoes low-rank em FlashAttention, reduzindo memoria de ativacoes em ate 71%.

---

## 2. LoRA e sua Conexao com Compressao Low-Rank

### 2.1 Fundamento Teorico: Dimensionalidade Intrinseca

**Referencia:** Aghajanyan et al., "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning", arXiv:2012.13255

- Modelos pre-treinados super-parametrizados residem em um espaco de dimensao intrinseca baixa
- Fine-tuning de RoBERTa otimizando apenas ~200 parametros projetados aleatoriamente atinge 90% do desempenho full-parameter
- **Implicacao:** Modelos maiores tendem a ter menor dimensao intrinseca

### 2.2 LoRA: Low-Rank Adaptation

**Referencia:** Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022, arXiv:2106.09685

- Congela pesos pre-treinados e injeta matrizes de decomposicao de rank baixo treinaveis: W' = W + AB onde A in R^{d x r}, B in R^{r x k}, r << min(d,k)
- Reduz parametros treinaveis em 10.000x e memoria GPU em 3x vs fine-tuning completo
- **Conexao com SVD:** LoRA explora a mesma propriedade que a compressao SVD - a estrutura intrinsecamente low-rank das atualizacoes de peso

### 2.3 DoRA: Weight-Decomposed Low-Rank Adaptation (ICML 2024 Oral)

**Referencia:** Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", ICML 2024

- Decompoe peso pre-treinado em magnitude (m) e direcao (V)
- Aplica LoRA apenas na componente direcional, treinando magnitude separadamente
- Supera LoRA consistentemente em LLaMA, LLaVA e VL-BART
- **Conexao com SVD:** A decomposicao magnitude/direcao e conceitualmente similar a decomposicao em valores singulares

---

## 3. Low-Rank + Quantizacao Combinados

### 3.1 LQ-LoRA (ICLR 2024)

**Referencia:** Guo & Greengard, "LQ-LoRA: Low-rank Plus Quantized Matrix Decomposition for Efficient Language Model Finetuning", arXiv:2311.12023

**Abordagem:** Decompoe cada matriz pre-treinada em componente low-rank de alta precisao + componente quantizada de baixa precisao, usando programacao linear inteira para configuracao dinamica de bit-width.

**Resultados (LLaMA-2, WikiText-2 perplexity):**

| Metodo | Bits Efetivos | 7B | 70B |
|--------|---------------|-----|------|
| Full 16-bit | 16 | 5.47 | 3.31 |
| LQ-LoRA 2.75-bit | 2.95 (7B) / 2.85 (70B) | 5.67 | 3.65 |
| QLoRA 3-bit | 3.127 | pior | pior |
| OmniQuant 3-bit | 3 | 7.75 (C4) | - |
| LQ-LoRA 2.75-bit | 2.75 | 7.60 (C4) | 5.88 (C4) |

**Achados sobre sensibilidade ao rank:**
- Rank 32: degradacao notavel
- **Rank 64: baseline adequado**
- Rank 128: melhorias marginais

**Conclusao critica:** LQ-LoRA a 2.75 bits supera metodos de quantizacao pura a 3 bits. A combinacao low-rank + quantizacao e especialmente vantajosa no regime sub-3 bits.

### 3.2 CALDERA (NeurIPS 2024)

**Referencia:** "Compressing Large Language Models using Low Rank and Low Precision Decomposition", NeurIPS 2024, arXiv:2405.18886

**Formulacao:** W ≈ Q + LR, onde Q, L, R sao todos quantizados. Usa minimizacao alternada para resolver o problema de otimizacao.

**Configuracoes testadas:**
- BQ = 2 bits para matriz backbone
- Rank-64 com fatores em 4-bit: ~2.1 bits/parametro total
- Rank-64 com fatores em FP16: ~2.4 bits/parametro
- Rank-256 com fatores em 4-bit: ~2.4 bits/parametro

**Resultado principal:** Supera todas as tecnicas post-training existentes no regime de menos de 2.5 bits por parametro. Os fatores low-rank sao quantizaveis com perda minima, permitindo capturar mais componentes singulares do que com fatores em FP16 para o mesmo budget de bits.

### 3.3 SVDQuant (ICLR 2025 Spotlight)

**Referencia:** Li & Lin, "SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models", arXiv:2411.05007

- Absorve outliers de ativacoes em componente low-rank de alta precisao via SVD
- Branch quantizado em 4-bit lida com residuos
- Motor de inferencia Nunchaku funde kernels low-rank nos kernels quantizados
- Reducao de memoria 3.5x para FLUX.1 12B, speedup 3.0x vs quantizacao weight-only 4-bit

### 3.4 QA-LoRA e Outros

- **QA-LoRA:** Quantization-Aware LoRA que integra quantizacao durante adaptacao
- **DL-QAT:** Weight-Decomposed Low-Rank Quantization-Aware Training (EMNLP 2024)
- **Low-Rank QAT:** Adaptadores low-rank podem ser fundidos em matrizes inteiras quantizadas sem perda

---

## 4. Rank Efetivo de Matrizes de Peso em Transformers

### 4.1 Espectro de Valores Singulares

**Referencia:** Beren, "The Singular Value Decompositions of Transformer Weight Matrices are Highly Interpretable", Alignment Forum / LessWrong

**Padroes observados:**
- Em escala log-log, os valores singulares de matrizes MLP seguem uma **lei de potencia (power law)** consistente, com queda rapida no final
- Matrizes de attention heads mostram decaimento aproximadamente exponencial (linear em escala log)
- Todas as matrizes de peso sao "ligeiramente low-rank" - o espectro decai mas nao colapsa abruptamente

**Diferenca entre tipos de camada:**
- **Matrizes MLP:** Vetores singulares permanecem significativos/interpretaveis alem dos primeiros 50 valores singulares, tipicamente ate o 100o valor singular
- **Circuitos OV (attention):** Vetores singulares perdem significado mais rapidamente, limitados pela dimensao do head (tipicamente 64)
- **Camadas iniciais vs finais:** Espectros de ativacoes no residual stream aumentam em camadas posteriores

### 4.2 Pequenos Valores Singulares Importam

**Referencia:** "Small Singular Values Matter: A Random Matrix Analysis of Transformer Models", NeurIPS 2025, arXiv:2410.17770

**Descobertas revolucionarias:**
- Desvios significativos da Teoria de Matrizes Aleatorias (RMT) ocorrem nao apenas entre os maiores valores singulares, mas **tambem entre os menores**
- O **decil inferior (10% menores)** e a **terceira porcao mais influente** do espectro apos fine-tuning
- Remover valores singulares que desviam das predicoes RMT causa aumento de perplexidade **substancialmente maior** do que remover valores do bulk central
- **Matrizes Query (Q):** Mostram maiores outliers e maior desvio da distribuicao Marchenko-Pastur
- **Matrizes Attention-Output (O):** Permanecem dentro da distribuicao MP sem outliers significativos
- **Matrizes Up/Down (FFN):** O decil inferior carrega informacao surpreendentemente importante

**Implicacao para compressao SVD:** Estrategias de truncamento que ignoram valores singulares pequenos podem ser sub-otimas. Nao basta manter apenas os maiores.

### 4.3 Rank e Decay de Peso

**Referencia:** "Weight Decay Induces Low-Rank Attention Layers", 2024, arXiv:2410.23819

- AdamW com weight decay induz uma reducao consistente no rank das matrizes de atencao
- Camadas treinadas com weight decay mostram rank consistentemente menor
- O efeito e equivalente a regularizacao pela norma nuclear das matrizes

**Referencia:** "Transformers Learn Through Gradual Rank Increase", NeurIPS 2023

- Aumento gradual do stable rank durante treinamento
- Bias inerente para baixo rank no inicio do treinamento

### 4.4 LASER: Evidencia de Redundancia Espectral

**Referencia:** Sharma, Ash, Misra, "The Truth Is In There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction", ICLR 2024

**Descoberta surpreendente:** Remover valores singulares de certas camadas **melhora** o desempenho do LLM em 20-30 pontos percentuais em certas tarefas.

**Detalhes:**
- Melhorias vem principalmente de reducao em matrizes MLP na segunda metade do modelo
- Minimo de 25% dos valores singulares deve ser mantido para funcionalidade
- Com ratio 0.5 na camada 27 do Mistral: HumanEval Pass@1 = 0.2195 vs original 0.1768
- Interpretacao: SVD funciona como "denoising", removendo informacao erronea

**Implicacao:** Nem todos os valores singulares contribuem positivamente. Existe informacao "ruidosa" que pode ser removida com beneficio.

---

## 5. Poda Estruturada via SVD

### 5.1 Conceito: Poda no Dominio Espectral

A poda via SVD funciona selecionando os top-P valores singulares e descartando o restante. Para uma matriz W de dimensao m x n com rank r:
- W = U * Sigma * V^T
- Manter os top-k valores singulares: W_k = U_k * Sigma_k * V_k^T
- Parametros originais: m * n
- Parametros comprimidos: k * (m + n + 1) ≈ k * (m + n)
- **Razao de compressao:** k*(m+n) / (m*n) = k/n + k/m

Para uma matriz 4096x4096:
- k=64: 64*(4096+4096)/(4096*4096) = 524288/16777216 = **3.1% dos parametros**
- k=128: **6.25%**
- k=256: **12.5%**
- k=512: **25%**
- k=1024: **50%**

### 5.2 Metodos Avancados

- **Sparse Low-Rank:** Combina truncamento SVD com esparsidade estruturada nas matrizes de decomposicao
- **Regularizador Hoyer:** Concentra energia nos maiores valores singulares durante treinamento, facilitando truncamento posterior
- **Treinamento low-rank dinamico com regularizacao espectral:** Treina diretamente em formato fatorado low-rank

### 5.3 Conexao entre SVD e Pruning

A poda espectral via SVD e equivalente a structured pruning no espaco de features latentes:
- Cada valor singular define uma "direcao" no espaco de representacoes
- Remover um valor singular = remover um "neuronio virtual" no espaco SVD
- Vantagem sobre pruning de neuronios: as direcoes SVD sao **otimas** para capturar variancia maxima
- Desvantagem: requer multiplicacao de duas matrizes durante inferencia (a menos que sejam fundidas)

---

## 6. Decomposicoes Tensoriais Alem do SVD

### 6.1 Decomposicao Tucker

- Generaliza SVD para tensores de ordem superior (3D, 4D)
- Tensor original decomposto em tensor core pequeno + matrizes fatoriais
- Equivalente a PCA de ordem superior
- Compressao tipica: 2-3x sem perda significativa de qualidade

### 6.2 Decomposicao CP (Canonical Polyadic / CANDECOMP-PARAFAC)

- Expressa tensor como soma de produtos externos de vetores (tensores de rank 1)
- Funciona bem em redes pequenas (MNIST), mas e instavel em redes maiores
- Encontrar o rank CP otimo e NP-hard

### 6.3 Tensor Train (TT) Decomposition

**Referencia:** "TensorGPT: Efficient Compression of Large Language Models based on Tensor-Train Decomposition", arXiv:2307.00526

- **Aplicacao principal:** Compressao de camadas de embedding
- Cada token embedding convertido em Matrix Product State (MPS) de dimensao menor
- Resultados: 39.38x-65.64x de compressao na camada de embedding
- Compressao total do modelo: ~46.89% menos parametros
- Desempenho comparavel ao original com ~2.0x de compressao no embedding

**TensorSLM (2025):** Mediacoes em Raspberry Pi 5 mostram reducao de 50% no custo energetico de inferencia com degradacao negligivel.

**Hardware:** Implementacao de TT em FPGA para ChatGLM3-6B e LLaMA2-7B.

### 6.4 Decomposicao de Kronecker

**Referencia:** "Krony-PT: GPT2 compressed with Kronecker Products", arXiv:2412.12351

- Comprime GPT-2 (124M) para 80-96M parametros
- Modelo de 81M supera DistilGPT2 em predicao de proximo token
- **KronA:** Adaptador baseado em produto de Kronecker, atualiza ~0.07% dos parametros
- **Kron-LoRA (2025):** Adaptador hibrido que combina LoRA com estrutura de Kronecker

### 6.5 Comparacao dos Metodos Tensoriais

| Metodo | Alvo Principal | Compressao Tipica | Estabilidade | Hardware-Friendly |
|--------|---------------|-------------------|-------------|-------------------|
| SVD/Low-Rank | Camadas lineares | 2-5x | Alta | Sim |
| Tucker | Tensores 3D/4D | 2-3x | Moderada | Moderada |
| CP | Tensores gerais | Alta | **Baixa** | Dificil |
| Tensor Train | Embeddings, FFN | 3-65x (embed) | Alta | FPGA possivel |
| Kronecker | FFN, adaptadores | 1.3-1.6x | Alta | Sim |

---

## 7. A Questao Central: SVD vs Quantizacao ao Mesmo Tamanho

### 7.1 Analise Teorica: Bits por Parametro

Para uma matriz W de dimensao d x d (ex: 4096 x 4096 = 16M parametros):

**Quantizacao direta Q4 (4 bits):**
- Tamanho: 16M * 4 bits = 64M bits = 8 MB

**SVD rank-r com fatores em Q2 (2 bits):**
- Matrizes U (d x r) e V (r x d), valores singulares S (r)
- Parametros: 2 * d * r + r = r * (2d + 1) ≈ 2*d*r
- Tamanho em Q2: 2 * d * r * 2 bits = 4*d*r bits
- Para igualar Q4: 4*d*r = 16M * 4 → r = 16M/(d) = 4096
- **Rank necessario para igualar Q4 com fatores Q2: r = d = 4096** (ou seja, sem compressao!)

Isso mostra que SVD rank-r com fatores Q2 nunca pode igualar Q4 em tamanho para a mesma matriz, porque SVD com rank completo em Q2 equivale exatamente a Q4 direto em termos de bits totais (com overhead da decomposicao).

**Reformulando a comparacao corretamente:**

Para 8 MB (tamanho Q4 de uma matriz 4096x4096):
- Q4 direto: rank completo (4096), 4 bits/param
- SVD rank-512, fatores FP16: 512 * 2 * 4096 * 16 bits = 67M bits = 8.4 MB (ligeiramente maior)
- SVD rank-256, fatores FP16: 4.2 MB (metade do Q4)
- SVD rank-256, fatores Q4: 2.1 MB (1/4 do Q4)
- SVD rank-1024, fatores Q2: 1024 * 2 * 4096 * 2 bits = 16.8M bits ≈ 2.1 MB

### 7.2 Evidencia Empirica: O Ponto de Cruzamento

**Regime onde SVD puro perde para quantizacao:**
- Compressao moderada (3-4 bits/param): Quantizacao pura (GPTQ, AWQ) e superior
- SVD-LLM a 20% compressao (~12.8 bits/param) alcanca perplexidade 7.12-7.73 no LLaMA-7B
- GPTQ 4-bit alcanca ~6.90 perplexidade com ~4 bits/param
- **Em bits/param equivalentes, quantizacao vence SVD puro neste regime**

**Regime onde SVD + quantizacao vence:**
- Sub-2.5 bits por parametro: CALDERA (Q + LR) supera todas as tecnicas post-training
- SVD-LLM + 2-bit quant: 9.83 perplexidade vs OneBit: 10.20 (regime ~1.3 GB, ~1.5 bits/param)
- LQ-LoRA 2.75-bit: 5.67 perplexidade no LLaMA-2-7B (vs full 5.47)

**Regime de alta compressao (>40%):**
- SVD-LLM V2 mantem funcionalidade onde quantizacao colapsa
- ZS-SVD a 60% compressao: 6.96 perplexidade (com remapeamento)
- Quantizacao abaixo de 2 bits tipicamente colapsa sem tecnicas sofisticadas

### 7.3 Vantagens Estruturais do SVD sobre Quantizacao

1. **Hardware-agnostico:** Requer apenas algebra linear padrao, sem kernels especializados
2. **Flexibilidade continua:** Compressao pode ser ajustada para qualquer razao alvo
3. **Compatibilidade:** Fatores SVD podem ser adicionalmente quantizados
4. **Reducao de KV cache:** Compressao SVD pode reduzir dimensoes de chave/valor
5. **Speedup real:** ZS-SVD alcanca 5.86x speedup vs 1-2x tipico de quantizacao

### 7.4 Desvantagens do SVD

1. **Overhead computacional:** Duas multiplicacoes matriciais por camada (a menos que seja rank muito baixo)
2. **Memoria de ativacoes:** FlashSVD resolve parcialmente, mas adiciona complexidade
3. **Perda mais rapida:** Em compressao moderada, perde mais qualidade que quantizacao por bit
4. **Valores singulares pequenos importam:** Truncamento simples pode remover informacao critica

### 7.5 Conclusao da Analise Comparativa

**O consenso emergente na literatura (2024-2026) e:**

1. **Quantizacao pura e superior no regime 3-8 bits/param** para compressao moderada
2. **SVD + quantizacao e superior no regime sub-2.5 bits/param** (CALDERA, LQ-LoRA)
3. **SVD e a tecnica de compressao mais flexivel** - pode ser combinada com quantizacao, pruning e distilacao
4. **O futuro e hibrido:** SPQ (SVD + Pruning + Quantizacao) alcanca 75% reducao de memoria mantendo/melhorando perplexidade
5. **Alocacao adaptativa de rank por camada e essencial** - compressao uniforme e muito sub-otima

---

## 8. Tabela Resumo: Metodos e Resultados

| Metodo | Ano | Tipo | LLaMA-7B PPL (WikiText-2) | Compressao | Notas |
|--------|-----|------|---------------------------|-----------|-------|
| ASVD | 2023 | SVD | 6.80 @15%, 11.14 @20% | 10-30% | Training-free, activation-aware |
| SVD-LLM | 2024/ICLR25 | SVD | 7.73 @20%, 9.27 @40% | 20-80% | Truncation-aware, data whitening |
| SVD-LLM V2 | 2025/NAACL25 | SVD | 7.12 @20% | 20-40% | Dynamic ratio allocation |
| ZS-SVD | 2026 | SVD | 8.2 @30%, 6.96 @60%* | 30-60% | Zero-sum rule, *c/ remap |
| ARA | 2025 | SVD | 6.42 @80%** | 20-80% | Adaptive rank, **LLaMA2-7B |
| LQ-LoRA | 2023 | SVD+Quant | 5.67 @2.75-bit | Sub-3 bit | Low-rank + quantized |
| CALDERA | 2024/NeurIPS24 | SVD+Quant | best sub-2.5 bit | Sub-2.5 bit | W≈Q+LR, tudo quantizado |
| SPQ | 2025 | SVD+Prune+Quant | 4.91 (melhor que original!) | 75% mem | Ensemble de 3 tecnicas |
| GPTQ | 2022 | Quantizacao | ~6.90 @4-bit | 4 bit/param | Baseline de quantizacao |

---

## 9. Direcoes Futuras e Lacunas na Pesquisa

1. **Quantizacao agressiva de fatores SVD:** CALDERA mostrou que e possivel, mas poucos trabalhos exploram INT2 para fatores U e V sistematicamente

2. **Compressao adaptativa por token/batch:** Ranks dinamicos que se adaptam ao conteudo da entrada

3. **Tensor decompositions para blocos de atencao completos:** Tucker/TT decomposition para tensores 4D QKV combinados e pouco explorado

4. **Teoria sobre ponto de cruzamento SVD vs quantizacao:** Falta um framework teorico unificado para prever quando low-rank supera quantizacao

5. **SVD para modelos de difusao e multimodais:** SVDQuant abriu caminho, mas ha muito espaco para exploracao

6. **Treinamento nativo em formato fatorizado:** Em vez de comprimir pos-treinamento, treinar diretamente em representacao low-rank

7. **Combinacao com DCT/transformadas no dominio da frequencia:** Possivel sinergia entre compressao espectral (SVD) e compressao no dominio da frequencia (DCT/DFT)

---

## 10. Fontes Principais

### Papers Fundamentais
- [ASVD](https://arxiv.org/abs/2312.05821) - Yuan et al., 2023
- [SVD-LLM](https://arxiv.org/abs/2403.07378) - Wang et al., ICLR 2025
- [SVD-LLM V2](https://arxiv.org/abs/2503.12340) - NAACL 2025
- [LoRA](https://arxiv.org/abs/2106.09685) - Hu et al., ICLR 2022
- [DoRA](https://arxiv.org/abs/2402.09353) - Liu et al., ICML 2024
- [LQ-LoRA](https://arxiv.org/abs/2311.12023) - Guo & Greengard, ICLR 2024
- [CALDERA](https://arxiv.org/abs/2405.18886) - NeurIPS 2024
- [SVDQuant](https://arxiv.org/abs/2411.05007) - ICLR 2025 Spotlight
- [Small Singular Values Matter](https://arxiv.org/abs/2410.17770) - NeurIPS 2025
- [LASER](https://pratyushasharma.github.io/laser/) - ICLR 2024
- [Intrinsic Dimensionality](https://arxiv.org/abs/2012.13255) - Aghajanyan et al., 2021
- [Weight Decay Low-Rank](https://arxiv.org/abs/2410.23819) - 2024
- [Transformers Gradual Rank Increase](https://proceedings.neurips.cc/paper_files/paper/2023/file/4d69c1c057a8bd570ba4a7b71aae8331-Paper-Conference.pdf) - NeurIPS 2023

### Trabalhos Recentes (2025-2026)
- [Zero Sum SVD](https://arxiv.org/abs/2602.02848) - Fevereiro 2026
- [SPQ](https://arxiv.org/abs/2602.18420) - Fevereiro 2026
- [ARA](https://arxiv.org/abs/2510.19389) - Outubro 2025
- [NSVD](https://arxiv.org/abs/2503.17101) - Marco 2025
- [ERC-SVD](https://arxiv.org/abs/2505.20112) - Maio 2025
- [FlashSVD](https://arxiv.org/abs/2508.01506) - Agosto 2025
- [FLAR-SVD](https://openaccess.thecvf.com/content/CVPR2025W/MAI/papers/Thoma_FLAR-SVD_Fast_and_Latency-Aware_Singular_Value_Decomposition_for_Model_Compression_CVPRW_2025_paper.pdf) - CVPR 2025 Workshop

### Tensor Decompositions
- [TensorGPT](https://arxiv.org/abs/2307.00526) - 2023
- [TensorSLM](https://arxiv.org/abs/2506.13514) - 2025
- [Krony-PT](https://arxiv.org/abs/2412.12351) - 2024
- [KronA](https://arxiv.org/abs/2212.10650) - 2022
- [Kron-LoRA](https://arxiv.org/abs/2508.01961) - 2025
- [Tensor Layer Compression](https://github.com/YangletLiu/Tensor_Layer_for_Deep_Neural_Network_Compression) - Survey

### Blogs e Analises
- [SVD of Transformer Weights](https://www.alignmentforum.org/posts/mkbGjzxD8d8XqKHzA/the-singular-value-decompositions-of-transformer-weight) - Alignment Forum
- [LASER SVD Evaluation](https://huggingface.co/blog/fractalego/mistral-laser-svd) - HuggingFace Blog
- [LLM Compression Survey](https://github.com/HuangOwen/Awesome-LLM-Compression) - GitHub
