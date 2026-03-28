# Pesquisa: Delta Coding e Compressao Inter-Camadas para Transformers

**Data:** 2026-03-28
**Objetivo:** Mapear tecnicas de codificacao delta de video codecs para compressao de camadas de transformers, identificando trabalhos existentes e lacunas para a abordagem DCT.

---

## 1. Codificacao Inter-Frame em Video Codecs: Tecnicas Transferiveis

### 1.1 Pipeline Fundamental: Motion + Residual + DCT

Os codecs de video (H.264/AVC, H.265/HEVC, AV1) compartilham um pipeline central que e diretamente analogico ao que propomos:

1. **Estimacao de Movimento (Motion Estimation):** Encontrar o bloco no frame de referencia mais similar ao bloco atual. Resultado: vetor de movimento.
2. **Compensacao de Movimento (Motion Compensation):** Usar o vetor de movimento para gerar uma predicao do bloco atual.
3. **Calculo do Residual:** Diferenca entre o bloco real e o predito (delta).
4. **Transformada (DCT):** Aplicar DCT ao residual para decorrelacionar os dados.
5. **Quantizacao:** Quantizar os coeficientes DCT (perdendo informacao de alta frequencia).
6. **Codificacao Entropica:** Comprimir os coeficientes quantizados (Huffman, CABAC, etc).

**Fontes:**
- [Inter frame - Wikipedia](https://en.wikipedia.org/wiki/Inter_frame)
- [H.264/AVC Inter Prediction - Vcodex](https://www.vcodex.com/h264avc-inter-prediction)
- [H265 Technical Overview](https://sonnati.wordpress.com/2014/06/20/h265-part-i-technical-overview/)

### 1.2 Tipos de Frames e Predicao

| Tipo | Descricao | Analogia em Transformers |
|------|-----------|--------------------------|
| **I-Frame** | Codificado independentemente (intra-frame) | Camada 0 armazenada integralmente |
| **P-Frame** | Predito a partir de um frame anterior | Camada n predita a partir de n-1 |
| **B-Frame** | Predito a partir de frames anteriores E posteriores | Camada n predita a partir de n-1 e n+1 (bidirecional) |

**Fonte:** [Video compression picture types - Wikipedia](https://en.wikipedia.org/wiki/Video_compression_picture_types)

### 1.3 Inovacoes do AV1 Transferiveis

O codec AV1 introduziu tecnicas avancadas particularmente relevantes:

- **Compensacao de Movimento Afim:** Modela movimentos complexos (rotacao, zoom, pan) com transformacoes afins -- diretamente aplicavel a "movimentos de pesos" entre camadas.
- **Predicao Composta (Compound Prediction):** Combina predicoes de duas referencias com mascaras de transicao suave -- analogia: combinar predicoes de camadas vizinhas.
- **OBMC (Overlapped Block Motion Compensation):** Suaviza fronteiras entre blocos -- analogia: suavizar fronteiras entre sub-blocos de pesos compartilhados.
- **56 modos de predicao angular intra** vs. 35 no HEVC, permitindo predicoes mais precisas.

**Fonte:** [AV1 Tools Overview](https://www.jmvalin.ca/papers/AV1_tools.pdf)

### 1.4 O que Traduz Diretamente para DCT em Pesos

A analogia mais forte e:

```
Video Codec:                    Nossa Abordagem (Proposta):
Frame_n                    -->  W_n (pesos da camada n)
Frame_{n+1}                -->  W_{n+1} (pesos da camada n+1)
Motion Vector              -->  Transformacao afim (escala + translacao?)
Residual = Frame - Pred    -->  Delta = W_{n+1} - f(W_n)
DCT(Residual)              -->  DCT(Delta)
Quantize(DCT coeffs)       -->  Quantize(DCT coeffs)
```

A pesquisa em codecs confirma que o pipeline **predicao + residual + DCT + quantizacao** e extremamente eficaz. O paper "Neural Network Compression using Transform Coding" (2018) ja demonstrou que DCT aplicada a pesos de redes neurais tem **impacto de reducao de entropia superior a quantizacao direta**, mesmo sem contexto espacial.

**Fonte:** [Neural Network Compression using Transform Coding and Clustering](https://arxiv.org/pdf/1805.07258)

---

## 2. Redundancia Inter-Camadas em Transformers: Estado da Arte

### 2.1 Evidencia Quantitativa de Similaridade entre Camadas

#### ShortGPT (Men et al., 2024) - "Layers in LLMs are More Redundant Than You Expect"

- **Metrica proposta:** Block Influence (BI) = 1 - E[cos(X_i, X_{i+1})], onde X_i e a representacao de entrada e X_{i+1} a de saida da camada i.
- **Descoberta principal:** Camadas mais profundas tem BI significativamente menor (mais redundantes).
- **Camadas medias/profundas sao altamente redundantes**, com transformacoes minimals nos hidden states.
- **Modelos testados:** LLaMA 2 (7B, 13B), Baichuan2 (7B, 13B), RWKV-7B.

**Fonte:** [ShortGPT - arXiv](https://arxiv.org/abs/2403.03853)

#### LaCo (Yang et al., 2024) - "Large Language Model Pruning via Layer Collapse"

- **Medida concreta:** Similaridade cosseno entre camadas adjacentes de **layers 3-28 no Llama2-7B e Baichuan2-7B e tipicamente muito proxima de 1**.
- **Apos merge de 4 camadas consecutivas (layers 10-19), a similaridade cosseno minima dos vetores 4096-dimensionais fica acima de 0.996.**
- **Resultados de poda:**
  - Llama2-7B: 27% de poda (32->23 camadas) mantem score medio de 37.46 vs 46.55 original
  - Llama2-13B: 25% de poda (40->30 camadas) mantem 47.55 vs 55.50 original
  - Estabilidade notavel na faixa de 10-25% de poda.

**Fonte:** [LaCo - arXiv](https://arxiv.org/abs/2402.11187)

#### The Unreasonable Ineffectiveness of the Deeper Layers (Gromov et al., 2024)

- **Resultado chocante:** Ate **50% das camadas** podem ser removidas do Llama-2-70B com degradacao minima.
- **Limites de poda por modelo (sem fine-tuning de recuperacao):**
  - Llama-2-70B: ~50%
  - Llama-2-7B/13B: 45-55%
  - Mistral-7B: ~35%
  - Phi-2 (2.7B): ~25%
  - Qwen: ~20%
- **Implicacao cientifica:** Metodos de pre-treinamento atuais nao utilizam adequadamente os parametros das camadas mais profundas.

**Fonte:** [arXiv:2403.17887](https://arxiv.org/abs/2403.17887)

#### Transformer Layers as Painters (Sun et al., 2024 -- AAAI 2025)

- **Classificacao em 3 tipos de camadas:**
  1. **Iniciais (2-3 camadas):** Altamente especializadas, remocao causa colapso catastrofico.
  2. **Intermediarias (maioria):** Compartilham um "espaco de representacao semantica comum" com alta similaridade cosseno mutua.
  3. **Finais (~1 camada):** Especializadas para output.
- **Camadas intermediarias podem ser reordenadas, puladas ou executadas em paralelo** com degradacao gradual ("graceful degradation").
- **Modelos:** Llama2-7B (32 camadas), Llama2-13B (40 camadas), BERT-Large (24 camadas), Pythia (14M-12B).

**Fonte:** [Transformer Layers as Painters - arXiv](https://arxiv.org/abs/2407.09298)

#### The Curse of Depth (Sun et al., 2025 -- NeurIPS 2025)

- **Causa raiz identificada:** Pre-Layer Normalization (Pre-LN) causa crescimento exponencial da variancia de output com a profundidade, fazendo com que a derivada dos blocos profundos se aproxime de uma **matriz identidade**.
- **Consequencia:** Camadas profundas mal contribuem para o treinamento.
- **Confirmado em:** Llama, Mistral, DeepSeek, Qwen.
- **Solucao proposta (LayerNorm Scaling):** Reduz perplexidade em 0.97-1.31 pontos.

**Fonte:** [The Curse of Depth - arXiv](https://arxiv.org/abs/2502.05795)

#### Sliding-Window Merging (2025)

- **Analise CKA revela "patches" de camadas consecutivas com alta correlacao** (visualizadas como regioes brilhantes em heatmaps).
- **Resultados:** 20% poda no LLaMA2-7B: 6.7B -> 5.5B parametros.
- **35% poda no Vicuna-7B:** melhoria de 1.654% sobre metodos existentes em tarefas zero-shot.

**Fonte:** [Sliding-Window Merging - arXiv](https://arxiv.org/abs/2502.19159)

#### Dynamic LLM Slicing (2024 -- EMNLP Findings)

- **Layer Redundancy (LR) Score:** LR(Li) = cos(input_i, output_i), normalizado para [0,1].
- **Quanto maior o LR, mais redundante a camada** (a camada pouco altera seu input).

**Fonte:** [Dynamic LLM Slicing - ACL Anthology](https://aclanthology.org/2024.findings-emnlp.579.pdf)

### 2.2 Numeros Concretos: Similaridade entre Camadas Adjacentes

| Medida | Valor | Modelo | Fonte |
|--------|-------|--------|-------|
| Cosseno entre camadas adjacentes (layers 3-28) | "muito proximo de 1" | Llama2-7B, Baichuan2-7B | LaCo |
| Cosseno apos merge de 4 camadas | > 0.996 | Llama2-7B | LaCo |
| Cosseno geral entre hidden states de camadas distantes | > 0.8 (minimo) | Multiplos LLMs | Multiplos estudos |
| Camadas removiveis sem colapso | 45-55% | Llama-2 family | Gromov et al. |
| Camadas removiveis sem colapso | ~50% | Llama-2-70B | Gromov et al. |
| Atencao removivel sem degradacao | 50% | Llama-2-70B | "What Matters in Transformers" |

**Conclusao:** Camadas adjacentes em LLMs tem similaridade de representacao extremamente alta (>0.99 para camadas intermediarias), o que implica que os **deltas entre pesos de camadas adjacentes devem ser altamente compressiveis**.

---

## 3. Weight Sharing e Compressao por Deltas entre Camadas

### 3.1 ALBERT (Lan et al., 2019 -- ICLR 2020)

- **Abordagem pioneira:** Compartilhamento total de parametros entre todas as camadas do encoder.
- **Resultado:** Reducao massiva de parametros (ALBERT-xxlarge supera BERT-large com menos parametros unicos), mas custo computacional permanece similar.
- **Limitacao:** Compartilhamento identico -- nao captura diferencas entre camadas.

**Fonte:** [ALBERT - ICLR 2020](https://openreview.net/pdf?id=H1eA7AEtvS)

### 3.2 ResidualTransformer (Wang et al., 2023 -- ICASSP 2024)

**Este e o trabalho mais proximo da nossa abordagem proposta.**

- **Formulacao:** W_l = W_shared + Delta_l, onde Delta_l = A_l * B_l + D_l
  - A_l, B_l: matrizes low-rank (rank R)
  - D_l: matriz diagonal
- **Pesos compartilhados + deltas low-rank por camada.**
- **Configuracoes testadas (encoder de 56.7M parametros):**
  - K=3, R=2: 34.0% dos parametros originais (WER: 13.53% vs 13.28% baseline)
  - K=3, R=16: 38.1% dos parametros
  - K=6, R=16: 21.5% dos parametros (WER: 14.12%)
  - K=9, R=16: 15.9% dos parametros (~6x reducao, WER: 14.62%)
- **Compressao de ~3x com degradacao minima.**

**Fonte:** [ResidualTransformer - arXiv](https://arxiv.org/abs/2310.02489)

### 3.3 DeltaLLM (2025)

**Extensao direta do conceito para LLMs.**

- **Formulacao:** W_{l+i} = W_l + delta_tilde_{l+i}^l, onde delta_tilde e a aproximacao low-rank do delta real.
- **Estrategias:** "sequencial" (um bloco ancora gera subsequentes) e "alternada" (blocos padrao intercalados com blocos delta).
- **Primeiras e ultimas 2 camadas preservadas intactas** (funcoes especializadas).
- **Resultados:**
  - DeltaPhi 3.35B: 12% reducao de parametros, retendo ~90% do desempenho
  - DeltaPhi 2.9B: 24% reducao, delta-layers consomem apenas 90MB
  - DeltaLlama 2.52B: 21% compressao
- **Apenas 30-40M tokens necessarios para treinamento.**

**Fonte:** [DeltaLLM - arXiv](https://arxiv.org/abs/2501.18596)

### 3.4 Basis Sharing (2024 -- ICLR 2025)

**Abordagem via SVD compartilhada entre camadas.**

- **Metodo:** Concatenar matrizes de peso de n camadas -> SVD -> bases compartilhadas + coeficientes unicos por camada.
- **Formulacao:** W = U_k * Sigma_k * V_k^T (compartilhado), C(i) (unico por camada).
- **Resultado chave:** Compartilhar bases entre camadas adjacentes tem **menor perda de Frobenius** que comprimir cada camada independentemente.
  - Exemplo: W_K compartilhada entre camadas 9-10: loss 61,817.3 vs 66,682.9 independente.
- **Desempenho (LLaMA-7B, WikiText-2):**
  - 20% compressao: PPL 7.74 (vs 7.94 SVD-LLM)
  - 30% compressao: PPL 9.25 (vs 9.56 SVD-LLM)
  - 50% compressao: PPL 19.99 (vs 23.97 SVD-LLM) -- **reducao de 16.6% na perplexidade**
- **Agrupamento ideal:** 4-5 camadas para <30% compressao; 2 camadas para >30%.

**Fonte:** [Basis Sharing - arXiv](https://arxiv.org/abs/2410.03765)

### 3.5 Relaxed Recursive Transformers (Bae et al., 2024 -- ICLR 2025, Google DeepMind)

- **Conceito:** Transformer recursivo (um bloco repetido N vezes) + LoRA por camada para diferenciacoes.
- **Compressao:** ~50% reducao de parametros (Gemma 2B -> Gemma 1B recursivo).
- **Ranks LoRA testados:** 64, 128, 256, 512.
- **Resultado notavel:** Relaxed Gemma (rank 512) atinge 58.4% vs 58.6% do modelo original -- **recuperacao quase completa com 50% dos parametros**.
- **Apenas 60B tokens de uptraining** necessarios (vs 3T tokens do modelo original).

**Fonte:** [Relaxed Recursive Transformers - arXiv](https://arxiv.org/abs/2410.20672)

### 3.6 MiniViT (Zhang et al., 2022 -- CVPR 2022)

- **Conceito:** Weight multiplexing -- pesos compartilhados entre camadas consecutivas + blocos de transformacao para aumentar diversidade.
- **Resultado:** Reducao de 48% do Swin-B com **aumento de 1.0% no Top-1 accuracy** no ImageNet.
- **Implicacao:** Pesos compartilhados + transformacao leve superam pesos independentes.

**Fonte:** [MiniViT - CVPR 2022](https://arxiv.org/abs/2204.07154)

### 3.7 Dynamic Layer Tying (2024 -- ICLR 2024)

- **Usa Reinforcement Learning para decidir dinamicamente quais camadas compartilham pesos.**
- **Reducao de consumo de memoria de ate uma ordem de magnitude durante treinamento.**

**Fonte:** [Dynamic Layer Tying - arXiv](https://arxiv.org/abs/2401.12819)

---

## 4. Delta Compression para Checkpoints e Versionamento de Modelos

### 4.1 FM-Delta (NeurIPS 2024)

- **Compressao lossless de modelos fine-tuned** via delta inteiro entre modelo fine-tuned e pre-treinado.
- **Resultado:** Reducao media de ~50% no armazenamento (GPT-NeoX-20B family: 423GB -> 205GB).
- **Insight:** A maioria dos modelos fine-tuned tem diferenca pequena do modelo pre-treinado.

**Fonte:** [FM-Delta - NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/7b75a7339dfb256ee4b4bec028a6890b-Paper-Conference.pdf)

### 4.2 Delta-CoMe (NeurIPS 2024)

- **Compressao de precisao mista para deltas de pesos.**
- **Inovacao:** Vetores singulares correspondentes a valores singulares maiores recebem mais bits; valores extremamente pequenos recebem 0 bits.
- **Motivacao:** Distribuicao long-tail dos valores singulares nos deltas.
- **Testado em:** Llama-2, Llama-3, Mistral (LLMs de math, code, chat e VLMs).

**Fonte:** [Delta-CoMe - arXiv](https://arxiv.org/abs/2406.08903)

### 4.3 Per-Axis Weight Deltas (NeurIPS 2025 Workshop)

- **Comprime checkpoints fine-tuned em deltas de 1 bit** + fatores de escala FP16 por eixo (linha/coluna).
- **Resultado:** Latencia cold-start reduzida de 2.08s para 0.80s (Llama-3.1-8B).
- **Insight chave:** O sinal (positivo/negativo) do delta carrega a maior parte da informacao.

**Fonte:** [Per-Axis Weight Deltas - arXiv](https://arxiv.org/abs/2512.19720)

### 4.4 Inshrinkerator (SoCC 2024)

- **Compressao de checkpoints de treinamento via quantizacao dinamica + delta encoding.**
- **Observacao:** A maioria dos parametros permanece no mesmo bin de quantizacao entre checkpoints consecutivos.
- **Compressao de ate 39x com impacto negligenciavel na precisao** (tolerando ate 10 restores).

**Fonte:** [Inshrinkerator - ACM SoCC 2024](https://dl.acm.org/doi/10.1145/3698038.3698553)

---

## 5. Estimacao de Movimento para "Movimento de Pesos" entre Camadas

### 5.1 Analogia Direta: AV1 Affine Motion Compensation

O AV1 implementa **compensacao de movimento afim** para modelar movimentos complexos como rotacao, zoom e pan. A transformacao afim e parametrizada por 6 valores (2D). Para matrizes de pesos de transformers, uma transformacao afim seria:

```
W_{n+1} ~= alpha * W_n + beta    (escalar: 2 parametros)
W_{n+1} ~= A * W_n + B           (matricial: mais parametros, mais expressivo)
W_{n+1} ~= A * W_n * C + B       (bilinear: maximo expressividade)
```

**Nenhum trabalho encontrado aplica explicitamente estimacao de movimento afim entre camadas de transformers para compressao de pesos.**

### 5.2 Trabalhos Relacionados que se Aproximam

- **MiniViT:** Usa "blocos de transformacao" entre camadas compartilhadas, mas nao formula como estimacao de movimento.
- **Basis Sharing:** A decomposicao SVD compartilhada implicitamente captura uma rotacao (U) e escala (Sigma) compartilhadas, com apenas os coeficientes (C) variando -- analogia parcial.
- **Spatial Transformer Networks (Jaderberg et al., 2015):** Predizem transformacoes afins para features, mas nao entre camadas de pesos.
- **FiLM (Feature-wise Linear Modulation):** Transformacoes afins condicionais (gamma * x + beta), mas aplicadas a features, nao pesos.

### 5.3 Lacuna Identificada

**Nao existe trabalho publicado que formule a relacao entre camadas adjacentes como "estimacao de movimento de pesos" usando transformacoes afins, seguido de codificacao do residual via DCT.**

Isto representa uma oportunidade de pesquisa original significativa.

---

## 6. DCT Aplicada a Pesos de Redes Neurais

### 6.1 Transform Coding para Redes Neurais

- O paper "Neural Network Compression using Transform Coding" (Wiedemann et al., 2018) demonstrou que **DCT aplicada a filtros convolucionais seguida de quantizacao supera quantizacao direta** em termos de reducao de entropia.
- Para filtros 7x7: DCT 7x7 aplicada, seguida de quantizacao uniforme de coeficientes.
- **Diferenca critica do JPEG:** Em redes neurais, todos os coeficientes sao quantizados igualmente (nao ha viés para baixas frequencias como no sistema visual humano).

**Fonte:** [Neural Network Compression using Transform Coding](https://arxiv.org/pdf/1805.07258)

### 6.2 DCT em Attention de Transformers

- **DCT-based Decorrelated Attention (2024):** Usa DCT para decorrelacionar attention scores em Vision Transformers, reduzindo parametros e computacao mantendo accuracy comparavel.

**Fonte:** [DCT Decorrelated Attention - arXiv](https://arxiv.org/html/2405.13901v3)

---

## 7. Sintese: O que Traduz para Nossa Abordagem DCT

### 7.1 Evidencias que Suportam a Viabilidade

| Evidencia | Implicacao para DCT-Delta |
|-----------|---------------------------|
| Cosseno entre camadas adjacentes ~0.996+ | Deltas sao pequenos, portanto altamente compressiveis |
| 45-55% das camadas sao removiveis | A informacao unica por camada e minima |
| Basis Sharing supera SVD independente | Ha redundancia exploravel nas bases entre camadas |
| ResidualTransformer: 3x compressao com R=16 | Deltas low-rank funcionam; DCT pode capturar mais |
| Relaxed Recursive: 50% parametros, ~100% perf | Pesos compartilhados + delta pequeno e suficiente |
| Transform coding > quantizacao direta | DCT e justificada como transformada para decorrelacao |
| AV1 affine motion compensation | Transformacao afim entre camadas e viavel |

### 7.2 Lacunas que Podemos Preencher

1. **DCT aplicada especificamente a deltas inter-camadas:** Nenhum trabalho encontrado faz isso. Os trabalhos existentes usam:
   - Low-rank (SVD/LoRA) para representar deltas -> **nao exploram dominio de frequencia**
   - Quantizacao direta dos deltas -> **menos eficiente que transform coding**

2. **Estimacao de "movimento" afim entre camadas antes do calculo do delta:** Nenhum trabalho encontrado. A cadeia completa seria:
   ```
   f(W_n) = alpha * W_n + beta   (estimacao de "movimento")
   Delta = W_{n+1} - f(W_n)       (residual apos predicao)
   DCT(Delta)                      (decorrelacao no dominio de frequencia)
   Quantize(DCT(Delta))            (compressao com perda controlada)
   ```

3. **Combinacao de predicao B-frame (bidirecional) para camadas:** Nenhum trabalho prediz uma camada a partir da anterior E posterior simultaneamente.

4. **Quantizacao adaptativa por frequencia dos coeficientes DCT dos deltas:** Enquanto Delta-CoMe usa mixed-precision nos valores singulares, ninguem aplica quantizacao diferenciada no dominio DCT dos deltas inter-camadas.

5. **Analise espectral (DCT) dos deltas entre camadas:** Nao ha publicacao que analise a distribuicao de energia no dominio de frequencia dos deltas W_{n+1} - W_n.

### 7.3 Pipeline Proposto (Baseado na Pesquisa)

```
Para cada par de camadas adjacentes (n, n+1):

1. PREDICAO (inspirada em motion estimation):
   - Estimar transformacao afim: f(W) = alpha * W + beta
   - Otimizar (alpha, beta) para minimizar ||W_{n+1} - f(W_n)||

2. RESIDUAL:
   - Delta = W_{n+1} - f(W_n)
   - Delta e menor que W_{n+1} - W_n (a transformacao afim captura escala/bias)

3. TRANSFORMADA:
   - Aplicar DCT 2D a blocos do Delta (8x8, 16x16, etc.)
   - Concentrar energia em poucos coeficientes de baixa frequencia

4. QUANTIZACAO:
   - Quantizar coeficientes DCT com mais bits para baixa frequencia
   - Usar menos bits (ou zero) para alta frequencia
   - Controle de qualidade via metricas de task performance

5. CODIFICACAO ENTROPICA:
   - Huffman/ANS nos coeficientes quantizados

6. ARMAZENAMENTO:
   - Camada 0: completa (I-frame)
   - Camadas 1-N: (alpha, beta, coeficientes DCT quantizados)
```

### 7.4 Vantagens Teoricas sobre Trabalhos Existentes

| Abordagem Existente | Limitacao | Nossa Vantagem |
|---------------------|-----------|----------------|
| ResidualTransformer (low-rank delta) | Rank fixo limita expressividade | DCT captura todas as frequencias, quantizacao adaptativa |
| DeltaLLM (low-rank delta) | Requer treinamento (30-40M tokens) | Training-free (post-hoc) |
| Basis Sharing (SVD compartilhado) | Perde informacao por truncamento de rank | DCT preserva toda informacao antes da quantizacao |
| Poda de camadas (ShortGPT, LaCo) | Remove camadas inteiras -- tudo ou nada | Granularidade fina -- comprime por frequencia |
| Quantizacao direta (GPTQ, AWQ) | Nao explora redundancia inter-camadas | Combina com delta inter-camadas |

---

## 8. Referencias Completas

### Redundancia e Similaridade entre Camadas
- Men et al. (2024). "ShortGPT: Layers in Large Language Models are More Redundant Than You Expect." [arXiv:2403.03853](https://arxiv.org/abs/2403.03853)
- Yang et al. (2024). "LaCo: Large Language Model Pruning via Layer Collapse." EMNLP Findings 2024. [arXiv:2402.11187](https://arxiv.org/abs/2402.11187)
- Gromov et al. (2024). "The Unreasonable Ineffectiveness of the Deeper Layers." [arXiv:2403.17887](https://arxiv.org/abs/2403.17887)
- Sun et al. (2024). "Transformer Layers as Painters." AAAI 2025. [arXiv:2407.09298](https://arxiv.org/abs/2407.09298)
- Sun et al. (2025). "The Curse of Depth in Large Language Models." NeurIPS 2025. [arXiv:2502.05795](https://arxiv.org/abs/2502.05795)
- (2025). "Sliding-Window Merging for Compacting Patch-Redundant Layers in LLMs." [arXiv:2502.19159](https://arxiv.org/abs/2502.19159)
- (2024). "Dynamic LLM Slicing based on Layer Redundancy." EMNLP Findings 2024. [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.579.pdf)
- (2024). "What Matters in Transformers? Not All Attention is Needed." [arXiv:2406.15786](https://arxiv.org/html/2406.15786v4)

### Weight Sharing e Delta entre Camadas
- Wang et al. (2023). "ResidualTransformer: Residual Low-Rank Learning with Weight-Sharing for Transformer Layers." ICASSP 2024. [arXiv:2310.02489](https://arxiv.org/abs/2310.02489)
- (2025). "DeltaLLM: Compress LLMs with Low-Rank Deltas between Shared Weights." [arXiv:2501.18596](https://arxiv.org/abs/2501.18596)
- (2024). "Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression." ICLR 2025. [arXiv:2410.03765](https://arxiv.org/abs/2410.03765)
- Bae et al. (2024). "Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA." ICLR 2025. [arXiv:2410.20672](https://arxiv.org/abs/2410.20672)
- Zhang et al. (2022). "MiniViT: Compressing Vision Transformers with Weight Multiplexing." CVPR 2022. [arXiv:2204.07154](https://arxiv.org/abs/2204.07154)
- (2024). "Dynamic Layer Tying for Parameter-Efficient Transformers." ICLR 2024. [arXiv:2401.12819](https://arxiv.org/abs/2401.12819)
- Lan et al. (2019). "ALBERT: A Lite BERT." ICLR 2020. [OpenReview](https://openreview.net/pdf?id=H1eA7AEtvS)

### Delta Compression para Checkpoints
- (2024). "FM-Delta: Lossless Compression for Storing Massive Fine-tuned Foundation Models." NeurIPS 2024. [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/7b75a7339dfb256ee4b4bec028a6890b-Paper-Conference.pdf)
- (2024). "Delta-CoMe: Training-Free Delta-Compression with Mixed-Precision for LLMs." NeurIPS 2024. [arXiv:2406.08903](https://arxiv.org/abs/2406.08903)
- Kuyumdzhiev (2025). "Per-Axis Weight Deltas for Frequent Model Updates." NeurIPS 2025 Workshop. [arXiv:2512.19720](https://arxiv.org/abs/2512.19720)
- (2024). "Inshrinkerator: Compressing Deep Learning Training Checkpoints via Dynamic Quantization." ACM SoCC 2024. [ACM](https://dl.acm.org/doi/10.1145/3698038.3698553)

### Video Codecs e Transform Coding
- Wiedemann et al. (2018). "Neural Network Compression using Transform Coding and Clustering." [arXiv:1805.07258](https://arxiv.org/pdf/1805.07258)
- (2024). "DCT-based Decorrelated Attention for Vision Transformers." [arXiv:2405.13901](https://arxiv.org/html/2405.13901v3)
- [AV1 Tools Overview](https://www.jmvalin.ca/papers/AV1_tools.pdf)
- [H.264/AVC Inter Prediction - Vcodex](https://www.vcodex.com/h264avc-inter-prediction)

---

## 9. Conclusao e Proximos Passos

A pesquisa revela um cenario muito favoravel para a abordagem DCT-delta:

1. **Ha evidencia massiva de redundancia inter-camadas** -- camadas adjacentes em LLMs tem similaridade >0.99 em hidden states, e ate 50% das camadas podem ser removidas.

2. **Deltas low-rank entre camadas ja funcionam** (ResidualTransformer: 3x compressao; DeltaLLM: 12-24% reducao; Relaxed Recursive: 50% reducao).

3. **Ninguem combinou DCT com deltas inter-camadas** -- esta e a lacuna principal que podemos preencher.

4. **Ninguem aplicou estimacao de "movimento" afim entre camadas** seguido de codificacao do residual -- outra contribuicao original possivel.

5. **O pipeline video-codec (predicao + residual + DCT + quantizacao) e comprovadamente eficaz** e pode ser adaptado diretamente para compressao de transformers.

**Primeiro experimento sugerido:** Calcular os deltas W_{n+1} - W_n para todas as camadas de um Llama2-7B, aplicar DCT 2D em blocos, e medir a distribuicao de energia nos coeficientes DCT. Se a energia estiver concentrada em baixas frequencias (como em video), a abordagem e viavel.
