# Pesquisa: Panorama Competitivo -- Compressao por Entropia em Arquivos GGUF

**Data:** 2026-03-28
**Escopo:** Levantamento do estado da arte em compressao de arquivos GGUF, codificacao por entropia aplicada a pesos quantizados, e projetos concorrentes ou complementares no ecossistema llama.cpp / Hugging Face.

---

## Sumario Executivo

A pesquisa revela que **ninguem esta fazendo exatamente o que este projeto propoe** -- adicionar codificacao por entropia (rANS/Huffman) como camada de compressao lossless sobre pesos GGUF quantizados. O cenario competitivo consiste em abordagens tangenciais: ZipNN (compressao de expoentes flutuantes, **nao funciona em GGUF**), ZipLLM (deduplicacao entre modelos), compressed-tensors (formato para safetensors, nao GGUF), e os trellis quants do ik_llama.cpp (quantizacao alternativa, **sem codificacao por entropia**). O formato GGUF v3 atual nao possui nenhum mecanismo nativo de compressao. Ha uma janela de oportunidade clara e significativa.

---

## 1. Issues e Discussoes no Repositorio ggml-org/llama.cpp

### 1.1 Discussion #8731 -- "Uncompressing Blocks when required to save VRAM"

**Status:** Aberta, 0 comentarios (sem resposta da comunidade)

Um usuario propoe comprimir blocos GGUF com LZ4 HC -9 para reduzir uso de VRAM, descomprimindo blocos individuais sob demanda durante inferencia. Resultados relatados:

- Arquivos safetensors comprimem 20-30% com LZ4
- Primeiras 10 camadas: ~20% de compressao
- Camadas finais: compressao crescente
- Recomendacao: LZ4 HC -9 (lento para comprimir, muito rapido para descomprimir)

**Analise:** Proposta focada em economia de VRAM, nao em tamanho de arquivo para download. Sem resposta da comunidade indica baixo interesse ou falta de viabilidade percebida. A proposta nao menciona codificacao por entropia -- apenas compressao generica LZ4.

**Fonte:** [github.com/ggml-org/llama.cpp/discussions/8731](https://github.com/ggml-org/llama.cpp/discussions/8731)

### 1.2 Discussion #5063 -- "Even more quantization types?"

Uma discussao extensa sobre novos tipos de quantizacao, com uma mencao direta a codificacao por entropia:

> "What if we had...entropy coding instead (or even on top of that), because we know that not all weights are likely to be outliers"

A sugestao foi apresentada como esboço abstrato, com o autor reconhecendo incerteza sobre implementacao pratica em shaders. **Nao houve follow-up tecnico concreto.**

Outras propostas discutidas:
- Quantizacao row-wise (economia de ~1.5% no tamanho)
- Quantizacao nao-linear com polinomio de 3a ordem (~10% menor que Q4_K)
- Clustering K-means para quantizacao "verdadeira" de N bits
- VPTQ (Vector Post-Training Quantization)

**Fonte:** [github.com/ggml-org/llama.cpp/discussions/5063](https://github.com/ggml-org/llama.cpp/discussions/5063)

### 1.3 Discussion #10125 -- "QTIP: Quantization with Trellises and Incoherence Processing"

Proposta de integrar o algoritmo QTIP ao llama.cpp. QTIP substitui o quantizador vetorial do QuIP# por um quantizador trellis. O autor nota que a integracao seria simples, ja que o quantizador vetorial do llama.cpp e baseado no E8P do QuIP#.

**Status ate 2026:** Interesse da comunidade, mas **sem integracao formal** no repositorio principal do llama.cpp. Modelos QTIP quantizados existem no Hugging Face, mas requerem framework separado.

**Fonte:** [github.com/ggml-org/llama.cpp/discussions/10125](https://github.com/ggml-org/llama.cpp/discussions/10125)

### 1.4 Conclusao sobre o repositorio llama.cpp

**Nao existe issue ou PR dedicado a adicionar codificacao por entropia ao formato GGUF.** As discussoes existentes sao tangenciais (compressao LZ4 para VRAM, novos tipos de quantizacao, trellis quants). A ideia de entropy coding foi mencionada uma unica vez, de forma abstrata, sem follow-up.

---

## 2. Medidas de Entropia de Pesos GGUF Quantizados

### 2.1 Dados Existentes (de nossa pesquisa anterior -- doc 06)

Conforme documentado no arquivo `06_entropy_coding_research.md`, pesos quantizados em 4 bits possuem entropia de Shannon real de **1.12-2.17 bits**, desperdicando 45-72% do espaco de armazenamento. Essa e a lacuna de entropia (entropy gap) que nosso projeto explora.

### 2.2 Evidencias Indiretas da Literatura

- **ZipNN** reporta que modelos GGUF "nao comprimem de forma alguma" com sua abordagem baseada em separacao de expoentes e Huffman coding. Isto ocorre porque GGUF armazena inteiros quantizados (nao floats), eliminando a redundancia nos expoentes que ZipNN explora.
- **LZ4** consegue 20-30% em safetensors (FP16/BF16), mas algoritmos Lempel-Ziv buscam repeticoes multi-byte e sao ineficientes para dados quantizados.
- **zstd** atinge ~33% de reducao em modelos genericos segundo o paper do ZipLLM, mas nao ha benchmark publicado especifico para arquivos GGUF quantizados em baixa precisao.

### 2.3 Nenhuma Medicao Publica de Entropia de GGUF

**Nao foi encontrada nenhuma publicacao ou experimento que tenha medido a entropia de Shannon dos indices quantizados dentro de blocos GGUF.** Isso confirma que nosso projeto esta explorando territorio virgem. A tabela de Artefact2 (gist do GitHub) documenta KL-divergence e bpw para cada tipo de quantizacao, mas nao mede entropia real dos valores armazenados.

**Tabela de referencia (Mistral-7B):**

| Formato  | BPW  | KL-Divergence |
|----------|------|---------------|
| IQ1_S    | 1.78 | 0.5495        |
| IQ2_XXS  | 2.20 | 0.1751        |
| IQ2_XS   | 2.43 | 0.1146        |
| IQ3_XXS  | 3.21 | 0.0330        |
| IQ4_XS   | 4.32 | 0.0088        |
| Q4_K_M   | 4.83 | 0.0075        |
| Q5_K_M   | 5.67 | 0.0043        |
| Q6_K     | 6.57 | 0.0032        |
| Q8_0     | 8.50 | ~0            |

**Fonte:** [gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9)

---

## 3. Propostas para Compressao Lossless de Arquivos GGUF

### 3.1 Nenhuma Proposta Formal Existe

Nao foi encontrada nenhuma proposta formal (RFC, issue, PR, paper) para adicionar compressao lossless nativa ao formato GGUF. A unica discussao relevante (#8731) propoe LZ4 para economia de VRAM em runtime, nao para reducao de tamanho de arquivo.

### 3.2 Formato GGUF v3 -- Sem Suporte a Compressao

O formato GGUF evoluiu por tres versoes:
- **v1:** Estrutura inicial
- **v2:** Uso de uint64 para campos, padding para mmap
- **v3 (atual):** Suporte a big-endian

Nenhuma versao inclui mecanismo de compressao. O formato e projetado para mmap direto, onde os tensores sao lidos diretamente do disco para a memoria sem transformacao. Adicionar compressao quebraria esse paradigma, a menos que a descompressao ocorra durante o carregamento.

**Fonte:** [github.com/ggml-org/ggml/blob/master/docs/gguf.md](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

---

## 4. QTIP / Trellis Quantization e a Relacao com Entropy Coding

### 4.1 QTIP (Cornell RelaxML)

QTIP usa "Trellis Coded Quantization" (TCQ) com processamento de incoerencia. TCQ e **diferente** de entropy coding: e uma tecnica de quantizacao que usa um grafo trellis para encontrar o caminho otimo de quantizacao via programacao dinamica. Nao aplica compressao lossless sobre os valores quantizados.

**Caracteristicas:**
- Transforma hadamard para tornar pesos ~i.i.d. gaussianos
- Quantizacao via trellis de bitshift (hardware-efficient)
- Codigos gaussianos aleatorios para codebook
- Paper: NeurIPS 2024 Spotlight

**Status no llama.cpp:** Nao integrado. Apenas discutido.

**Fonte:** [arxiv.org/abs/2406.11235](https://arxiv.org/abs/2406.11235)

### 4.2 EXL3 (ExLlamaV3 / turboderp)

EXL3 implementa QTIP com modificacoes. Usa trellis encoding como algoritmo core de quantizacao, mapeando 256 valores continuos (tile 16x16 para Tensor Cores) para indices de K bits. **Nao utiliza entropy coding (Huffman, ANS, aritmetica).** O armazenamento e em safetensors, nao GGUF.

**Observacao do autor do EXL3:** "GGUF i-quants are abundant, and it's worth noting that they hold up well in comparison to SOTA formats."

**Fonte:** [github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md](https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md)

### 4.3 ik_llama.cpp (fork de ikawrakow)

Fork do llama.cpp com tipos de quantizacao adicionais:

**Trellis Quants (IQ1_KT - IQ4_KT):**
- Baseados em trellis de inteiros (nao requer lookup tables)
- IQ2_KT: 2.125 bpw, menor perplexidade que IQ2_KS (2.1875 bpw)
- Quantizacao ~5x mais lenta que IQK quants
- Implementacoes: CUDA, ARM NEON, AVX2, Zen4, Metal

**IQK Quants (IQ2_K - IQ6_K):**
- Familia estendida com variantes _R4/_R8/_R16 (row-interleaved packing)
- Otimizacoes SIMD para processamento de multiplas linhas simultaneamente

**Codificacao por entropia: NAO UTILIZADA.** A documentacao nao menciona nenhuma forma de entropy coding. O foco e em quantizacao e otimizacao de kernels.

**Fonte:** [github.com/ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)

---

## 5. ZipNN Aplicado a Modelos GGUF

### 5.1 Resultados: GGUF Nao Comprime com ZipNN

O paper do ZipNN (Nov 2024) reporta explicitamente: **"other quantized models using GGUF do not compress at all."**

**Por que?** ZipNN funciona separando expoentes de numeros flutuantes dos demais bits, e usando Huffman coding nos expoentes (onde dos 256 valores possiveis, os mesmos 12 aparecem 99.9% das vezes). Pesos GGUF sao inteiros quantizados empacotados, nao floats -- nao possuem a estrutura de expoente/mantissa que ZipNN explora.

**Contraste com outros formatos:**
- BF16: comprime para ~66% (33% de economia)
- FP32: comprime para ~42-84% dependendo do modelo
- GPTQ/AWQ: comprime para 85-91%
- **GGUF: 0% de compressao**

**Velocidades:** Descompressao ate 80 GB/s, compressao ate 13 GB/s (multi-thread)

**Fonte:** [arxiv.org/abs/2411.05239](https://arxiv.org/abs/2411.05239)

### 5.2 Implicacao para Nosso Projeto

O fracasso do ZipNN em GGUF **valida nossa hipotese**: compressao generica (LZ/Huffman de bytes) nao funciona em dados quantizados. E necessaria uma abordagem que entenda a estrutura dos blocos GGUF -- codificacao por entropia dos indices quantizados dentro de cada bloco, usando o conhecimento da distribuicao desses indices. Nosso projeto, que mede a entropia real dos indices e aplica rANS/Huffman sobre a distribuicao empirica, ataca exatamente a lacuna que ZipNN nao consegue atingir.

---

## 6. Compressao de GGUF com zstd/LZ4 -- Eficacia

### 6.1 zstd em Modelos Genericos

Segundo o paper do ZipLLM (NSDI 2026), zstd sozinho atinge ~33% de reducao em modelos genericos. Porem, essa medida inclui modelos BF16/FP32, que possuem redundancia significativa nos expoentes. Para modelos GGUF quantizados, espera-se eficacia muito menor.

### 6.2 LZ4 em Safetensors (evidencia indireta)

A Discussion #8731 do llama.cpp reporta 20-30% de compressao em safetensors (provavelmente FP16/BF16), com variacao por camada:
- Primeiras camadas: ~20%
- Camadas finais: compressao crescente

### 6.3 Explicacao Teorica

Algoritmos LZ (LZ4, zstd, gzip) buscam repeticoes de sequencias de bytes. Pesos quantizados GGUF possuem:
- Indices distribuidos de forma quase-uniforme dentro de cada bloco
- Pouca repeticao de sequencias multi-byte
- Alguma correlacao entre blocos adjacentes (mesma camada)

Resultado: compressao LZ e minima ou nula. A redundancia existe na **distribuicao de probabilidade** dos indices (entropia < log2(N)), nao em repeticoes de padroes. Apenas codificacao por entropia (Huffman, ANS, aritmetico) pode explorar essa redundancia.

### 6.4 Resumo

| Metodo     | Alvo          | Resultado em GGUF   | Resultado em BF16/FP32 |
|------------|---------------|----------------------|------------------------|
| ZipNN      | Expoentes     | ~0% compressao       | 33-58% economia        |
| zstd       | Repeticoes LZ | ~0-5% (estimado)     | ~33% economia          |
| LZ4        | Repeticoes LZ | ~0-5% (estimado)     | 20-30% economia        |
| rANS/Huffman (nosso) | Distribuicao de indices | **30-70% estimado** | N/A |

---

## 7. Como o Hugging Face Lida com Compressao de Modelos

### 7.1 Situacao Atual (Marco 2026)

O Hugging Face **nao aplica compressao nos arquivos durante download**. Modelos sao baixados como arquivos brutos (safetensors ou GGUF) via CDN CloudFront da AWS. Nao ha gzip/brotli na camada de transporte para arquivos grandes.

### 7.2 Sistema Xet (Novo Backend de Storage)

Desde maio 2025, o Hugging Face adotou o Xet como backend de storage padrao:
- **Deduplicacao em nivel de chunk** (~64 KiB por chunk)
- Economia de ~53% em storage via deduplicacao
- Compressao adicional de ~10%
- Upload/download otimizado com concorrencia adaptativa (ate 64 streams)

O Xet foca em **deduplicacao** (mesmo conteudo entre versoes/modelos), nao em compressao dos dados em si.

**Fonte:** [huggingface.co/blog/from-files-to-chunks](https://huggingface.co/blog/from-files-to-chunks)

### 7.3 ZipNN no Hugging Face (RFC #34737)

ZipNN propoe integrar compressao lossless ao pipeline de download:
- Arquivos `.znn` descomprimidos on-the-fly durante `load_state_dict()`
- BF16: compressao para ~66% (34% economia)
- FP32: compressao para ~42-84%
- Velocidade: 2.48 GB/s descompressao (vs 1.02 GB/s do zstd)
- **GGUF nao e discutido no RFC**

**Status:** Aguardando adocao da comunidade. Sem timeline de integracao.

**Fonte:** [github.com/huggingface/transformers/issues/34737](https://github.com/huggingface/transformers/issues/34737)

### 7.4 compressed-tensors (vLLM / Neural Magic)

Extensao do safetensors para armazenar tensores quantizados/esparsos de forma compacta. Suporta GPTQ, AWQ, SmoothQuant, INT8, FP8, SparseGPT. **Nao se aplica a GGUF.**

**Fonte:** [github.com/vllm-project/compressed-tensors](https://github.com/vllm-project/compressed-tensors)

### 7.5 ZipLLM (NSDI 2026)

Sistema de armazenamento que combina deduplicacao tensor-level com compressao BitX (XOR entre modelo fine-tuned e base, seguido de compressao generica). Reduz storage em 54.1% para colecoes de modelos. Reconhece que repositorios GGUF frequentemente contem multiplos arquivos derivados do mesmo modelo base.

**Fonte:** [arxiv.org/abs/2505.06252](https://arxiv.org/abs/2505.06252)

---

## 8. GGUF vs Safetensors -- Comparacao de Tamanhos

### 8.1 Mesmo Modelo, Mesma Precisao

Em FP16, GGUF e safetensors sao **essencialmente identicos** em tamanho. Exemplo (Flux.1-dev):
- F16.gguf: ~22.2 GB
- .safetensors: ~22.1 GB

A diferenca real surge com quantizacao GGUF:

### 8.2 Exemplo: Llama 3.2 3B

| Formato       | Tamanho | Reducao vs BF16 |
|---------------|---------|------------------|
| BF16 (safetensors) | 6.00 GB | baseline     |
| Q8_0 GGUF     | ~3.2 GB  | ~47%             |
| Q4_K_M GGUF   | 1.88 GB  | 69%              |
| IQ2_XS GGUF   | ~0.8 GB  | ~87%             |

### 8.3 Exemplo: Llama 2 13B

| Formato       | Tamanho | Reducao vs FP16 |
|---------------|---------|------------------|
| FP16 (safetensors) | 26 GB | baseline     |
| Q4_K_M GGUF   | 7.9 GB   | 70%              |

### 8.4 Implicacao

Modelos GGUF quantizados ja sao dramaticamente menores que safetensors. O ganho adicional da codificacao por entropia seria sobre o tamanho ja reduzido. Para Q4_K_M (4.83 bpw), se a entropia real for ~2.5 bits, o ganho adicional seria ~48% -- transformando um modelo de 7.9 GB em ~4.1 GB.

---

## 9. "GGUF Comprimido" ou "GGUF v4" -- Padrao Existente?

### 9.1 Resposta: Nao Existe

**Nao existe nenhum padrao, proposta formal ou implementacao de "GGUF comprimido" com codificacao por entropia.**

O formato GGUF v3 e o atual. Suas tres versoes focaram em:
- v1: Estrutura basica (substituindo GGML/GGMF/GGJT)
- v2: uint64 para campos, padding para mmap
- v3: Suporte big-endian

Nenhuma menciona compressao. O design fundamental do GGUF privilegia mmap direto, onde tensores sao lidos do disco sem transformacao.

### 9.2 BitNet como Precedente Parcial

O BitNet b1.58 (Microsoft) e o unico caso onde o GGUF armazena dados com empacotamento nao-trivial: 4 valores ternarios {-1, 0, +1} empacotados em um unico int8 (2 bits por peso, com overhead). Isso demonstra que o ecossistema GGUF pode acomodar esquemas de armazenamento mais sofisticados, embora nao seja entropy coding propriamente dito.

**Fonte:** [huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf)

---

## 10. Mapa Competitivo -- Quem Faz O Que

| Projeto/Metodo | O que faz | Funciona em GGUF? | Usa Entropy Coding? | Status |
|----------------|-----------|--------------------|-----------------------|--------|
| **ZipNN** | Compressao de expoentes via Huffman | **NAO** (0% compressao) | Sim (expoentes apenas) | Ativo, RFC no HF |
| **ZipLLM** | Deduplicacao + BitX delta | Parcial (dedup entre variantes) | Nao | Paper NSDI 2026 |
| **compressed-tensors** | Formato unificado para quants | **NAO** (safetensors apenas) | Nao | Ativo, integrado vLLM |
| **QTIP** | Trellis coded quantization | **NAO** (formato proprio) | Nao | Paper NeurIPS 2024 |
| **EXL3** | QTIP modificado, GPU-optimized | **NAO** (safetensors) | Nao | Ativo, ExLlamaV3 |
| **ik_llama.cpp** | IQK + Trellis quants | Sim (fork GGUF) | **NAO** | Ativo, fork |
| **Xet (HuggingFace)** | Deduplicacao chunk-level | Sim (como formato generico) | Nao | Producao desde 2025 |
| **Unsloth Dynamic** | Quantizacao dinamica por camada | Sim (GGUF) | Nao | Ativo |
| **Nosso projeto (DCT-Q)** | Entropy coding de indices GGUF | **SIM** | **SIM (rANS/Huffman)** | Em desenvolvimento |

---

## 11. Analise de Oportunidade

### 11.1 Lacuna Clara no Mercado

Nenhum projeto existente aplica codificacao por entropia sobre os indices quantizados de blocos GGUF. Todos os concorrentes atacam problemas diferentes:
- ZipNN: floats, nao inteiros quantizados
- ZipLLM: colecoes de modelos, nao arquivo individual
- compressed-tensors: ecossistema safetensors/vLLM
- QTIP/EXL3: quantizacao alternativa, formato diferente
- ik_llama.cpp: quantizacao melhorada, sem compressao extra

### 11.2 Validacao pela Falha de Outros

O fracasso do ZipNN em comprimir GGUF demonstra que abordagens genericas nao funcionam. Isso **valida** a necessidade de uma solucao especializada que entenda a estrutura interna dos blocos GGUF e a distribuicao dos indices quantizados.

### 11.3 Demanda Existente

- Discussion #8731 mostra interesse em compressao para economia de VRAM
- Discussion #5063 menciona entropy coding como possibilidade
- Hugging Face investiu em infraestrutura (Xet, ZipNN) mostrando que tamanho de download e uma prioridade da industria
- O ecossistema GGUF e massivo: GGUF e safetensors juntos representam >90% do storage total no HF Hub

### 11.4 Barreiras de Entrada

- O design mmap do GGUF resiste a compressao (precisa descomprimir durante carregamento)
- Kernels de dequantizacao precisam ser modificados ou a descompressao precisa ser transparente
- Performance de descompressao precisa ser negligivel vs tempo de inferencia
- Integracao com llama.cpp requer conhecimento profundo do codebase C++

### 11.5 Vantagem Estrategica

Nosso projeto e o **unico** que ataca especificamente a lacuna de entropia em pesos GGUF quantizados. Se conseguirmos demonstrar:
1. Reducao de 30-50% no tamanho de arquivos GGUF sem perda
2. Descompressao rapida o suficiente para nao impactar inferencia
3. Compatibilidade com o ecossistema llama.cpp

...teremos uma contribuicao genuinamente nova e de alto impacto para o ecossistema de LLMs locais.

---

## 12. Riscos e Consideracoes

### 12.1 Risco: mmap Incompativel

GGUF e projetado para mmap, onde tensores sao lidos diretamente do disco. Compressao quebra isso. Mitigacao: descomprimir durante carregamento em buffer temporario, ou usar compressao apenas para distribuicao (nao para inferencia).

### 12.2 Risco: Ganho Marginal em Quants Baixos

IQ1_S (1.78 bpw) e IQ2_XXS (2.20 bpw) ja estao proximo do limite entropico. O ganho de entropy coding pode ser marginal nessas quantizacoes extremas. O sweet spot provavelmente esta em Q4_K_M a Q6_K, onde a lacuna de entropia e maior.

### 12.3 Risco: ik_llama.cpp Adiciona Entropy Coding

O fork ik_llama.cpp ja adiciona trellis quants; se ikawrakow decidir adicionar entropy coding como camada sobre seus quants, seria um concorrente direto com base de usuarios existente.

### 12.4 Oportunidade: Formato "GGUF-C" ou Extensao

Propor uma extensao oficial do formato GGUF com metadado de compressao (tipo de codec, parametros, offsets para random access) poderia se tornar um padrao de fato se aceito pela comunidade llama.cpp.

---

## Fontes

- [ZipNN Paper](https://arxiv.org/abs/2411.05239)
- [ZipLLM Paper (NSDI 2026)](https://arxiv.org/abs/2505.06252)
- [QTIP Paper (NeurIPS 2024)](https://arxiv.org/abs/2406.11235)
- [GGUF Format Spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [llama.cpp Discussion #8731](https://github.com/ggml-org/llama.cpp/discussions/8731)
- [llama.cpp Discussion #5063](https://github.com/ggml-org/llama.cpp/discussions/5063)
- [llama.cpp Discussion #10125](https://github.com/ggml-org/llama.cpp/discussions/10125)
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
- [ExLlamaV3 / EXL3](https://github.com/turboderp-org/exllamav3)
- [compressed-tensors](https://github.com/vllm-project/compressed-tensors)
- [ZipNN RFC no HuggingFace](https://github.com/huggingface/transformers/issues/34737)
- [HuggingFace Xet Blog](https://huggingface.co/blog/from-files-to-chunks)
- [HuggingFace Rearchitecting Uploads](https://huggingface.co/blog/rearchitecting-uploads-and-downloads)
- [GGUF Quantizations Overview (Artefact2)](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9)
- [BitNet b1.58 2B4T GGUF](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf)
- [Unified Evaluation of llama.cpp Quantization](https://arxiv.org/abs/2601.14277)
