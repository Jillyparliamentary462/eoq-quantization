# Pesquisa: Contribuindo para o llama.cpp

## Resumo Executivo

O llama.cpp e um dos projetos open source mais ativos do GitHub, com mais de 1.038
contribuidores, ~810 PRs abertos, ~9.693 PRs ja fechados, e um ritmo de releases
extremamente acelerado (ate 9 releases por dia, frequentemente com intervalos de
30-60 minutos entre eles). O projeto tem regras rigorosas de contribuicao, incluindo
uma politica **anti-codigo-gerado-por-IA** que pode resultar em banimento permanente.
Qualquer PR que adicione novos tipos de quantizacao precisa incluir benchmarks extensivos,
suporte multi-plataforma, e demonstrar valor claro sobre abordagens existentes.

---

## 1. Diretrizes de Contribuicao (CONTRIBUTING.md)

### 1.1 Niveis de Contribuidores

O projeto reconhece tres niveis:
- **Contributors**: contribuidores anteriores sem acesso especial
- **Collaborators/Triage**: contribuidores significativos que mantem areas especificas do codigo
- **Maintainers**: revisam e fazem merge de PRs

### 1.2 Politica de Uso de IA (CRITICO)

O projeto **rejeita explicitamente** PRs que sejam total ou predominantemente gerados por IA.
Ferramentas de IA so podem ser usadas em capacidade limitada e assistiva.

**Regras:**
- Codigo inicialmente gerado por IA, mesmo depois editado, ainda conta como gerado por IA
- Se usar assistencia de IA: divulgar como foi usada, revisar manualmente tudo
- Deve ser capaz de explicar cada linha quando perguntado
- NUNCA usar IA para escrever comunicacoes (relatorios de issues, descricoes de PRs, discussoes)
- Violacoes podem resultar em **banimento permanente**

**Excecoes de divulgacao:**
- Autocompletions triviais que voce ja visualizava
- Consultas de conhecimento nao relacionadas as suas modificacoes
- Pedidos de links/guias para trabalho autodirigido

### 1.3 Requisitos para Submissao de PRs

**Antes de submeter:**
- Pesquisar PRs existentes para evitar duplicacao
- Familiarizar-se com a biblioteca de tensores ggml
- Testar localmente: rodar CI completo, verificar metricas de perplexidade/performance
- Usar `test-backend-ops` para modificacoes no ggml
- Submeter PRs separados para funcionalidades distintas
- **Comecar com suporte somente CPU** (backends GPU em PRs separados)
- Novos tipos de quantizacao requerem: conversao de modelo, comparacoes de perplexidade,
  dados de divergencia KL, e benchmarks de performance
- Permitir acesso de escrita ao revisor na sua branch
- **Novos contribuidores: limite de 1 PR aberto por vez**

**Apos submissao:**
- Esperar pedidos de modificacao
- Fazer rebase de PRs desatualizados no ultimo `master`
- Considerar adicionar-se ao arquivo CODEOWNERS

### 1.4 Padrao de Commits

Formato do titulo de commit (squash-merge):
```
<modulo> : <titulo do commit> (#<numero_da_issue>)
```
Exemplo: `utils : fix typo in utils.py (#1234)`

### 1.5 Estilo de Codigo

- Minimizar dependencias de terceiros
- Garantir compatibilidade cross-platform
- Usar loops basicos, evitar STL/templates complexos
- Manter alinhamento vertical para legibilidade
- Indentacao de 4 espacos, chaves na mesma linha
- Usar tipos inteiros com tamanho (`int32_t`) em APIs publicas
- Preferir `struct foo {}` ao inves de padrao typedef
- Seguir padroes existentes; usar `clang-format` (v15+) em caso de duvida
- Arquivos C/C++ em minusculas com hifens, headers `.h`, fontes `.c`/`.cpp`
- Arquivos Python em minusculas com underscores

### 1.6 Convencoes de Nomenclatura

- **snake_case** para funcoes, variaveis, tipos
- Otimizar para prefixo comum mais longo: `number_small` ao inves de `small_number`
- Valores de enum em MAIUSCULAS com prefixo do enum: `LLAMA_VOCAB_TYPE_SPM`
- Padrao: `<classe>_<acao>_<substantivo>` (ex: `llama_sampler_get_seed`)
- Sufixo `_t` para tipos opacos
- Omitir keywords `struct` e `enum` opcionais quando nao necessarios

---

## 2. PRs Bem-Sucedidos que Adicionaram Novos Recursos

### 2.1 PR #1684 - k-quants (ikawrakow) -- O PR Fundacional

**Timeline:** Aberto em 3 de junho de 2023, merged em 5 de junho de 2023 (2 dias)

**Estrutura:**
- 32 commits
- Novos arquivos separados: `k_quants.h` e `k_quants.c` (modular, fora do `ggml.c`)
- Tipos introduzidos: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K (nivel GGML)
- Nove quantization mixes no nivel LLAMA
- Implementacoes: Scalar, AVX2, ARM_NEON, CUDA

**Benchmarks incluidos:**
- Metricas de performance em multiplas plataformas: M2 Max, RTX-4080, Ryzen 7950X
- Medidas de perplexidade no dataset WikiText para modelos 7B e 13B
- Tabelas comparativas mostrando F16 baseline vs Q2_K ate Q6_K
- Resultado chave: "quantizacao 6-bit com perplexidade dentro de 0.1% do modelo fp16 original"

**Licao:** O merge rapido de 2 dias refletiu benchmarks abrangentes e implementacao
completa em multiplas arquiteturas. Este PR e o template de ouro para novos tipos de
quantizacao.

### 2.2 PR #5676 - IQ3_S (ikawrakow) -- Quantizacao de 3 bits SOTA

**Timeline:** Aberto em 23 de fevereiro de 2024, merged em 24 de fevereiro de 2024 (~24 horas)

**Estrutura:**
- 27 commits
- Funcoes core de quantizacao/dequantizacao
- Otimizacoes de backend (CUDA, AVX2, ARM_NEON, Metal)
- Gerenciamento de codebooks usando tabelas lookup de 512 entradas
- Testes cross-platform para variantes QK_K=256 e QK_K=64

**Benchmarks:**
- "40-70% de melhoria no erro de quantizacao" em LLaMA-v1, LLaMA-v2 e Mistral-7B
- IQ3_S atingiu KL-divergence de 0.0205 vs 0.0304 para Q3_K_S no Mistral-7B

**Feedback do ggerganov:**
- Identificou falhas de compilacao com superblocks menores (QK_K=64)
- Solicitou correcoes antes da aprovacao
- Forneceu mensagens de erro especificas e guiou a resolucao

### 2.3 PR #5999 - "1.5 bit: we can do even better" (ikawrakow)

**Timeline:** Merged em 11 de marco de 2024

**Conteudo:** Melhoria na quantizacao de 1.5 bits usando um bit sobressalente para
"quant shift" em blocos de 32. Em vez de quants {-1, 0, 1}, usam {-1+delta, delta, 1+delta}.

**Interacao com ggerganov:** Aprovacao direta. No PR #5971 relacionado, ikawrakow
descobriu que o uso de `__constant__` no CUDA causava regressao drastica de performance
(17 t/s vs 204 t/s no RTX-4080). ggerganov reconheceu que foi um erro dele e que nao
sabia que podia causar mudancas tao drasticas.

**Licao:** Mostra que a interacao e colaborativa e construtiva. Os mantenedores
reconhecem erros abertamente.

### 2.4 PR #19769 - NVFP4 (richarddd) -- Exemplo Recente (Marco 2026)

**Timeline:** ~3 semanas de revisao e iteracao, merged em 11 de marco de 2026

**Estrutura:**
- Novo `GGML_TYPE_NVFP4` com estrutura de bloco
- Helpers de conversao para escala UE4M3
- Funcoes de referencia quantize/dequantize para CPU fallback
- Python tooling em `convert_hf_to_gguf.py`
- Suporte `gguf-py` com constantes de tipo
- Implementacao scalar de dot product + caminho otimizado ARM NEON

**Feedback dos revisores:**
1. **Estrategia de backend**: "engavetar todas as implementacoes de backend por agora" --
   suporte GPU adiado para PRs separados (alinhado com diretriz de comecar somente CPU)
2. **Requisitos de validacao**: Pedido de analise KLD, verificacoes de perplexidade,
   testes de benchmark usando modelos como Qwen3-4B (8-12B preferido sobre 400B)
3. **Questoes tecnicas**: Verificacao de formato contra documentacao NVIDIA
4. **Refinamento de formato**: Discussao sobre tamanho de bloco e implicacoes de
   alinhamento de 4 bytes

**Metricas:** NVFP4 a ~5.0 bits/peso com "perplexidade 15.25 vs 14.33 para FP16"

**Licao:** Este PR moderno mostra o padrao atual: CPU primeiro, backends GPU em PRs
separados, benchmarks com KLD e perplexidade obrigatorios, ~3 semanas de iteracao.

---

## 3. Os Mantenedores e Suas Preferencias

### 3.1 ggerganov (Georgi Gerganov) -- Mantenedor Principal

**Background:** Criador do llama.cpp (marco 2023), implementacao original de inferencia
Llama em C/C++ puro sem dependencias.

**Estilo de revisao:**
- Tecnico e direto, focado em correcao e arquitetura
- Identifica anti-patterns: ex. pre-alocar tensores com `ggml_new_tensor()` no compute
  graph e sempre um anti-pattern
- Reconhece erros abertamente quando feedback tecnico prova problemas
- Valoriza design modular e minimalista
- Prefere abordagens que usam infraestrutura existente (block-wise) em vez de mudancas
  fundamentais na arquitetura

**O que aceita:**
- Codigo bem testado com benchmarks extensivos em multiplas plataformas
- Melhorias incrementais que cabem na arquitetura existente
- PRs que comecam com CPU e adicionam backends GPU separadamente
- Contribuicoes de mantenedores de longo prazo que entendem o codebase

**O que rejeita/resiste:**
- Mudancas fundamentais na infraestrutura (ex: quantizacao por-linha em vez de por-bloco
  requer mudancas extensivas como `nb[0] == ggml_type_size(type)` espalhado pelo codigo)
- PRs predominantemente gerados por IA
- Contribuicoes sem benchmarks adequados
- PRs com ratio alto de "informacao nao verificada para trabalho necessario do mantenedor"
- Dependencias de terceiros desnecessarias

### 3.2 Outros Mantenedores Ativos

O repositorio usa arquivo CODEOWNERS para delegar responsabilidade por areas especificas.
O projeto foi destacado no GitHub Octoverse 2025 como "Top OSS by contributors".

### 3.3 O Caso ikawrakow -- Licao Importante

ikawrakow foi o contribuidor mais significativo para quantizacao no llama.cpp (criou
k-quants, i-quants, etc.). Eventualmente criou um fork separado (`ik_llama.cpp`) por
razoes que incluem:

- **Preferencia pessoal:** "Estou hackeando aqui para manter meu cerebro utilizado e me
  divertir. Definitivamente nao procuro fama e/ou adocao em massa."
- **Divergencia tecnica:** O fork implementa quantizacao per-tensor-row que diverge da
  regra estrita de quantizacao block-wise do ggml
- **Inovacoes nao upstream:** CPU Flash Attention com melhor performance que upstream,
  multiplicacoes de matrizes quantizadas superiores, row interleaving, suporte MLA
- **Posicao:** Nao tem intencao de fazer upstream para llama.cpp. Licenca MIT permite
  que upstream use livremente.

**Licao para nosso projeto:** Mesmo contribuidores altamente respeitados encontram
limites na arquitetura do llama.cpp. Mudancas fundamentais (como per-row scaling) sao
resistidas. Qualquer contribuicao precisa trabalhar dentro do paradigma block-wise
existente ou propor mudancas incrementais muito bem justificadas.

---

## 4. Requisitos Tecnicos para Novos Tipos de Quantizacao

### 4.1 Checklist Obrigatorio (do CONTRIBUTING.md)

Para novos tipos de quantizacao, o PR **deve** incluir:

1. **Conversao de modelo** -- suporte em `convert_hf_to_gguf.py`
2. **Comparacoes de perplexidade** -- vs tipos existentes em modelos padrao
3. **Dados de divergencia KL** -- metricas quantitativas de qualidade
4. **Benchmarks de performance** -- velocidade de inferencia em multiplas plataformas

### 4.2 Padrao Observado nos PRs Bem-Sucedidos

Baseado na analise dos PRs de k-quants, IQ3_S, e NVFP4:

**Arquivos a modificar (estimativa):**
- `ggml/include/ggml.h` -- registro do novo `GGML_TYPE_*`
- `ggml/src/ggml.c` -- funcoes de quantize/dequantize
- Arquivo(s) separado(s) para implementacao (padrao modular)
- Backend CPU com implementacao scalar de referencia
- `convert_hf_to_gguf.py` -- deteccao e conversao
- `gguf-py/` -- constantes de tipo e suporte de formato
- `tools/quantize/` -- integracao com ferramenta de quantizacao

**Implementacao por fases (padrao NVFP4):**
1. **Fase 1:** Registro de tipo GGML + funcoes CPU scalar de referencia
2. **Fase 2:** Python tooling (conversao HF -> GGUF)
3. **Fase 3:** Otimizacoes CPU (ARM NEON, AVX2)
4. **Fase 4:** (PR separado) Backend CUDA
5. **Fase 5:** (PR separado) Backend Metal
6. **Fase 6:** (PR separado) Outros backends

### 4.3 Testes Requeridos

- `test-backend-ops` para qualquer modificacao no ggml
- Comparacao de logits contra implementacao de referencia
- Testes de geracao de sequencias longas para coerencia
- CI completo passando em todas as plataformas
- Benchmarks em modelos de 7B-13B (preferido sobre modelos maiores para validacao)

### 4.4 Template Formal

Nao existe um template formal documentado para adicionar novos tipos de quantizacao.
Porem, o guia "Adding New Model Architectures" (Discussion #16770) e o documento
`docs/development/HOWTO-add-model.md` fornecem padroes analogos. O padrao observado
nos PRs bem-sucedidos serve como template de facto.

---

## 5. Discussoes sobre Compressao de Transporte

### 5.1 Discussion #8731 -- "Uncompressing Blocks when Required to Save VRAM"

**Proponente:** @snapo

**Proposta:** Compressao lossless de blocos GGUF usando LZ4 HC-9 para reduzir
requisitos de VRAM na GPU.

**Abordagem tecnica:**
- Comprimir blocos no momento da criacao do modelo
- Upload para GPU comprimido
- Descomprimir primeiro bloco -> operacoes matematicas -> descartar -> repetir

**Beneficios alegados:**
- Reducao de 20-30% no tamanho do arquivo (variando por camada)
- Permitiria rodar modelos maiores em GPUs consumer (ex: Llama 3.1 70B em 24GB VRAM)
- Qualidade lossless (zero perda vs quantizacao)

**Resultado:** A discussao recebeu **0 comentarios** de mantenedores ou comunidade.
Nenhuma resposta ou feedback de implementacao.

**Licao:** Propostas de compressao de transporte nao geraram interesse. Isso pode
indicar que: (a) o caso de uso nao e prioritario para os mantenedores, (b) a
complexidade de runtime e considerada inaceitavel, ou (c) a proposta nao tinha
implementacao concreta suficiente.

### 5.2 Compressao no Nivel de Formato GGUF

Nao foram encontradas discussoes ou propostas especificas sobre:
- Codificacao entropy (ANS/rANS) para pesos quantizados no GGUF
- Delta coding para reducao de tamanho de arquivo
- Compressao DCT de pesos de modelo
- Camada de transporte/armazenamento separada da representacao em memoria

A comunidade foca quase exclusivamente em **quantizacao** como mecanismo de compressao,
nao em compressao lossless adicional sobre pesos ja quantizados.

### 5.3 Abordagens Relacionadas no Ecossistema

- **ik_llama.cpp:** Experimentou com row-interleaved quant packing e novos formatos,
  mas no contexto de quantizacao, nao compressao de transporte
- **TurboQuant (Discussion #20969):** Compressao extrema de KV cache (3-4 bits), mas
  via quantizacao, nao entropy coding
- **PR #21038 (ggerganov):** Rotacao de ativacoes com Hadamard transform para melhor
  quantizacao -- mostra interesse em melhorar qualidade de quantizacao sem novos tipos

---

## 6. Fila de PRs e Ritmo de Merges

### 6.1 Estatisticas Atuais (Marco 2026)

| Metrica | Valor |
|---------|-------|
| PRs abertos | ~810 |
| PRs ja fechados | ~9.693 |
| Issues abertas | ~485 |
| Contribuidores totais | 1.038+ |
| Ultima issue fechada | ha horas |
| Ultimo PR merged | ha horas |

### 6.2 Ritmo de Releases

O projeto mantem um ciclo de releases **extremamente rapido**:
- So no dia 28 de marco de 2026: 9 releases distintos em ~18 horas
- Releases individuais as vezes separados por apenas 30-60 minutos
- Indica deploy automatizado/continuo acionado por PRs merged
- Build numbers atuais: b8559-b8574+

### 6.3 Tempo de Revisao Observado

| PR | Tipo | Contribuidor | Tempo ate Merge |
|----|------|-------------|-----------------|
| #1684 (k-quants) | Novo sistema de quant | ikawrakow (veterano) | 2 dias |
| #5676 (IQ3_S) | Novo tipo de quant | ikawrakow (veterano) | ~24 horas |
| #5999 (1.5-bit melhor) | Melhoria de quant | ikawrakow (veterano) | ~1 dia |
| #19769 (NVFP4) | Novo tipo de quant | richarddd (novo) | ~3 semanas |

**Padrao observado:**
- Contribuidores veteranos com historico: 1-2 dias
- Contribuidores novos com PRs complexos: 2-4 semanas
- PRs que precisam de iteracao (feedback -> correcao -> re-review): adicionar 1-2 semanas
- PRs sem benchmarks adequados ou com escopo muito amplo: podem ficar abertos indefinidamente

### 6.4 PRs Recentes na Fila (Marco 2026)

Exemplos de PRs ativos recentes:
- `CI: Enable CUDA ARM64 runners` (#21122)
- `metal: add opt-in V skip for negligible attention weights` (#21119)
- `convert: Add compressed-tensors NVFP4 conversion` (#21095)
- `ggml: add CPU TurboQuant KV cache types` (#21089)
- `[CUDA] Reduce stream-k blocks overhead` (#21086)
- `common: add bounds check in sampler` (#21082)

**Labels usados:** 93 labels incluindo `devops`, `ggml`, `Nvidia GPU`, `Apple Metal`,
`WebGPU`, `OpenCL`, `examples`, `server`, `python`, `testing`, `model`.

---

## 7. Implicacoes para Nosso Projeto (Compressao DCT de Pesos)

### 7.1 Viabilidade de Contribuicao

**Desafios significativos:**

1. **Sem precedente:** Nao existe discussao previa sobre compressao de transporte/
   armazenamento no llama.cpp que tenha ganhado tracao. A unica proposta (Discussion
   #8731) foi completamente ignorada.

2. **Filosofia do projeto:** O llama.cpp foca em quantizacao como mecanismo de compressao.
   Compressao lossless adicional nao esta no roadmap visivel.

3. **Politica anti-IA:** Se nosso codigo foi significativamente assistido por IA, isso
   e uma barreira real. O projeto pode banir permanentemente contribuidores que violem
   esta politica.

4. **Complexidade de runtime:** Adicionar decompressao no caminho critico de inferencia
   vai contra o principio de "minimalismo" do projeto.

5. **Infraestrutura block-wise:** Qualquer abordagem precisa trabalhar dentro do paradigma
   de quantizacao por blocos, nao por tensor inteiro ou por camada.

### 7.2 Estrategia Recomendada

Se quisermos contribuir para o llama.cpp, a abordagem mais viavel seria:

1. **Comecar com discussao, nao PR:** Abrir uma Discussion propondo compressao de
   transporte com dados concretos de compressao e benchmarks de velocidade de
   decompressao. Validar interesse antes de implementar.

2. **Focar em caso de uso claro:** O argumento mais forte seria reducao de tamanho de
   download/armazenamento SEM impacto na inferencia (decompressao ao carregar, nao
   durante inferencia).

3. **Formato de contribuicao ideal:**
   - Ferramenta standalone de compressao/decompressao de GGUF (menor escopo)
   - Opcional e transparente (nao muda fluxo de inferencia)
   - Suporte integrado no `llama-quantize` ou ferramenta separada
   - Zero dependencias externas

4. **Alternativa mais pragmatica:** Manter como ferramenta independente no nosso
   repositorio, nao tentar upstream. Similar ao que ikawrakow faz com ik_llama.cpp.

5. **Se submeter PR:**
   - Comecar somente CPU
   - Incluir benchmarks extensivos (ratio de compressao, velocidade, uso de memoria)
   - Comparar com alternativas triviais (gzip, lz4 no arquivo GGUF)
   - Demonstrar zero impacto na qualidade e performance de inferencia
   - Limitar escopo: compressao de armazenamento/transporte, nao compressao em runtime

---

## Fontes

### Repositorio Principal
- [llama.cpp - GitHub](https://github.com/ggml-org/llama.cpp)
- [CONTRIBUTING.md](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- [AGENTS.md](https://github.com/ggml-org/llama.cpp/blob/master/AGENTS.md)
- [Pull Requests](https://github.com/ggml-org/llama.cpp/pulls)
- [Releases](https://github.com/ggml-org/llama.cpp/releases)

### PRs Analisados
- [PR #1684 - k-quants (ikawrakow)](https://github.com/ggml-org/llama.cpp/pull/1684)
- [PR #5676 - IQ3_S (ikawrakow)](https://github.com/ggml-org/llama.cpp/pull/5676)
- [PR #5999 - 1.5 bit improved (ikawrakow)](https://github.com/ggml-org/llama.cpp/pull/5999)
- [PR #5971 - Better 1.5 bit (ikawrakow)](https://github.com/ggml-org/llama.cpp/pull/5971)
- [PR #19769 - NVFP4 (richarddd)](https://github.com/ggml-org/llama.cpp/pull/19769)
- [PR #21038 - Rotate activations (ggerganov)](https://github.com/ggml-org/llama.cpp/pull/21038)

### Discussoes
- [Discussion #5063 - Even more quantization types?](https://github.com/ggml-org/llama.cpp/discussions/5063)
- [Discussion #8731 - Uncompressing Blocks for VRAM](https://github.com/ggml-org/llama.cpp/discussions/8731)
- [Discussion #16770 - Guide: Adding New Model Architectures](https://github.com/ggml-org/llama.cpp/discussions/16770)
- [Discussion #20969 - TurboQuant](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [HOWTO-add-model.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md)

### Fork ik_llama.cpp
- [ik_llama.cpp - GitHub](https://github.com/ikawrakow/ik_llama.cpp)
- [Discussion #256 - Diverging from llama.cpp](https://github.com/ikawrakow/ik_llama.cpp/discussions/256)

### Pesquisas Academicas
- [Unified Evaluation of llama.cpp Quantization (arXiv:2601.14277)](https://arxiv.org/html/2601.14277v1)

### Guias e Artigos
- [llama.cpp GGUF Quantization Guide 2026](https://www.decodesfuture.com/articles/llama-cpp-gguf-quantization-guide-2026)
- [llama.cpp - Wikipedia](https://en.wikipedia.org/wiki/Llama.cpp)
- [quantize README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
