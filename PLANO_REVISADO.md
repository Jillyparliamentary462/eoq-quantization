# DCT Project — Plano Revisado com Base em Dados Reais

## O que os experimentos provaram

### DESCARTADO (não funciona em pesos densos de transformers)

| Técnica | Por quê não funciona | Evidência |
|---------|---------------------|-----------|
| Delta coding (pesos brutos) | Camadas adjacentes têm correlação ZERO (cosine ~0.0) | Exp A: 3 runs |
| DCT/Wavelet 2D | Pesos não têm correlação espacial (autocorr lag-1 = 0.003) | Exp D: 3 runs |
| SVD+Q em 4-bit+ | Quantização direta vence 100% das vezes acima de 3 bits | Exp C: 3 runs |
| Delta SVD | Deltas têm rank efetivo MAIOR que originais | Exp C: Phase 5 |

### CONFIRMADO (funciona)

| Técnica | Ganho | Evidência |
|---------|-------|-----------|
| **Entropy coding** | **62.4% compressão extra** em Q4 (entropia real = 1.5 bits de 4 alocados) | Exp B: 3 runs |
| SVD+Q em sub-3-bit | 100% win rate no regime Q2 | Exp C: 3 runs |
| Delta coding em LayerNorm | 35x vantagem | Exp A: componentes ln1/ln2 |

## Nova Tese

A contribuição original viável **não é delta coding + DCT**. É:

> **Entropy-Optimal Quantization (EOQ)**: combinar quantização existente com
> entropy coding (rANS) para eliminar o gap de 62% entre bits alocados e
> entropia real, opcionalmente usando SVD híbrido para regime sub-2.5 bpw.

### Impacto projetado (com dados reais)

```
Qwen 3.5 4B em BF16:           8.42 GB
Q4_K_M atual (GGUF):           2.71 GB  (4.83 bpw)
Q4 + entropy coding (EOQ):    ~1.02 GB  (1.50 bpw efetivo)
Q2 + SVD híbrido + entropy:   ~0.70 GB  (~1.0 bpw efetivo)
```

## Plano de Execução Revisado

### Fase 1: EOQ-Core (Entropy-Optimal Quantization)

**Objetivo**: Implementar rANS encoder/decoder para pesos quantizados e demonstrar
compressão de 62% sobre Q4 padrão sem perda de qualidade.

1. Implementar rANS encoder/decoder em Python puro (protótipo)
2. Integrar com quantize_absmax: quantizar → computar tabela de frequências → entropy code
3. Implementar decodificação por bloco (256 pesos) com tabela de offsets para random access
4. Benchmark: tamanho comprimido vs Q4_K_M no Qwen2.5-0.5B
5. Validar: dequantizar, comparar bit-a-bit com Q4 direto (deve ser lossless)

### Fase 2: SVD Híbrido para Ultra-Low-Bit

**Objetivo**: Para regime sub-2.5 bpw, usar formato W = Q_low + U*S*V' onde Q_low
é quantização agressiva e USV' é correção low-rank.

1. Implementar formato híbrido: Q2 dos pesos + rank-R SVD do resíduo
2. Encontrar R ótimo por camada (variance-based allocation)
3. Entropy coding nos fatores SVD quantizados
4. Comparar contra IQ2_M (1.76 GB) e UD-IQ2_XXS (1.52 GB)

### Fase 3: Formato .eoq e Integração

1. Definir formato binário .eoq (header + entropy-coded blocks + offset table)
2. Implementar loader Python que decodifica para tensores PyTorch
3. Benchmark de velocidade de decodificação
4. Documentar especificação para futura integração com llama.cpp

## O que NÃO fazer mais

- Não investir mais em delta coding para matrizes densas
- Não investir em DCT/wavelet 2D para matrizes de peso
- Não investir em neural dequantizer (pesquisa mostrou ~500x mais lento, e LUT resolve)
- Não investir em quantização progressiva/LOD (complexidade alta, ganho incerto)
