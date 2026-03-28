# Analise Teorica de GEMV (batch=1) em GPUs: INT4 vs FP16

> **Objetivo**: Entender por que GEMV e limitada por bandwidth, calcular a intensidade
> aritmetica, determinar o speedup teorico maximo de INT4 vs FP16, e investigar por que
> o speedup de 4x nao se materializa na pratica. Analise de kernels de producao (Marlin,
> ExLlama, llama.cpp) e suas eficiencias relativas ao pico teorico.

---

## 1. Por que GEMV e Bandwidth-Bound e Nao Compute-Bound?

### 1.1 O Problema Fundamental

Na geracao autoregressiva de tokens (batch=1), cada novo token exige a multiplicacao de
uma **matriz de pesos** (MxN) por um **unico vetor** de ativacoes (Nx1). Essa e a operacao
GEMV (General Matrix-Vector Multiply), fundamentalmente diferente da GEMM (General
Matrix-Matrix Multiply) usada no processamento de prompts.

A diferenca critica: **cada peso e carregado da memoria, usado exatamente uma vez, e
descartado**. Nao ha reutilizacao de dados.

### 1.2 Modelo Roofline

O modelo Roofline define a performance atingivel como:

```
P = min(pi, beta x I)
```

Onde:
- `pi` = pico de performance computacional (FLOP/s)
- `beta` = bandwidth de memoria (bytes/s)
- `I` = intensidade aritmetica (FLOP/byte)

O ponto de inflexao (ridge point) ocorre quando `I = pi / beta`. Operacoes com `I` abaixo
desse limiar sao **memory-bound**; acima sao **compute-bound**.

### 1.3 Numeros Concretos para GPUs Modernas

| GPU          | Pico FP16 (TFLOP/s) | Bandwidth (TB/s) | Ridge Point (FLOP/byte) |
|--------------|----------------------|-------------------|-------------------------|
| A100 SXM     | 312 (Tensor Core)    | 2.0               | ~156                    |
| A100 SXM     | 19.5 (CUDA Core)     | 2.0               | ~10                     |
| H100 SXM     | 990 (Tensor Core)    | 3.35              | ~296                    |
| A10          | 125 (Tensor Core)    | 0.6               | ~208                    |
| RTX 4090     | 330 (Tensor Core)    | 1.0               | ~330                    |

Para uma operacao ser compute-bound, ela precisa de **pelo menos 10-330 FLOPs por byte
transferido**, dependendo da GPU. GEMV tem intensidade aritmetica **muito abaixo** desse
limiar.

### 1.4 Por que GEMM (Prefill) e Diferente

Na GEMM com batch B, a mesma matriz de pesos e reutilizada B vezes. A intensidade
aritmetica cresce linearmente com B:

```
GEMM: I ~ 2B / sizeof(element)     (cresce com batch)
GEMV: I ~ 2 / sizeof(element)      (fixo, independe de M ou N)
```

E por isso que o processamento de prompt (batch=512) e compute-bound, mas a geracao
de tokens (batch=1) e memory-bound.

---

## 2. Intensidade Aritmetica de GEMV (FLOPs por Byte)

### 2.1 Calculo para FP16

Para uma GEMV com matriz A de dimensao MxN e vetor x de dimensao Nx1:

**FLOPs:**
```
2 * M * N   (N multiplicacoes + N-1 somas por cada um dos M elementos do resultado)
```

**Bytes transferidos (FP16, 2 bytes por elemento):**
```
Pesos:    2 * M * N  bytes  (matriz A)
Input:    2 * N      bytes  (vetor x)
Output:   2 * M      bytes  (vetor y)
Total:    2*M*N + 2*N + 2*M bytes
```

**Intensidade aritmetica (para M, N >> 1):**
```
I_FP16 = 2*M*N / (2*M*N + 2*N + 2*M)
       ~ 2*M*N / (2*M*N)         (termos menores desprezados)
       ~ 1.0 FLOP/byte
```

**Resultado: A intensidade aritmetica de GEMV FP16 e aproximadamente 1 FLOP/byte.**

Isso esta 10-330x abaixo do ridge point de GPUs modernas. GEMV e **profundamente
memory-bound** -- a GPU fica ociosa esperando dados da memoria.

### 2.2 Calculo para INT4 (Pesos Quantizados)

Com pesos INT4 (0.5 byte por peso), a conta muda:

**Bytes transferidos (pesos INT4, ativacoes FP16):**
```
Pesos:    0.5 * M * N bytes  (pesos INT4 empacotados)
Scales:   ~overhead  bytes   (metadados de quantizacao)
Input:    2 * N      bytes   (vetor x em FP16)
Output:   2 * M      bytes   (vetor y em FP16)
```

**FLOPs (saida em FP16 -- mesmos FLOPs apos desquantizacao):**
```
2 * M * N  (identico ao FP16)
```

**Intensidade aritmetica (sem overhead de metadados):**
```
I_INT4 = 2*M*N / (0.5*M*N + 2*N + 2*M)
       ~ 2*M*N / (0.5*M*N)         (para M, N grandes)
       ~ 4.0 FLOPs/byte
```

**INT4 tem intensidade aritmetica ~4x maior que FP16**, mas ainda e ~25-80x abaixo do
ridge point. Permanece firmemente no regime memory-bound.

### 2.3 Tabela Comparativa

| Precisao    | Bytes/peso | I (FLOP/byte) | Regime        | Distancia do Ridge (A100) |
|-------------|------------|----------------|---------------|---------------------------|
| FP32        | 4          | 0.5            | Memory-bound  | ~20x abaixo               |
| FP16        | 2          | 1.0            | Memory-bound  | ~10x abaixo               |
| INT8        | 1          | 2.0            | Memory-bound  | ~5x abaixo                |
| INT4        | 0.5        | 4.0            | Memory-bound  | ~2.5x abaixo              |
| INT2        | 0.25       | 8.0            | Memory-bound  | ~1.2x abaixo              |

Mesmo INT2 ainda seria memory-bound em GPUs com Tensor Cores!

---

## 3. Speedup Teorico Maximo de INT4 vs FP16

### 3.1 Caso Ideal (Sem Overhead)

Se a operacao e puramente memory-bound e o kernel utiliza 100% da bandwidth disponivel,
o speedup e determinado pela **razao de dados transferidos**:

```
Speedup_ideal = Bytes_FP16 / Bytes_INT4
              = (2 * M * N) / (0.5 * M * N)
              = 4.0x
```

**No caso ideal, INT4 seria exatamente 4x mais rapido que FP16.**

### 3.2 Caso Realista: O Limite de 3.87x

Na pratica, pesos INT4 com quantizacao por grupo (group_size=128) carregam metadados
adicionais:

- **Scale** (FP16): 2 bytes por grupo de 128 pesos = 0.0156 bytes/peso
- **Zero point** (opcional): mais 0.0156 bytes/peso

Para quantizacao assimetrica com group_size=128:
```
Bytes_efetivos_INT4 = 0.5 + 0.0156 + 0.0156 = 0.531 bytes/peso  (~4.25 bits/peso)
```

Para quantizacao simetrica (apenas scale):
```
Bytes_efetivos_INT4 = 0.5 + 0.0156 = 0.516 bytes/peso  (~4.13 bits/peso)
```

O **speedup teorico maximo** com metadados de grupo:
```
Speedup_real = 2.0 / 0.516 = 3.87x   (quantizacao simetrica, group=128)
Speedup_real = 2.0 / 0.531 = 3.77x   (quantizacao assimetrica, group=128)
```

**O Marlin paper confirma este calculo: o speedup maximo teorico e 3.87x**, contabilizando
0.125 bits de overhead dos group scales.

### 3.3 Formatos Comuns e Seus Limites Teoricos

| Formato      | Bits/peso efetivos | Speedup teorico max vs FP16 |
|--------------|--------------------|-----------------------------|
| Q4_0 (GGUF)  | 4.5               | 3.56x                       |
| Q4_K (GGUF)  | 4.5               | 3.56x                       |
| GPTQ g128    | ~4.13              | 3.87x                       |
| AWQ g128     | ~4.13              | 3.87x                       |
| EXL2 4.0bpw  | 4.0               | 4.00x                       |
| NF4 (bnb)    | ~4.5               | 3.56x                       |

Nota: Q4_0 no llama.cpp usa blocos de 32 pesos com 1 scale FP16 (2 bytes) para 32 pesos
de 4-bit (16 bytes), totalizando 18 bytes para 32 pesos = 4.5 bits/peso efetivos.

---

## 4. Por que INT4 NAO Atinge 4x na Pratica?

Esta e a questao central. Se INT4 le ~4x menos dados e a operacao e memory-bound,
por que nao vemos ~4x de speedup? Existem multiplas causas, ordenadas por impacto.

### 4.1 Overhead de Desquantizacao (Custo ALU)

**Impacto: ALTO**

Pesos INT4 precisam ser convertidos para FP16 antes da multiplicacao. Este processo
envolve:

1. **Desempacotamento (unpacking)**: Extrair valores de 4 bits de bytes empacotados
   - Operacoes de shift e mask: `val_lo = byte & 0xF; val_hi = byte >> 4;`
   - 2-3 instrucoes ALU por par de pesos

2. **Conversao de tipo**: INT4 -> FP16
   - Pode usar LUT (lookup table) ou instrucoes de conversao
   - Em Ampere/Hopper: intrinsics especializados (`__nv_cvt_fp4x2_to_halfraw2`)
   - Em hardware mais antigo: sequencia mask-unpack-convert mais longa

3. **Aplicacao de scale e zero point**:
   - `peso_fp16 = scale * (peso_int4 - zero_point)`
   - 1-2 FMA por peso (multiply-add)

**Custo total**: 5-10 instrucoes ALU adicionais por peso, que nao existem no path FP16.

**Dados concretos do QServe paper**: Em kernels W4A16 do TensorRT-LLM, a desquantizacao
ocorre **dentro do main loop** nos CUDA cores, enquanto os tensor cores processam a
multiplicacao. Uma operacao de CUDA core custa o equivalente a ~50 operacoes INT4 de
tensor core em GPUs datacenter como A100. Isso significa que o custo de desquantizacao
nos CUDA cores e o gargalo real, nao a multiplicacao nos tensor cores.

**Insight chave**: Para GEMV no regime memory-bound, o custo ALU da desquantizacao
idealmente deveria ser **completamente escondido** atras da latencia de memoria. Isso
so e possivel com pipelines cuidadosamente orquestrados (como o Marlin faz). Em kernels
naive, a desquantizacao serializa com os loads de memoria.

### 4.2 Metadados de Quantizacao (Scale Lookups)

**Impacto: MEDIO**

Os group scales adicionam trafego de memoria extra:

```
Overhead de scales com group_size=128:
  - 2 bytes (FP16) por grupo de 128 pesos
  - = 0.0156 bytes por peso adicional
  - = ~3% de trafego extra

Overhead de scales com group_size=32 (Q4_K):
  - scales quantizados a 6 bits no llama.cpp
  - Estrutura complexa com super-blocos e sub-blocos
  - ~12.5% overhead efetivo
```

Alem do trafego extra, os scales introduzem **padroes de acesso nao-sequencial**:
- Pesos sao lidos sequencialmente (coalesced)
- Scales sao compartilhados por grupo e requerem broadcast
- Em implementacoes naive, isso cria acessos "random" adicionais

### 4.3 Output em FP16 (Assimetria de Dados)

**Impacto: BAIXO-MEDIO**

Independente da precisao dos pesos, o vetor de saida e sempre escrito em FP16 (ou FP32):

```
GEMV FP16: Le 2*M*N bytes de pesos, escreve 2*M bytes de output
GEMV INT4: Le 0.5*M*N bytes de pesos, escreve 2*M bytes de output
```

A razao output/input muda dramaticamente:
```
FP16: output/pesos = 2*M / (2*M*N) = 1/N  (desprezivel para N grande)
INT4: output/pesos = 2*M / (0.5*M*N) = 4/N (4x maior proporcao)
```

Para N=4096 (tipico de LLMs), o output ainda e ~0.1% dos pesos. **Este fator e
negligenciavel para matrizes grandes**, mas pode importar para camadas menores.

Mais relevante: o **acumulador intermediario** geralmente usa FP32 (4 bytes), e a
conversao final FP32->FP16 adiciona um passo extra. Usar `atomicAdd` em FP32 para
somas parciais entre blocos introduz contencao e trafego extra de memoria.

### 4.4 Warp Divergence no Unpacking

**Impacto: BAIXO**

O desempacotamento de INT4 requer caminhos de codigo diferentes para nibbles
pares e impares:

```cuda
// Pseudo-codigo simplificado
uint8_t packed = weights[i/2];
half value;
if (i % 2 == 0)
    value = scale * ((packed & 0xF) - zero);  // nibble inferior
else
    value = scale * ((packed >> 4) - zero);    // nibble superior
```

Em principio, isso poderia causar warp divergence (threads no mesmo warp seguindo
caminhos diferentes). Na pratica, kernels bem escritos evitam isso processando
**pares de valores** por thread, eliminando o branch. O Marlin, por exemplo, processa
4 pesos simultaneamente por thread com paralelismo a nivel de registrador.

**Veredicto**: Nao e um problema em kernels otimizados.

### 4.5 Pressao de Registradores e Reducao de Occupancy

**Impacto: MEDIO**

A desquantizacao requer registradores adicionais para:
- Valores INT4 empacotados (antes do unpacking)
- Valores FP16 desempacotados (apos conversao)
- Scales e zero points do grupo atual
- Acumuladores FP32 para somas parciais

O paper do QServe demonstra que a desquantizacao de **pesos** e preferida sobre a
desquantizacao de **somas parciais** justamente por causar menor pressao de registradores.
O design de 4-way register-level parallelism (decodificar 4 pesos INT4 simultaneamente)
minimiza a sobrecarga.

No entanto, maior uso de registradores -> menor occupancy -> menos warps em voo ->
**menor capacidade de esconder latencia de memoria**. Isso e particularmente danoso
para GEMV, que depende criticamente de ter muitos warps em voo para saturar a
bandwidth.

### 4.6 Subutilizacao de Bandwidth (Problema do Kernel)

**Impacto: MUITO ALTO**

Este e frequentemente o fator dominante. Mesmo operacoes FP16 de GEMV frequentemente
**nao saturam a bandwidth de memoria disponivel**. As razoes incluem:

1. **Overhead de lancamento de kernel**: Cada operacao GEMV e um kernel CUDA separado;
   o custo de despacho (~5-10 us) nao e desprezivel para operacoes rapidas

2. **Tamanho de dados insuficiente**: Em INT4, os dados sao tao pequenos que o kernel
   pode terminar antes de saturar o pipeline de memoria
   - Uma camada 4096x4096 em INT4 = 8 MB de pesos
   - O A100 tem 2 TB/s de bandwidth = 4 us para ler 8 MB
   - O overhead de lancamento e sincronizacao pode ser comparavel!

3. **Fragmentacao de requisicoes**: Loads menores nao preenchem as filas de requisicao
   de memoria tao eficientemente

4. **Competicao por recursos**: L2 cache, NOC (Network on Chip), e schedulers sao
   compartilhados entre todos os SMs

**Dados empiricos do llama.cpp**: Rodando um modelo 7B Q4 no M2 Max (400 GB/s),
~60 tok/s corresponde a apenas ~50% de utilizacao de bandwidth de memoria (MBU).
Metade da bandwidth esta sendo desperdicada.

### 4.7 Resumo: Decomposicao da Perda de Speedup

```
Speedup teorico maximo:                    4.00x
  - Overhead de metadados (scales):       -0.13x  -> 3.87x
  - Subutilizacao de bandwidth do kernel: -0.87x  -> 3.00x  (tipico)
  - Desquantizacao nao-pipelined:         -0.30x  -> 2.70x  (kernels mediocres)
  - Pressao de registradores:             -0.10x  -> 2.60x
  - Overhead de lancamento de kernel:     -0.10x  -> 2.50x

Speedup tipico observado:                  2.5x - 3.5x  (depende do kernel e hardware)
Speedup de kernels estado-da-arte (Marlin): 3.87x       (perto do limite teorico)
```

---

## 5. Kernels de Producao: Comparacao com o Pico Teorico

### 5.1 Marlin (IST-DASLab)

**O kernel INT4 GEMV mais otimizado disponivel publicamente.**

| Metrica               | Valor                                      |
|-----------------------|--------------------------------------------|
| Speedup batch=1       | ~3.87x (maximo teorico atingido!)          |
| Speedup batch=16-32   | ~3.87x (mantido ate batch moderado)        |
| Speedup batch=64+     | Gradualmente decrescente                   |
| GPU testada           | NVIDIA A10                                 |
| Bandwidth utilization | Proximo de 100% no regime memory-bound     |

**Tecnicas chave do Marlin:**

1. **Pipeline assincrono de memoria**: Usa `cp.async` para prefetch de pesos do proximo
   bloco enquanto computa o bloco atual. Pipeline depth P permite sobrepor P etapas
   de load com P etapas de compute.

2. **Desquantizacao pipelined**: A desquantizacao do proximo operando B e sobreposta com
   a operacao de Tensor Core do operando atual. O custo ALU da desquantizacao e
   completamente escondido.

3. **Loads de 128 bits**: Cada thread carrega 16 bytes por instrucao (load maximo),
   maximizando a eficiencia de bandwidth.

4. **Reutilizacao de ativacoes via L2**: Ativacoes (vetor de input) sao mantidas no L2
   cache e reutilizadas por multiplos blocos de pesos, reduzindo trafego para DRAM.

5. **Layouts otimizados offline**: Pesos e scales sao reorganizados em layouts que
   garantem acessos coalescidos e livres de conflitos em shared memory.

6. **Double buffering em shared memory**: Carrega o proximo tile em um buffer enquanto
   computa com o buffer atual, eliminando stalls.

**Por que Marlin consegue o limite teorico**: A chave e que no regime memory-bound, o
custo de compute (incluindo desquantizacao) pode ser **completamente escondido** se
o pipeline for profundo o suficiente. Marlin prova que nao existe barreira fundamental --
o limite e realmente a bandwidth de memoria vezes a razao de compressao.

### 5.2 ExLlamaV2 (turboderp)

| Metrica               | Valor                                      |
|-----------------------|--------------------------------------------|
| Speedup vs load_in_4b | 147% mais tok/s                            |
| Speedup vs llama.cpp  | 85% mais tok/s                             |
| tok/s (7B, T4)        | ~56 tok/s                                  |
| Formato               | EXL2 (mixed bitrate) e GPTQ                |
| Bandwidth utilization | Estimado 70-85% (inferido de benchmarks)   |

**Tecnicas chave:**
- Kernel GEMV customizado com desquantizacao fundida
- Suporte a bitrates mistos (2-8 bits por camada)
- Otimizado especificamente para GPUs consumer (RTX 3090, 4090)

**Limitacao principal**: Focado em single-GPU consumer; nao atinge a eficiencia do
Marlin em hardware datacenter.

### 5.3 llama.cpp (ggml)

| Metrica               | Valor                                      |
|-----------------------|--------------------------------------------|
| MBU (M2 Max, Q4)     | ~50% (60 tok/s em 400 GB/s)               |
| MBU (RTX 4090, Q4)   | ~60-70% (estimado)                         |
| MBU (CPU AVX2)       | ~70-80% (framework maturo)                 |
| Formato               | GGUF (Q4_0, Q4_K_M, etc.)                 |
| Plataformas           | CUDA, Metal, Vulkan, CPU (AVX/NEON/SVE)   |

**Tecnicas chave (CUDA backend):**
- Kernels GEMV dedicados (`mul_mat_vec`) para cada tipo de quantizacao
- Desquantizacao fundida com multiply-accumulate
- Uso de `__dp4a` para dot product de inteiros em hardware que suporta

**Tecnicas chave (Metal backend -- Apple Silicon):**
- Kernels Metal compute shaders otimizados
- Uso de SIMD group operations para reducao
- Overhead nao desprezivel observado mesmo com kernels NOP (no-operation)

**Limitacao principal**: Suporte multi-plataforma implica em generalizacao que sacrifica
otimizacao peak. O overhead do framework Metal e significativo (~15-20% em Apple Silicon).

### 5.4 Tabela Comparativa

| Kernel       | Speedup INT4/FP16 | % do Teorico (3.87x) | MBU Estimada | Plataforma     |
|--------------|--------------------|-----------------------|--------------|----------------|
| Marlin       | 3.87x              | ~100%                 | ~95%+        | CUDA (datactr) |
| ExLlamaV2    | ~3.0-3.5x          | ~80-90%               | ~70-85%      | CUDA (consumer)|
| llama.cpp    | ~2.5-3.0x          | ~65-78%               | ~50-70%      | Multi-platform |
| bitsandbytes | ~2.0-2.5x          | ~52-65%               | ~40-60%      | CUDA           |
| PyTorch naive| ~1.5-2.0x          | ~39-52%               | ~30-50%      | CUDA           |

---

## 6. Utilizacao de Bandwidth: Analise Detalhada

### 6.1 O que Significa "MBU" (Memory Bandwidth Utilization)

```
MBU = (Bytes transferidos por segundo pelo kernel) / (Bandwidth de pico do hardware)
```

Para GEMV: se um kernel esta atingindo 100% MBU, ele esta lendo pesos da DRAM na
velocidade maxima possivel. Qualquer perda de MBU significa que a GPU esta ociosa
parcialmente.

### 6.2 Como Medir

**Metodo 1 -- Throughput de tokens:**
```
MBU = (tokens/s * bytes_por_token) / bandwidth_pico

Exemplo (7B Q4_K_M no M2 Max):
  bytes_por_token = tamanho_modelo / num_camadas... (modelo inteiro e lido por token)
  ~3.8 GB para 7B Q4_K_M
  60 tok/s * 3.8 GB = 228 GB/s
  MBU = 228 / 400 = 57%
```

**Metodo 2 -- Profiling direto:**
- `ncu` (Nsight Compute) no CUDA: reporta DRAM throughput por kernel
- Metal System Trace no macOS: reporta bandwidth de GPU

### 6.3 Por que Kernels Nao Atingem 100% MBU

**Fatores hardware:**
1. ECC ativado reduz bandwidth efetiva em 10-15% (A10, A100 com ECC)
2. Refresh de DRAM consome ciclos periodicamente
3. Competicao com outros consumidores de bandwidth (display, OS, KV cache)

**Fatores de software:**
1. Overhead de lancamento de kernel (~5-10 us por kernel)
2. Sincronizacao entre kernels (barriers, stream sync)
3. Loads nao-coalescidos (acessos a scales, zero points)
4. Pipeline depth insuficiente (nao esconde latencia de DRAM)
5. Occupancy reduzida (poucos warps, nao satura controladores de memoria)

### 6.4 A Curva de Utilizacao vs Tamanho da Matriz

```
Tamanho matriz  |  Bytes INT4   |  MBU tipica  |  Nota
----------------|---------------|--------------|------
256 x 256       |  32 KB        |  10-20%      |  Muito pequeno, overhead domina
1024 x 1024     |  512 KB       |  30-50%      |  Subotimo
4096 x 4096     |  8 MB         |  60-80%      |  Tipico de LLM
4096 x 11008    |  22 MB        |  70-90%      |  Camada MLP tipica
8192 x 28672    |  117 MB       |  85-95%      |  Camadas grandes, quase ideal
```

Matrizes maiores naturalmente atingem MBU mais alta, pois o overhead fixo de
lancamento/sincronizacao e amortizado sobre mais dados.

---

## 7. A Pergunta Central: Se INT4 Le 4x Menos Dados, Por que Nao e 4x Mais Rapido?

### 7.1 Resposta Curta

**E possivel ser ~4x mais rapido. O Marlin prova isso.** A maioria dos kernels simplesmente
nao e otimizada o suficiente.

### 7.2 Resposta Longa -- Decomposicao dos Fatores

| Fator                           | Perda | Evitavel? | Como evitar                        |
|---------------------------------|-------|-----------|------------------------------------|
| Metadados de grupo (scales)     | ~3%   | Parcial   | Grupos maiores (mas perde precisao)|
| Desquantizacao nos CUDA cores   | 5-15% | Sim       | Pipeline com tensor core math      |
| Subutilizacao de bandwidth      | 10-40%| Sim       | Pipeline assincrono, loads largos  |
| Pressao de registradores        | 2-5%  | Sim       | 4-way parallel dequant             |
| Overhead de kernel launch       | 2-10% | Parcial   | Fusao de kernels, persistent kernels|
| Output FP16/FP32 writeback      | <1%   | N/A       | Desprezivel para matrizes grandes  |
| Warp divergence (unpacking)     | ~0%   | Sim       | Processar pares, sem branches      |

### 7.3 O Insight do Blog "Twelve Attempts at an FP4 Kernel"

Um caso ilustrativo: durante o NVFP4 Kernel Hackathon em GPUs B200, o autor documentou
12 tentativas de otimizar um kernel GEMV para FP4 (4-bit floating point do Blackwell).

Licoes aprendidas:
1. **Split-K nao ajuda em kernels memory-bound** -- mais blocos significam mais
   scheduling overhead e contencao atomica, sem ganho de bandwidth
2. **Loads mais largos so ajudam se os dados podem ser usados diretamente** -- se
   precisam de unpacking extenso, o ganho se perde
3. **ILP (Instruction-Level Parallelism) e irrelevante quando o gargalo e memoria**
4. **Tuning de registradores nao faz nada se ja esta abaixo do limite**
5. Os melhores kernels ficaram **~2x distantes do speed of light** (50% MBU)

A conclusao: **entender o perfil (memory vs compute bound) ANTES de otimizar e critico**.
Muitos ciclos de otimizacao sao desperdicados tentando melhorar compute quando o gargalo
e memoria.

### 7.4 Hierarquia de Otimizacao para GEMV INT4

Em ordem de impacto:

```
1. Saturar bandwidth de memoria (MBU > 90%)
   - Loads assincrono com cp.async
   - Pipeline depth suficiente (P >= 3-4)
   - Loads de 128 bits (16 bytes/thread)

2. Esconder latencia de desquantizacao
   - Sobrepor dequant do bloco N+1 com compute do bloco N
   - Usar intrinsics de conversao quando disponiveis

3. Minimizar trafego de metadados
   - Layouts de scales coalescidos
   - Broadcast eficiente de scales para o grupo

4. Maximizar occupancy
   - Minimizar uso de registradores por thread
   - Minimizar uso de shared memory por bloco

5. Reduzir overhead fixo
   - Fusao de kernels adjacentes
   - Persistent kernels para camadas pequenas
```

---

## 8. Implicacoes para Nosso Projeto (DCT-Quantization)

### 8.1 Qualquer Formato de Compressao e Limitado pela Bandwidth

Se criarmos um formato que comprime pesos a 3 bits/peso efetivos (vs 4 bits do INT4),
o speedup teorico adicional sobre INT4 seria:

```
Speedup_adicional = 4.0 / 3.0 = 1.33x
```

Mas **somente se o kernel conseguir manter a mesma MBU**. Se a descompressao DCT for
mais complexa que a desquantizacao INT4 simples, podemos perder esse ganho no overhead.

### 8.2 A Restricao Critica

Para um formato customizado ser mais rapido que INT4:
```
Tempo_total = Tempo_load + Tempo_dequant_custom
Tempo_total < Tempo_load_INT4 + Tempo_dequant_INT4
```

Como ambos sao memory-bound:
```
Dados_custom / Bandwidth + Overhead_dequant_custom < Dados_INT4 / Bandwidth + Overhead_dequant_INT4
```

Se `Overhead_dequant_custom >> Overhead_dequant_INT4`, podemos acabar **mais lentos**
mesmo com menos dados. A descompressao DCT (transformada inversa, lookup de codebook)
e significativamente mais cara que INT4 simples (shift + mask + FMA).

### 8.3 Caminhos Viaveis

1. **Se o formato e decodificavel com <= 25-50 operacoes por peso** (limiar Marlin),
   o custo de dequant pode ser escondido atras da latencia de memoria

2. **Se o formato permite loads coalescidos e sem branches**, a MBU nao sera degradada

3. **Se o ganho de compressao e > 25%** (ex: 3 bits vs 4 bits), ha margem para absorver
   algum overhead de descompressao adicional

---

## 9. Conclusoes

1. **GEMV (batch=1) e fundamentalmente memory-bound** em todas as GPUs modernas, com
   intensidade aritmetica de ~1 FLOP/byte (FP16) ou ~4 FLOP/byte (INT4) -- ambos
   muito abaixo do ridge point de 10-330 FLOP/byte.

2. **O speedup teorico maximo de INT4 vs FP16 e ~3.87x** (contabilizando metadados de
   grupo com group_size=128). Para Q4_0/Q4_K (4.5 bits/peso), o maximo e ~3.56x.

3. **Nao existe barreira fundamental que impeca INT4 de atingir ~4x** -- o Marlin
   demonstra 3.87x em hardware real (A10 GPU), provando que o limite teorico e
   alcancavel.

4. **A maioria dos kernels fica entre 2.5x e 3.5x** porque:
   - Nao saturam a bandwidth de memoria (50-70% MBU)
   - Nao escondem completamente o custo de desquantizacao
   - Tem overhead de framework (especialmente multi-plataforma)

5. **Os melhores kernels (Marlin) atingem >95% MBU** atraves de:
   - Pipeline assincrono profundo (cp.async)
   - Overlap de dequant com tensor core math
   - Loads de 128 bits com layouts pre-otimizados

6. **Para formatos de compressao customizados (como DCT)**, o desafio nao e reduzir
   bits por peso, mas garantir que a descompressao adicional possa ser completamente
   escondida atras da latencia de memoria, mantendo MBU > 90%.

---

## Fontes

- [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models (arXiv 2408.11743)](https://arxiv.org/abs/2408.11743)
- [Marlin GitHub -- IST-DASLab](https://github.com/IST-DASLab/marlin)
- [How Marlin pushes the boundaries of mixed-precision LLM inference (Red Hat)](https://developers.redhat.com/articles/2024/04/17/how-marlin-pushes-boundaries-mixed-precision-llm-inference)
- [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](https://arxiv.org/html/2405.04532v3)
- [Understanding INT4 Quantization for Transformer Models (arXiv 2301.12017)](https://arxiv.org/abs/2301.12017)
- [Twelve Attempts at an FP4 Kernel (Amandeep Singh)](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/)
- [A Systematic Characterization of LLM Inference on GPUs (arXiv 2512.01644)](https://arxiv.org/html/2512.01644v1)
- [Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference (arXiv 2503.08311)](https://arxiv.org/html/2503.08311v2)
- [LLM Inference Optimization on AMD GPUs (ROCm Blogs)](https://rocm.blogs.amd.com/artificial-intelligence/llm-inference-optimize/README.html)
- [NVIDIA Matrix Multiplication Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
- [Roofline Model -- JAX Scaling Book](https://jax-ml.github.io/scaling-book/roofline/)
- [FastGEMV -- High-speed GEMV kernels](https://github.com/wangsiping97/FastGEMV)
- [llama.cpp Memory Bandwidth Utilization Discussion #3909](https://github.com/ggml-org/llama.cpp/discussions/3909)
- [llama.cpp Apple Silicon Performance Discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167)
- [ExLlamaV2 GitHub](https://github.com/turboderp-org/exllamav2)
- [A guide to LLM inference and performance (Baseten)](https://www.baseten.co/blog/llm-transformer-inference-guide/)
- [LLM Inference Series: Dissecting Model Performance (Pierre Lienhart)](https://medium.com/@plienhar/llm-inference-series-5-dissecting-model-performance-6144aa93168f)
