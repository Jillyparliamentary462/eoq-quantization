# Nível 2: Compressão de RAM — Pesos INT4 na memória, dequant on-the-fly

## Objetivo
Manter pesos quantizados (INT4) na RAM em vez de FP32.
Dequantizar DURANTE o matmul, não antes.
Resultado: ~4x menos RAM → ~4x menos bandwidth → mais tok/s.

## Arquitetura

```
Hoje:     nn.Linear(weight=FP32[4096x4096])     → 64 MB por camada
Nível 2:  QuantizedLinear(codes=INT4, scales=FP16) → ~8 MB por camada
          forward(): dequant(codes, scales) @ input
```

## Componentes

1. QuantizedLinear — nn.Module com pesos INT4 na memória
2. EOQLinear — variante com pesos rANS na memória (decode lazy)
3. ModelPatcher — substitui nn.Linear → QuantizedLinear automaticamente
4. Benchmark tok/s — FP32 vs Q4-in-RAM vs EOQ-in-RAM
5. Memory profiler — medir RAM real
6. Server atualizado — chat com inferência quantizada real
