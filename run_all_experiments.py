#!/usr/bin/env python3
"""
DCT (Delta-Coded Transformer) - Master Experiment Runner

Runs all experiments for the DCT quantization research project.
Usage:
    python run_all_experiments.py --model Qwen/Qwen2.5-0.5B --device cpu
    python run_all_experiments.py --model Qwen/Qwen2.5-4B --device cuda --experiments a,b,c
    python run_all_experiments.py --model Qwen/Qwen2.5-4B --device mps --parallel
"""

import argparse, subprocess, sys, os, time, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

EXPERIMENTS = {
    'a': ('Inter-layer Correlation', 'experiments/exp_a_correlation/measure_correlation.py'),
    'b': ('Entropy Analysis', 'experiments/exp_b_entropy/measure_entropy.py'),
    'c': ('SVD + Quantization', 'experiments/exp_c_svd/svd_quantize.py'),
    'd': ('Frequency Domain Analysis', 'experiments/exp_d_frequency/frequency_analysis.py'),
    'e': ('Neural Dequantizer', 'experiments/exp_e_neural_dequant/neural_dequantizer.py'),
    'f': ('Delta Coding', 'experiments/exp_f_delta_coding/delta_coding.py'),
    'g': ('Progressive Quantization', 'experiments/exp_g_progressive/progressive_quant.py'),
    'h': ('Combined DCT Pipeline', 'experiments/exp_h_combined/combined_pipeline.py'),
}

def run_experiment(key, script_path, model, device, output_base):
    name = EXPERIMENTS[key][0]
    print(f"\n{'='*60}")
    print(f"  Running Experiment {key.upper()}: {name}")
    print(f"{'='*60}")

    output_dir = os.path.join(output_base, f'exp_{key}')
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, script_path,
        '--model', model,
        '--device', device,
        '--output', output_dir,
    ]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    status = 'SUCCESS' if result.returncode == 0 else 'FAILED'
    print(f"  [{status}] Experiment {key.upper()} completed in {elapsed:.1f}s")

    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[:500]}")

    return {
        'experiment': key,
        'name': name,
        'status': status,
        'elapsed_seconds': elapsed,
        'returncode': result.returncode,
        'stdout_tail': result.stdout[-500:] if result.stdout else '',
        'stderr_tail': result.stderr[-500:] if result.stderr else '',
    }

def main():
    parser = argparse.ArgumentParser(description='DCT Experiment Runner')
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B', help='HuggingFace model name')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--experiments', default='all', help='Comma-separated experiment keys (a,b,c,...) or "all"')
    parser.add_argument('--output', default='results', help='Base output directory')
    parser.add_argument('--parallel', action='store_true', help='Run experiments in parallel')
    args = parser.parse_args()

    project_dir = Path(__file__).parent
    os.chdir(project_dir)

    if args.experiments == 'all':
        exp_keys = list(EXPERIMENTS.keys())
    else:
        exp_keys = [k.strip() for k in args.experiments.split(',')]

    output_base = os.path.join(str(project_dir), args.output)
    os.makedirs(output_base, exist_ok=True)

    print(f"DCT Quantization Research - Experiment Runner")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Experiments: {', '.join(exp_keys)}")
    print(f"Output: {output_base}")
    print(f"Parallel: {args.parallel}")

    results = []

    if args.parallel:
        with ProcessPoolExecutor(max_workers=min(4, len(exp_keys))) as executor:
            futures = {}
            for key in exp_keys:
                script = str(project_dir / EXPERIMENTS[key][1])
                future = executor.submit(run_experiment, key, script, args.model, args.device, output_base)
                futures[future] = key

            for future in as_completed(futures):
                results.append(future.result())
    else:
        for key in exp_keys:
            script = str(project_dir / EXPERIMENTS[key][1])
            result = run_experiment(key, script, args.model, args.device, output_base)
            results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: x['experiment']):
        status_icon = 'OK' if r['status'] == 'SUCCESS' else 'FAIL'
        print(f"  [{status_icon}] Exp {r['experiment'].upper()}: {r['name']} ({r['elapsed_seconds']:.1f}s)")

    summary_path = os.path.join(output_base, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")

if __name__ == '__main__':
    main()
