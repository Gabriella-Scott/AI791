"""
Quick results collection - Essential tables only for 9-page report.
Outputs: Main Results + Statistical Tests + Configuration

Usage:
    python quick_results.py > results.txt
    python quick_results.py --benchmark sphere  # Test on single benchmark first
"""

import numpy as np
from typing import Dict, List
import sys
from pso_implementation import (
    StandardPSO, StochasticScalingPSO, SubspaceInitPSO, HybridPSO,
    get_benchmark_function
)
from scipy import stats

# Assignment configuration
DIMENSIONS = [5, 10, 15, 20, 25, 50, 100, 150, 200]
BENCHMARKS = ['sphere', 'ackley', 'griewank', 'rastrigin', 'weierstrass']
RUNS = 5
SWARM_SIZE = 10
MAX_ITER = 1000

ALGORITHMS = {
    'Standard': StandardPSO,
    'Stochastic': StochasticScalingPSO,
    'Subspace': SubspaceInitPSO,
    'Hybrid': HybridPSO
}

def run_experiment(algo_class, benchmark: str, dim: int, seed: int):
    """Run single PSO experiment."""
    obj_func, bounds = get_benchmark_function(benchmark)
    
    kwargs = {
        'objective_func': obj_func,
        'dim': dim,
        'bounds': bounds,
        'swarm_size': SWARM_SIZE,
        'max_iterations': MAX_ITER,
        'w': 0.729844,
        'c1': 1.49618,
        'c2': 1.49618,
        'random_seed': 42 + seed
    }
    
    if algo_class == StochasticScalingPSO:
        kwargs['group_count'] = 10
    elif algo_class == SubspaceInitPSO:
        kwargs['seed_set_size'] = 1
        kwargs['pos_margin_ratio'] = 0.1
    elif algo_class == HybridPSO:
        kwargs['seed_set_size'] = 1
        kwargs['pos_margin_ratio'] = 0.1
        kwargs['group_count_start'] = 1
        kwargs['group_count_end'] = 25
    
    pso = algo_class(**kwargs)
    _, fitness, _, _, _ = pso.optimize()
    return fitness


def collect_data(benchmark_name=None):
    """Collect all experimental data."""
    results = {}
    benchmarks = [benchmark_name] if benchmark_name else BENCHMARKS
    
    total = len(benchmarks) * len(DIMENSIONS) * len(ALGORITHMS) * RUNS
    count = 0
    
    print(f"Running {total} experiments...", file=sys.stderr)
    
    for benchmark in benchmarks:
        results[benchmark] = {}
        for dim in DIMENSIONS:
            results[benchmark][dim] = {}
            for algo_name, algo_class in ALGORITHMS.items():
                fitness_values = []
                
                for run in range(RUNS):
                    count += 1
                    if count % 30 == 0:
                        print(f"Progress: {count}/{total} ({100*count/total:.1f}%)", 
                              file=sys.stderr)
                    
                    fitness = run_experiment(algo_class, benchmark, dim, run)
                    fitness_values.append(fitness)
                
                results[benchmark][dim][algo_name] = fitness_values
    
    print("Done!", file=sys.stderr)
    return results


def print_main_table(results):
    """Table 1: Main Results (Mean ± Std)"""
    print("\n" + "="*120)
    print("TABLE 1: ALGORITHM PERFORMANCE COMPARISON")
    print("Mean ± Standard Deviation (5 runs)")
    print("="*120)
    
    for benchmark in results.keys():
        print(f"\n{benchmark.upper()} FUNCTION")
        print("-"*120)
        print(f"{'Dim':<8}", end="")
        for algo in ALGORITHMS.keys():
            print(f"{algo:<28}", end="")
        print()
        print("-"*120)
        
        for dim in DIMENSIONS:
            print(f"{dim:<8}", end="")
            for algo in ALGORITHMS.keys():
                vals = results[benchmark][dim][algo]
                mean = np.mean(vals)
                std = np.std(vals, ddof=1)
                print(f"{mean:.6e} ± {std:.2e}   ", end="")
            print()
        print("-"*120)


def print_statistical_table(results):
    """Table 2: Statistical Tests (Hybrid vs others)"""
    print("\n" + "="*100)
    print("TABLE 2: STATISTICAL SIGNIFICANCE (Wilcoxon signed-rank test)")
    print("Hybrid PSO vs other algorithms | ** p<0.01, * p<0.05, - not significant")
    print("="*100)
    
    for benchmark in results.keys():
        print(f"\n{benchmark.upper()} FUNCTION")
        print("-"*100)
        print(f"{'Dim':<8}{'vs Standard':<30}{'vs Stochastic':<30}{'vs Subspace':<30}")
        print("-"*100)
        
        for dim in DIMENSIONS:
            hybrid = results[benchmark][dim]['Hybrid']
            print(f"{dim:<8}", end="")
            
            for algo in ['Standard', 'Stochastic', 'Subspace']:
                other = results[benchmark][dim][algo]
                _, p = stats.wilcoxon(hybrid, other, alternative='two-sided')
                
                sig = '**' if p < 0.01 else ('*' if p < 0.05 else '-')
                print(f"{p:.6f} {sig:<23}", end="")
            print()
        print("-"*100)


def print_config_table():
    """Configuration table for Empirical Procedure section."""
    print("\n" + "="*80)
    print("EXPERIMENTAL CONFIGURATION")
    print("="*80)
    print(f"{'Parameter':<45}{'Value':<35}")
    print("-"*80)
    print(f"{'Swarm Size':<45}{SWARM_SIZE:<35}")
    print(f"{'Maximum Iterations':<45}{MAX_ITER:<35}")
    print(f"{'Independent Runs per Configuration':<45}{RUNS:<35}")
    print(f"{'Inertia Weight (w)':<45}{'0.729844':<35}")
    print(f"{'Cognitive Coefficient (c₁)':<45}{'1.49618':<35}")
    print(f"{'Social Coefficient (c₂)':<45}{'1.49618':<35}")
    print(f"{'Dimensions Tested':<45}{'5, 10, 15, 20, 25, 50':<35}")
    print(f"{'Stochastic Scaling Group Count':<45}{'10':<35}")
    print(f"{'Subspace Seed Set Size':<45}{'1':<35}")
    print(f"{'Position Margin Ratio':<45}{'0.1':<35}")
    print(f"{'Hybrid Group Schedule':<45}{'1 → 25 (linearly increasing)':<35}")
    print(f"{'Base Random Seed':<45}{'42':<35}")
    print("="*80)


def print_benchmark_table():
    """Benchmark functions table."""
    print("\n" + "="*90)
    print("BENCHMARK FUNCTIONS")
    print("="*90)
    print(f"{'Function':<15}{'Bounds':<20}{'Optimum':<15}{'Properties':<40}")
    print("-"*90)
    print(f"{'Sphere':<15}{'[-100, 100]':<20}{'f(0) = 0':<15}{'Unimodal, Separable':<40}")
    print(f"{'Ackley':<15}{'[-32, 32]':<20}{'f(0) = 0':<15}{'Multimodal, Non-separable':<40}")
    print(f"{'Rastrigin':<15}{'[-5.12, 5.12]':<20}{'f(0) = 0':<15}{'Multimodal, Separable':<40}")
    print(f"{'Griewank':<15}{'[-600, 600]':<20}{'f(0) = 0':<15}{'Multimodal, Non-separable':<40}")
    print(f"{'Weierstrass':<15}{'[-0.5, 0.5]':<20}{'f(0) = 0':<15}{'Multimodal, Non-separable':<40}")
    print("="*90)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, help='Test single benchmark')
    args = parser.parse_args()
    
    # Collect data
    results = collect_data(args.benchmark)
    
    # Print tables
    print_config_table()
    print_benchmark_table()
    print_main_table(results)
    print_statistical_table(results)


if __name__ == "__main__":
    main()