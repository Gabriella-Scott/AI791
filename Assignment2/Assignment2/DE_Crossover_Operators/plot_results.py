import numpy as np
import matplotlib.pyplot as plt
from statistical_analysis import print_statistical_analysis

# Colors for each benchmark
BENCHMARK_COLORS = {
    'sphere': '#1f77b4',
    'ackley': '#ff7f0e',
    'griewank': '#2ca02c',
    'rastrigin': '#d62728',
    'rosenbrock': '#9467bd'
}

# Colors and markers for each algorithm
ALGORITHM_COLORS = {
    'DE/rand/2/bin': '#1f77b4',
    'DE/rand/2/AX': '#ff7f0e',
    'DE/rand/2/SPX': '#2ca02c',
    'DE/rand/2/UNDX': '#d62728',
    'DE/rand/2/PCX': '#9467bd'
}

ALGORITHM_MARKERS = {
    'DE/rand/2/bin': 'o',
    'DE/rand/2/AX': 's',
    'DE/rand/2/SPX': '^',
    'DE/rand/2/UNDX': 'v',
    'DE/rand/2/PCX': 'D'
}

def plot_convergence_curves(results, benchmarks):
    """Plot convergence curves - one plot per algorithm showing all benchmarks"""
    # Get all algorithm names
    alg_names = sorted(list(results[benchmarks[0]].keys()))
    
    for alg_name in alg_names:
        plt.figure(figsize=(10, 6))
        
        for benchmark in benchmarks:
            avg_curve = results[benchmark][alg_name]['avg_convergence']
            iterations = range(len(avg_curve))
            plt.plot(iterations, avg_curve, 
                    label=benchmark.capitalize(), 
                    color=BENCHMARK_COLORS[benchmark],
                    linewidth=2)
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Fitness', fontsize=12)
        plt.title(f'{alg_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        
        # Create safe filename
        safe_name = alg_name.replace('/', '_')
        filename = f'convergence_{safe_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved {filename}")
        plt.close()



def plot_fitness_comparison(results, benchmarks):
    """Plot mean fitness comparison - bar chart with error bars for each benchmark"""
    # Get all algorithm names
    alg_names = sorted(list(results[benchmarks[0]].keys()))
    
    # Create subplots - one for each benchmark
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(16, 5))
    
    for idx, benchmark in enumerate(benchmarks):
        ax = axes[idx]
        
        means = [results[benchmark][alg]['mean'] for alg in alg_names]
        stds = [results[benchmark][alg]['std'] for alg in alg_names]
        colors = [ALGORITHM_COLORS[alg] for alg in alg_names]
        
        x_pos = np.arange(len(alg_names))
        ax.bar(x_pos, means, yerr=stds, capsize=5,
               color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Mean Fitness' if idx == 0 else '', fontsize=11)
        ax.set_title(benchmark.capitalize(), fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([alg.split('/')[-1] for alg in alg_names], 
                          rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('fitness_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved fitness_comparison.png")
    plt.close()

def plot_fitness_boxplots(results, benchmarks):
    """Plot box plots showing fitness distribution for each benchmark"""
    # Get all algorithm names
    alg_names = sorted(list(results[benchmarks[0]].keys()))
    
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(16, 5))
    
    for idx, benchmark in enumerate(benchmarks):
        ax = axes[idx]
        
        data = [results[benchmark][alg]['fitnesses'] for alg in alg_names]
        labels = [alg.split('/')[-1] for alg in alg_names]
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        # Color the boxes
        for patch, alg in zip(bp['boxes'], alg_names):
            patch.set_facecolor(ALGORITHM_COLORS[alg])
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Fitness' if idx == 0 else '', fontsize=11)
        ax.set_title(benchmark.capitalize(), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('fitness_boxplots.png', dpi=300, bbox_inches='tight')
    print("Saved fitness_boxplots.png")
    plt.close()

def calculate_additional_metrics(results, benchmarks):
    """Calculate additional performance metrics for algorithm comparison"""
    alg_names = sorted(list(results[benchmarks[0]].keys()))
    
    metrics = {}
    
    for alg_name in alg_names:
        metrics[alg_name] = {
            'median_fitnesses': {},
            'best_fitnesses': {},
            'worst_fitnesses': {},
            'cv': {},  # Coefficient of variation (std/mean)
        }
        
        for benchmark in benchmarks:
            fitnesses = results[benchmark][alg_name]['fitnesses']
            mean = results[benchmark][alg_name]['mean']
            std = results[benchmark][alg_name]['std']
            
            metrics[alg_name]['median_fitnesses'][benchmark] = np.median(fitnesses)
            metrics[alg_name]['best_fitnesses'][benchmark] = np.min(fitnesses)
            metrics[alg_name]['worst_fitnesses'][benchmark] = np.max(fitnesses)
            metrics[alg_name]['cv'][benchmark] = (std / mean) if mean > 0 else 0
    
    return metrics

def calculate_rankings(results, benchmarks):
    """Calculate algorithm rankings based on mean fitness for each benchmark"""
    alg_names = sorted(list(results[benchmarks[0]].keys()))
    rankings = {alg: [] for alg in alg_names}
    
    for benchmark in benchmarks:
        # Sort algorithms by mean fitness (lower is better)
        sorted_algs = sorted(alg_names, 
                           key=lambda x: results[benchmark][x]['mean'])
        
        # Assign ranks (1 is best)
        for rank, alg in enumerate(sorted_algs, 1):
            rankings[alg].append(rank)
    
    # Calculate average rank
    avg_rankings = {alg: np.mean(ranks) for alg, ranks in rankings.items()}
    
    return rankings, avg_rankings


def plot_mean_fitness_bars(results, benchmarks):
    """This function is no longer needed - fitness comparison handles this"""
    pass


def print_statistics(results, benchmarks):
    """Print comprehensive summary statistics"""
    print("\n" + "="*100)
    print("BASIC STATISTICS")
    print("="*100)
    
    for benchmark in benchmarks:
        print(f"\n{benchmark.upper()} Function:")
        print("-" * 100)
        print(f"{'Algorithm':<20} {'Mean':<15} {'Std':<15} {'Median':<15} {'Min':<15} {'Max':<15}")
        print("-" * 100)
        
        for alg_name in sorted(results[benchmark].keys()):
            stats = results[benchmark][alg_name]
            median = np.median(stats['fitnesses'])
            print(f"{alg_name:<20} {stats['mean']:<15.6e} {stats['std']:<15.6e} "
                  f"{median:<15.6e} {stats['min']:<15.6e} {stats['max']:<15.6e}")
    
    # Calculate and print rankings
    rankings, avg_rankings = calculate_rankings(results, benchmarks)
    
    print("\n" + "="*100)
    print("ALGORITHM RANKINGS (1=best, lower is better)")
    print("="*100)
    print(f"{'Algorithm':<20}", end='')
    for bench in benchmarks:
        print(f"{bench.capitalize():<15}", end='')
    print(f"{'Average Rank':<15}")
    print("-" * 100)
    
    for alg_name in sorted(rankings.keys()):
        print(f"{alg_name:<20}", end='')
        for rank in rankings[alg_name]:
            print(f"{rank:<15}", end='')
        print(f"{avg_rankings[alg_name]:<15.2f}")
    
    # Print best algorithm
    best_alg = min(avg_rankings, key=avg_rankings.get)
    print("\n" + "="*100)
    print(f"BEST OVERALL ALGORITHM: {best_alg} (Average Rank: {avg_rankings[best_alg]:.2f})")
    print("="*100)
    
    # Calculate coefficient of variation
    metrics = calculate_additional_metrics(results, benchmarks)
    
    print("\n" + "="*100)
    print("COEFFICIENT OF VARIATION (CV = Std/Mean, lower indicates more consistent)")
    print("="*100)
    print(f"{'Algorithm':<20}", end='')
    for bench in benchmarks:
        print(f"{bench.capitalize():<15}", end='')
    print()
    print("-" * 100)
    
    for alg_name in sorted(metrics.keys()):
        print(f"{alg_name:<20}", end='')
        for bench in benchmarks:
            cv = metrics[alg_name]['cv'][bench]
            print(f"{cv:<15.4f}", end='')
        print()

    print_statistical_analysis(results, benchmarks)

