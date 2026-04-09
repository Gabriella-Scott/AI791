import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Configuration to match your code
ITERATIONS = [100, 200, 400, 600, 800, 1000]
DIMENSIONS = [5, 10, 15, 20, 25, 50, 100, 150, 200]
TRIALS = 5

# Create output directory
output_dir = Path("plots_output")
output_dir.mkdir(exist_ok=True)

# Set plot style to match
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

"""
Parse PSO results from text file.
    
Returns:
    convergence_data: dict[(dim, max_iter)] = [list of convergence curves]
    fitness_data: dict[(dim, max_iter)] = final average fitness
    algorithm_name: extracted algorithm name
"""


def parse_results_file(filepath):
    convergence_data = {}
    fitness_data = {}
    algorithm_name = None

    with open(filepath, 'r') as f:
        content = f.read()

    # Extract algorithm name from first line
    first_line = content.split('\n')[0]
    if 'Testing' in first_line:
        algorithm_name = first_line.replace(
            'Testing', '').replace('PSO', '').strip()

    # Split into blocks using separator
    blocks = content.split('-' * 50)

    for block in blocks:
        if not block.strip():
            continue

        lines = block.strip().split('\n')

        # Find the benchmark line
        benchmark_line = None
        avg_fitness_line = None
        convergence_line = None

        for line in lines:
            if line.startswith('Benchmark:'):
                benchmark_line = line
            elif line.startswith('Average Fitness:'):
                avg_fitness_line = line
            elif line.startswith('Convergence Curves:'):
                convergence_line = line

        if not benchmark_line or not avg_fitness_line or not convergence_line:
            continue

        # Extract dimension and iterations
        match = re.search(
            r'Dimension:\s*(\d+),\s*Iterations:\s*(\d+)', benchmark_line)
        if not match:
            continue

        dim = int(match.group(1))
        max_iter = int(match.group(2))

        # Extract average fitness
        avg_fitness = float(avg_fitness_line.split(
            'Average Fitness:')[1].strip())

        # Extract convergence curve
        # Remove "Convergence Curves: " prefix and extract the list
        conv_str = convergence_line.split('Convergence Curves:')[1].strip()

        # Remove np.float64() wrappers using regex
        conv_str = re.sub(
            r'np\.float64\(([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)', r'\1', conv_str)

        # Now parse the cleaned list
        try:
            # Safe since we control the format
            convergence_curve = eval(conv_str)

            # Store in expected format (list of trials, but we only have one)
            key = (dim, max_iter)
            # Wrap in list for consistency
            convergence_data[key] = [convergence_curve]
            fitness_data[key] = avg_fitness

        except Exception as e:
            print(
                f"Error parsing convergence curve for D={dim}, iter={max_iter}: {e}")
            continue

    return convergence_data, fitness_data, algorithm_name


"""
Create 2x3 subplot showing convergence curves for each dimension
Fixed at maximum iterations for fair comparison
"""


def plot_convergence_by_dimension(convergence_data, benchmark_name, algorithm_name, file_path=None):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    fixed_iterations = 1000  # Use max iterations for comparison

    for idx, dim in enumerate(DIMENSIONS):
        ax = axes[idx]
        key = (dim, fixed_iterations)

        if key in convergence_data:
            curves = convergence_data[key]
            avg_curve = np.mean(curves, axis=0)
            iterations = np.arange(len(avg_curve))

            # Simple line plot
            ax.plot(iterations, avg_curve, 'k-', linewidth=1.5)

            ax.set_xlabel('Iteration')
            ax.set_ylabel('Fitness')
            ax.set_title(f'D = {dim}', fontsize=11)
            ax.set_yscale('log')
            ax.grid(True)
            ax.tick_params(labelsize=9)

    fig.suptitle(f'{benchmark_name.upper()} Function - {algorithm_name}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    filename = f'{algorithm_name}_{benchmark_name}_convergence.png'
    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        filename = file_path
    else:
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: {filename}")


def plot_multiple_curves(curve_data, benchmark_name, title, x_label, y_label, algorithm_names=None, file_path=None):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    fixed_iterations = 1000  # Use max iterations for comparison
    colors = ["blue", "red", "green", "purple", "orange", "cyan", "pink", "black"]
    
    for idx, dim in enumerate(DIMENSIONS):
        ax = axes[idx]
        key = (dim, fixed_iterations)
        
        c = 0
        for algo_curve in curve_data:
            if key in algo_curve:
                curves = algo_curve[key]
                avg_curve = np.mean(curves, axis=0)
                iterations = np.arange(len(avg_curve))
                # Simple line plot with label only for first subplot (for legend)
                label = algorithm_names[c] if algorithm_names and idx == 0 else None
                ax.plot(iterations, avg_curve, linewidth=1.5, color=colors[c], label=label)
            c += 1

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'D = {dim}', fontsize=11)
        ax.set_yscale('log')
        ax.grid(True)
        ax.tick_params(labelsize=9)
        
        # Add legend only to the first subplot
        if idx == 0 and algorithm_names:
            ax.legend(loc='best', fontsize=8)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    filename = f'{title}_{benchmark_name}_convergence.png'
    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        filename = file_path
    else:
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: {filename}")


def plot_fitness_comparison(fitness_data_list, benchmark_name, title, x_label, y_label, algorithm_names=None, file_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    fixed_iterations = 1000  # Use max iterations for comparison
    colors = ["blue", "red", "green", "purple", "orange", "cyan", "pink", "black"]
    markers = ['o', 's', '^', 'v', 'D', '*', 'p', 'h']
    
    for idx, fitness_data in enumerate(fitness_data_list):
        dims = []
        fitnesses = []
        
        for dim in DIMENSIONS:
            key = (dim, fixed_iterations)
            if key in fitness_data:
                dims.append(dim)
                fitnesses.append(fitness_data[key])
        
        if dims:
            label = algorithm_names[idx] if algorithm_names else f'Algorithm {idx+1}'
            ax.plot(dims, fitnesses,
                    marker=markers[idx % len(markers)], markersize=6,
                    label=label,
                    linewidth=1.5, color=colors[idx % len(colors)])
    
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend(loc='best', fontsize=9)
    ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    
    filename = f'{title}_{benchmark_name}_fitness.png'
    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        filename = file_path
    else:
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: {filename}")


"""
Plot how final fitness scales with dimension
Shows performance degradation as dimension increases
"""
def plot_final_fitness_vs_dimension(fitness_data, benchmark_name, algorithm_name, file_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot for each iteration count
    markers = ['o', 's', '^', 'v', 'D', '*']

    for idx, max_iter in enumerate(ITERATIONS):
        dims = []
        fitnesses = []

        for dim in DIMENSIONS:
            key = (dim, max_iter)
            if key in fitness_data:
                dims.append(dim)
                fitnesses.append(fitness_data[key])

        if dims:
            ax.plot(dims, fitnesses,
                    marker=markers[idx], markersize=6,
                    label=f'{max_iter} iter',
                    linewidth=1.5, color=f'C{idx}')

    ax.set_xlabel('Dimension', fontsize=11)
    ax.set_ylabel('Final Fitness (Mean)', fontsize=11)
    ax.set_title(f'{benchmark_name.upper()} Function - {algorithm_name}',
                 fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.tick_params(labelsize=9)

    plt.tight_layout()

    filename = f'{algorithm_name}_{benchmark_name}_scalability.png'
    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        filename = file_path
    else:
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: {filename}")


"""
Compare convergence speed across different dimensions on a single plot
Fixed at maximum iterations
"""


def plot_convergence_comparison(convergence_data, benchmark_name, algorithm_name):
    fixed_iterations = 1000

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(DIMENSIONS)))

    for idx, dim in enumerate(DIMENSIONS):
        key = (dim, fixed_iterations)
        if key in convergence_data:
            curves = convergence_data[key]
            avg_curve = np.mean(curves, axis=0)

            iterations = np.arange(len(avg_curve))

            ax.plot(iterations, avg_curve,
                    label=f'D = {dim}',
                    color=colors[idx],
                    linewidth=1.5)

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Fitness', fontsize=11)
    ax.set_title(f'{benchmark_name.upper()} Function - {algorithm_name} (Max Iter = {fixed_iterations})',
                 fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.tick_params(labelsize=9)

    plt.tight_layout()

    filename = f'{algorithm_name}_{benchmark_name}_comparison.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: {filename}")


"""
Create all plots for a given algorithm
"""


def create_all_plots_for_algorithm(convergence_data, fitness_data, benchmark_name, algorithm_name):
    print(f"\nGenerating plots for {algorithm_name} on {benchmark_name}...")

    plot_convergence_by_dimension(
        convergence_data, benchmark_name, algorithm_name)
    plot_final_fitness_vs_dimension(
        fitness_data, benchmark_name, algorithm_name)
    plot_convergence_comparison(
        convergence_data, benchmark_name, algorithm_name)


if __name__ == "__main__":
    print("\nGenerating plots from PSO benchmark results...")
    print("-" * 60)

    # Parse Standard PSO results
    print("\nParsing Standard PSO results from std_pso.txt...")
    std_conv_data, std_fit_data, std_algo_name = parse_results_file(
        'std_pso.txt')
    if not std_algo_name:
        std_algo_name = "StandardPSO"
    print(f"Found {len(std_conv_data)} data points for {std_algo_name}")

    # Parse Stochastic Scaling PSO results
    print("\nParsing Stochastic Scaling PSO results from stochastic_pso.txt...")
    stoch_conv_data, stoch_fit_data, stoch_algo_name = parse_results_file(
        'stochastic_pso.txt')
    if not stoch_algo_name:
        stoch_algo_name = "StochasticScalingPSO"
    print(f"Found {len(stoch_conv_data)} data points for {stoch_algo_name}")

    # Generate plots for Standard PSO
    create_all_plots_for_algorithm(
        std_conv_data,
        std_fit_data,
        'sphere',
        std_algo_name
    )

    # Generate plots for Stochastic Scaling PSO
    create_all_plots_for_algorithm(
        stoch_conv_data,
        stoch_fit_data,
        'sphere',
        stoch_algo_name
    )
