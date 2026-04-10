
import numpy as np
import sys
from typing import List, Tuple, Dict
from benchmark_functions import BENCHMARK_FUNCTIONS

"""Wrapper for benchmark functions"""
class BenchmarkFunction:
    def __init__(self, name: str, dimensions: int):
        if name not in BENCHMARK_FUNCTIONS:
            raise ValueError(f"Unknown function: {name}")
        
        self.name = name
        self.dimensions = dimensions
        self.func, bounds = BENCHMARK_FUNCTIONS[name]
        self.lower_bound, self.upper_bound = bounds
             
    #Evaluate the function at point x
    def evaluate(self, x: np.ndarray) -> float:
        return self.func(x)

"""Simple DE/rand/1/bin implementation"""
class DifferentialEvolution:
    def __init__(self, 
                 func: BenchmarkFunction,
                 pop_size: int = 50,
                 F: float = 0.8,
                 CR: float = 0.9,
                 max_fes: int = 10000,
                 seed: int = None,
                 crossover_type: str = 'bin'):
        
        # Initialize DE algorithm
        self.func = func
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.max_fes = max_fes
        self.dimensions = func.dimensions
        self.crossover_type = crossover_type
        
        
        if seed is not None:
            np.random.seed(seed)
            
        self.population = None
        self.fitness = None
        self.best_individual = None
        self.best_fitness = np.inf
        self.convergence_curve = []
        self.fes = 0

    # Initialize population uniformly within bounds    
    def initialize_population(self):
        lower = self.func.lower_bound
        upper = self.func.upper_bound
        self.population = np.random.uniform(
            lower, upper, (self.pop_size, self.dimensions)
        )
        
        # Evaluate initial population
        self.fitness = np.array([self._evaluate(ind) for ind in self.population])
        
        # Track best
        best_idx = np.argmin(self.fitness)
        self.best_fitness = self.fitness[best_idx]
        self.best_individual = self.population[best_idx].copy()
        
    # Evaluate individual and increment FES counter    
    def _evaluate(self, individual: np.ndarray) -> float:
        self.fes += 1
        return self.func.evaluate(individual)
        
    # Clip individual to bounds    
    def _repair(self, individual: np.ndarray) -> np.ndarray:
        return np.clip(individual, self.func.lower_bound, self.func.upper_bound)

    # DE/rand/2 mutation strategy: v_i = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5) 
    def mutate(self, idx: int) -> np.ndarray:
        # Select 5 distinct random indices different from idx
        candidates = [i for i in range(self.pop_size) if i != idx]
        r1, r2, r3, r4, r5 = np.random.choice(candidates, 5, replace=False)
        
        # Create mutant vector using DE/rand/2 strategy
        mutant = (self.population[r1] + 
                  self.F * (self.population[r2] - self.population[r3]) +
                  self.F * (self.population[r4] - self.population[r5]))
        
        return mutant
        
    # Binomial crossover
    def crossover_bin(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        trial = target.copy()
        jrand = np.random.randint(0, self.dimensions)

        for j in range(self.dimensions):
            if np.random.rand() < self.CR or j == jrand:
                trial[j] = mutant[j]

        return trial
    
    # Arithmetic crossover
    def crossover_arith(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        # Sample gamma_ij from uniform distribution [0, 1] for each dimension
        gamma = np.random.uniform(0, 1, self.dimensions)
        # Create offspring using arithmetic crossover
        offspring = (1 - gamma) * target + gamma * mutant
        
        return offspring
    
    # Unimodal Normal Distribution Crossover (UNDX)
    def crossover_undx(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        candidates = list(range(self.pop_size))
        
        # Find and exclude target index
        target_idx = None
        for i in range(self.pop_size):
            if np.array_equal(self.population[i], target):
                target_idx = i
                break
                
        if target_idx is not None:
            candidates.remove(target_idx)
            
        third_parent_idx = np.random.choice(candidates)
        third_parent = self.population[third_parent_idx]
        
        # For nu = 3 parents, use nu - 1 = 2 parents to form main direction
        # Use target and mutant as the first two parents
        parent1 = target
        parent2 = mutant
        parent3 = third_parent  # The third parent for orthogonal component
        
        # Calculate center of mass of first nu-1 = 2 parents
        x_bar = (parent1 + parent2) / 2.0
        
        # Direction vector from center to parent1
        d = parent1 - x_bar
        d_norm = np.linalg.norm(d)
        
        # Direction cosine (unit vector)
        if d_norm > 1e-10:
            e = d / d_norm
        else:
            # If parents are identical, use random direction
            e = np.random.randn(self.dimensions)
            e = e / np.linalg.norm(e)
            d_norm = 0.0
        
        # Orthogonal distance of third parent to the main direction
        # Project parent3 - x_bar onto the direction e
        diff = parent3 - x_bar
        projection = np.dot(diff, e) * e
        orthogonal = diff - projection
        delta = np.linalg.norm(orthogonal)
        
        # Standard deviations
        sigma1 = 1.0
        # For ns = dimensions
        sigma2 = np.sqrt(0.35 / max(1, self.dimensions - 1))
        
        # Main direction component
        main_component = np.random.randn() * sigma1 * d_norm * e
        
        # Orthogonal components - create orthonormal basis perpendicular to e
        orthogonal_component = np.zeros(self.dimensions)
        
        # For simplicity in high dimensions, sample random orthogonal directions
        for _ in range(self.dimensions - 1):
            # Generate random vector
            rand_vec = np.random.randn(self.dimensions)
            # Make it orthogonal to main direction e
            rand_vec = rand_vec - np.dot(rand_vec, e) * e
            # Normalize
            norm = np.linalg.norm(rand_vec)
            if norm > 1e-10:
                rand_vec = rand_vec / norm
                # Add Gaussian noise in this orthogonal direction
                orthogonal_component += np.random.randn() * sigma2 * delta * rand_vec
        
        # Create offspring
        offspring = x_bar + main_component + orthogonal_component
        
        return offspring

    # Simplex Crossover (SPX)
    def crossover_spx(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        # Select third parent (different from target)
        candidates = list(range(self.pop_size))
        
        # Find and exclude target index
        target_idx = None
        for i in range(self.pop_size):
            if np.array_equal(self.population[i], target):
                target_idx = i
                break
                
        if target_idx is not None:
            candidates.remove(target_idx)
            
        third_parent_idx = np.random.choice(candidates)
        third_parent = self.population[third_parent_idx]
        
        parents = np.array([target, mutant, third_parent])
        nµ = len(parents)
        
        # Calculate center of mass (center of gravity)
        G = np.mean(parents, axis=0)
        
        # Expansion coefficient (gamma in textbook)
        epsilon = 0.5
        
        # Generate random weights that sum to 1 for uniform sampling from simplex
        r = np.random.uniform(0, 1, nµ)
        r = r / np.sum(r)
        
        # Create offspring using SPX formula from textbook
        offspring = np.zeros(self.dimensions)
        for i, parent in enumerate(parents):
            offspring += r[i] * (G + (1 + epsilon) * (parent - G))
            
        return offspring

    # Parent Centric Crossover (PCX)
    def crossover_pcx(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        # PCX uses 3 parents: target, mutant, and a third parent
        candidates = list(range(self.pop_size))
        
        # Find and exclude target index
        target_idx = None
        for i in range(self.pop_size):
            if np.array_equal(self.population[i], target):
                target_idx = i
                break
                
        if target_idx is not None:
            candidates.remove(target_idx)
            
        third_parent_idx = np.random.choice(candidates)
        third_parent = self.population[third_parent_idx]
        
        # Collect all nµ = 3 parents
        parents = [target, mutant, third_parent]
        nµ = len(parents)
        
        # Calculate center of mass (mean) of all parents
        x_bar = np.mean(parents, axis=0)
        
        # Randomly select one parent around which to generate offspring
        selected_idx = np.random.randint(0, nµ)
        selected_parent = parents[selected_idx]
        
        # Calculate direction vector from center to selected parent
        d = selected_parent - x_bar
        d_norm = np.linalg.norm(d)
        
        # Normalize direction vector
        if d_norm > 1e-10:
            d_unit = d / d_norm
        else:
            # If selected parent is at center, use random direction
            d_unit = np.random.randn(self.dimensions)
            d_unit = d_unit / np.linalg.norm(d_unit)
            d_norm = 0.0
        
        # Calculate perpendicular distances from other parents to the line d
        perpendicular_distances = []
        for idx, parent in enumerate(parents):
            if idx != selected_idx:
                # Vector from center to this parent
                diff = parent - x_bar
                # Project onto direction d to get parallel component
                projection = np.dot(diff, d_unit) * d_unit
                # Orthogonal component
                orthogonal = diff - projection
                # Distance
                delta_l = np.linalg.norm(orthogonal)
                perpendicular_distances.append(delta_l)
        
        # Average perpendicular distance (equation 9.17)
        delta_bar = np.mean(perpendicular_distances)
        
        # Variance parameters (commonly used values for PCX)
        sigma1 = 0.1  # For main direction
        sigma2 = 0.1  # For orthogonal directions
        
        # Generate offspring according to equation (9.18)
        # x̃(t) = x(t) + N(0,σ₁²)|d(t)| + Σ N(0,σ₂²)δ̄e_l(t)
        
        # Main direction component with Gaussian noise
        main_component = np.random.randn() * sigma1 * d_norm * d_unit
        
        # Orthogonal components - create orthonormal basis perpendicular to d
        orthogonal_component = np.zeros(self.dimensions)
        
        # Generate nµ-1 = 2 orthogonal directions
        for _ in range(nµ - 1):
            # Random vector
            rand_vec = np.random.randn(self.dimensions)
            # Make orthogonal to main direction
            rand_vec = rand_vec - np.dot(rand_vec, d_unit) * d_unit
            # Normalize
            norm = np.linalg.norm(rand_vec)
            if norm > 1e-10:
                rand_vec = rand_vec / norm
                # Add Gaussian noise in this orthogonal direction
                orthogonal_component += np.random.randn() * sigma2 * delta_bar * rand_vec
        
        # Create offspring (equation 9.18)
        offspring = selected_parent + main_component + orthogonal_component
        
        return offspring
    

    # Run the DE algorithm    
    def run(self) -> Tuple[np.ndarray, float]:
        self.initialize_population()
        self.convergence_curve = [self.best_fitness]
        while self.fes < self.max_fes:
            for i in range(self.pop_size):
                if self.fes >= self.max_fes:
                    break
                    
                # Mutation
                mutant = self.mutate(i)
                
                # Crossover select based on crossover type
                if self.crossover_type == 'bin':
                    trial = self.crossover_bin(self.population[i], mutant)
                elif self.crossover_type == 'ax':
                    trial = self.crossover_arith(self.population[i], mutant)
                elif self.crossover_type == 'spx':
                    trial = self.crossover_spx(self.population[i], mutant)
                elif self.crossover_type == 'undx':
                    trial = self.crossover_undx(self.population[i], mutant)
                elif self.crossover_type == 'pcx':
                    trial = self.crossover_pcx(self.population[i], mutant)
                else:
                    raise ValueError(f"Unknown crossover type: {self.crossover_type}")
                    
                trial = self._repair(trial)
                
                # Selection
                trial_fitness = self._evaluate(trial)
                if trial_fitness <= self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_individual = trial.copy()
            self.convergence_curve.append(self.best_fitness)
        return self.best_individual, self.best_fitness

TRIALS = 20
POPULATION_SIZE = 50
DIMENSIONS = 10
MAX_FES = 10000
# Mapping from short names to (full name, crossover type)
DE_VARIANT_MAP = {
    'bin': ('DE/rand/2/bin', 'bin'),
    'ax': ('DE/rand/2/AX', 'ax'),
    'spx': ('DE/rand/2/SPX', 'spx'),
    'undx': ('DE/rand/2/UNDX', 'undx'),
    'pcx': ('DE/rand/2/PCX', 'pcx')
}

DE_OPTIONS = {
    'DE/rand/2/bin': 'bin',
    'DE/rand/2/AX': 'ax',
    'DE/rand/2/SPX': 'spx',
    'DE/rand/2/UNDX': 'undx',
    'DE/rand/2/PCX': 'pcx'
}

BENCHMARKS = ["sphere", "ackley", "griewank", "rastrigin", "rosenbrock"]
# TODO: Think about differing population size and benchmark dimensions

def get_avg_convergence_curve(convergence_curves: List[List[float]]) -> List[float]:
    n = len(convergence_curves) #num trials
    cols = len(convergence_curves[0]) # num iterations
    avg_curve = []
    for col in range(cols):
        total = 0.0
        for j in range(n):
            total += convergence_curves[j][col] 
        avg_value = total / n
        avg_curve.append(avg_value)
    return avg_curve

def de_stats(benchmark_name: str, crossover_type: str):
    benchmark_function = BenchmarkFunction(benchmark_name, DIMENSIONS)
    all_fitnesses = []
    convergence_curves = []
    
    for trial in range(TRIALS):
        de = DifferentialEvolution(benchmark_function, pop_size=POPULATION_SIZE, 
                                   max_fes=MAX_FES, crossover_type=crossover_type)
        best_individual, best_fitness = de.run()
        convergence_curves.append(de.convergence_curve)
        all_fitnesses.append(best_fitness)
        print(f"Trial {trial + 1}: Best Fitness = {best_fitness:.6f}")

    print(f"Average Fitness: {np.mean(all_fitnesses):.6f}")
    print("Min Fitness: {:.6f}".format(np.min(all_fitnesses)))
    print("Max Fitness: {:.6f}".format(np.max(all_fitnesses)))
    print(f"Standard Deviation: {np.std(all_fitnesses):.6f}")
    avg_curve = get_avg_convergence_curve(convergence_curves)
    
    # Return results for plotting
    return {
        'fitnesses': all_fitnesses,
        'avg_convergence': avg_curve,
        'mean': np.mean(all_fitnesses),
        'std': np.std(all_fitnesses),
        'min': np.min(all_fitnesses),
        'max': np.max(all_fitnesses)
    }

def run_all_de_stats(de_option=None):
    results = {}
    for benchmark in BENCHMARKS:
        results[benchmark] = {}
        for de_variant, crossover_type in DE_OPTIONS.items():
            if de_option is None or de_option == de_variant:
                print(f"\nRunning {de_variant} on {benchmark} benchmark:")
                results[benchmark][de_variant] = de_stats(benchmark, crossover_type)
    
    # Generate plots if running all algorithms
    if de_option is None or len([k for k in DE_OPTIONS.keys() if de_option == k]) > 0:
        print("\n" + "="*80)
        print("Generating plots and statistics...")
        print("="*80)
        
        # Import plotting functions
        from plot_results import plot_convergence_curves, print_statistics
        
        plot_convergence_curves(results, BENCHMARKS)
        print_statistics(results, BENCHMARKS)
        
        print("\n" + "="*80)
        print("All plots saved successfully!")
        print("="*80)


def main():
    if len(sys.argv) > 1:
        de_option = sys.argv[1]
        
        if de_option in DE_VARIANT_MAP:
            full_name, _ = DE_VARIANT_MAP[de_option]
            run_all_de_stats(full_name)
        elif de_option in DE_OPTIONS:
            run_all_de_stats(de_option)
        else:
            print(f"Unknown DE option: {de_option}")
            print(f"Available options: {list(DE_VARIANT_MAP.keys())} or {list(DE_OPTIONS.keys())}")
            return
    else:
        run_all_de_stats()

if __name__ == "__main__":
    main()
    