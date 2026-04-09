
import numpy as np
import sys
from typing import List, Tuple
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
                 seed: int = None):
        
        # Initialize DE algorithm
        self.func = func
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.max_fes = max_fes
        self.dimensions = func.dimensions
        self.convergence_curve = []
        
        if seed is not None:
            np.random.seed(seed)
            
        self.population = None
        self.fitness = None
        self.best_individual = None
        self.best_fitness = np.inf
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
        
    def mutate(self, idx: int) -> np.ndarray:
        """
        DE/rand/1 mutation: v = x_r1 + F * (x_r2 - x_r3)
        """
        candidates = [i for i in range(self.pop_size) if i != idx]
        r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
        mutant = self.population[r1] + self.F * (self.population[r2] - self.population[r3])
        return self._repair(mutant)
        
    # Binomial crossover
    def crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        trial = target.copy()
        jrand = np.random.randint(0, self.dimensions)
        for j in range(self.dimensions):
            if np.random.rand() < self.CR or j == jrand:
                trial[j] = mutant[j]
        return trial
    
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
                
                # Crossover
                trial = self.crossover(self.population[i], mutant)
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

TRIALS = 5
POPULATION_SIZE = 50
DIMENSIONS = 10
MAX_FES = 10000
DE_OPTIONS = {
    'DE/rand/1/bin': DifferentialEvolution,
    # Future: Add more DE variants here
}
BENCHMARKS = ["sphere", "ackley", "griewank", "rastrigin", "weierstrass"]
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

def de_stats(benchmark_name: str, class_name: object):
    benchmark_function = BenchmarkFunction(benchmark_name, DIMENSIONS)
    all_fitnesses = []
    convergence_curves = []
    
    for trial in range(TRIALS):
        de = class_name(benchmark_function, pop_size=POPULATION_SIZE, max_fes=MAX_FES)
        best_individual, best_fitness = de.run()
        convergence_curves.append(de.convergence_curve)
        all_fitnesses.append(best_fitness)
        print(f"Trial {trial + 1}: Best Fitness = {best_fitness:.6f}")
    print(f"Average Fitness: {np.mean(all_fitnesses):.6f}")
    print("Min Fitness: {:.6f}".format(np.min(all_fitnesses)))
    print("Max Fitness: {:.6f}".format(np.max(all_fitnesses)))
    print(f"Standard Deviation: {np.std(all_fitnesses):.6f}")
    avg_curve = get_avg_convergence_curve(convergence_curves)
    print(f"Average Convergence Curve: {avg_curve}")

def run_all_de_stats(de_option=None):
    for benchmark in BENCHMARKS:
        for de_variant, de_class in DE_OPTIONS.items():
            if de_option is None or de_option == de_variant:
                print(f"\nRunning {de_variant} on {benchmark} benchmark:")
                de_stats(benchmark, de_class)

def main():
    if len(sys.argv) > 1:
        de_option = sys.argv[1]
        if de_option not in DE_OPTIONS:
            print(f"Unknown DE option: {de_option}")
            print(f"Available options: {list(DE_OPTIONS.keys())}")
            return
        run_all_de_stats(de_option)
    else:
        run_all_de_stats()

if __name__ == "__main__":
    main()
    