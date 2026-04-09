"""
Implementations:
1. Std PSO
2. Stochastic scaling PSO
3. Subspace initialisation PSO
4. Hybrid PSO (Subspace + Stochastic with linearly increasing groups)
"""

import numpy as np
from typing import Callable, Optional, Tuple, List
from benchmark_functions import get_benchmark_function, BENCHMARK_FUNCTIONS
import sys

# Class to represent a single particle int the swarm
class Particle:
    def __init__(self, dim: int, bounds: Tuple[float, float]):
        self.dim = dim
        self.bounds = bounds
        self.pos = np.zeros(dim)
        self.velocity = np.zeros(dim)
        self.best_pos = np.zeros(dim)
        self.best_fitness = np.inf
        self.fitness = np.inf

    # Evaluate the fitness of the current position
    def evaluate(self, objective_func:Callable) -> float:
        self.fitness = objective_func(self.pos)
        if self.fitness < self.best_fitness: 
            self.best_fitness = self.fitness
            self.best_pos =  self.pos.copy()# Updating particles personal best

        return self.fitness
    
    def velocity_mag(self) -> float:
        return np.linalg.norm(self.velocity)
    
# Standard PSO implementation
class StandardPSO:
    def __init__(self, objective_func: Callable, dim: int, bounds: Tuple[float, float], 
                    swarm_size: int = 25, max_iterations: int = 2000,
                      w: float = 0.729844,
                    c1: float = 1.49618, c2: float = 1.49618, 
                    max_velocity_ratio: Optional[float] = None, 
                    random_seed: Optional[int] = None):
        self.objective_func = objective_func
        self.dim = dim
        self.bounds = bounds
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.w = w # weight
        self.c1 = c1 # cognitive coefficient
        self.c2 = c2 # social coefficient
        self.max_velocity_ratio = max_velocity_ratio
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialise swarm
        self.swarm: List[Particle] = []
        self.gbest_pos = np.zeros(dim)
        self.gbest_fitness = np.inf

        # Tracking
        self.convergence_curve = []
        self.velocities = []
        self.diversity = []
        self.iterations = 0

    # Initialise particle positions and velocities
    def _initialize_swarm(self):
        self.swarm = []
        lower, upper = self.bounds

        for _ in range(self.swarm_size):
            particle = Particle(self.dim, self.bounds)

            # Random initialisation of position
            particle.pos = np.random.uniform(lower, upper, self.dim)

            # Random initialisation of velocity
            particle.velocity = np.random.uniform(-(upper - lower), upper - lower, self.dim)

            # Evaluate initial position
            particle.evaluate(self.objective_func)

            #Updater global best
            if particle.fitness < self.gbest_fitness:
                self.gbest_fitness = particle.fitness
                self.gbest_pos = particle.pos.copy()
            self.swarm.append(particle)

    # Update particle velocity using PSO equation
    def _update_velocity(self, particle: Particle):
        r1 = np.random.random(self.dim)
        r2 = np.random.random(self.dim)
        
        cognitive = self.c1 * r1 * (particle.best_pos - particle.pos)
        social = self.c2 * r2 * (self.gbest_pos - particle.pos)
        
        particle.velocity = self.w * particle.velocity + cognitive + social
        
        # Apply velocity clamping if specified
        if self.max_velocity_ratio is not None:
            lower, upper = self.bounds
            max_velocity = self.max_velocity_ratio * (upper - lower)
            particle.velocity = np.clip(
                particle.velocity, -max_velocity, max_velocity
            )

    # Update particle position and enforce boundaries
    def _update_pos(self, particle: Particle):
        particle.pos = particle.pos + particle.velocity
        
        # Enforce boundaries
        lower, upper = self.bounds
        particle.pos = np.clip(particle.pos, lower, upper)

    def _swarm_diversity(self) -> float:
        x_bar = np.mean([p.pos for p in self.swarm], axis=0)

        D = 0.0
        for i in range(len(self.swarm)):
            D += np.linalg.norm(self.swarm[i].pos - x_bar)
        D /= len(self.swarm)
        return D

    # executing optimization process
    def optimize(self) -> Tuple[np.ndarray, float, List[float], List[float], List[float]]:
        self._initialize_swarm()
        self.convergence_curve = [self.gbest_fitness]
        self.diversity = [self._swarm_diversity()]
        self.velocities = [np.mean([p.velocity_mag() for p in self.swarm])]
        for iteration in range(self.max_iterations):
            for particle in self.swarm:
                # Update velocity and position
                self._update_velocity(particle)
                self._update_pos(particle)

                #evaluate new position
                particle.evaluate(self.objective_func)

                # Update global best
                if particle.fitness < self.gbest_fitness:
                    self.gbest_fitness = particle.fitness
                    self.gbest_pos = particle.pos.copy()

            self.convergence_curve.append(self.gbest_fitness)
            self.diversity.append(self._swarm_diversity())
            self.velocities.append(np.mean([p.velocity_mag() for p in self.swarm]))
        return self.gbest_pos, self.gbest_fitness, self.convergence_curve, self.diversity, self.velocities
            
"""
Stochastic Scalling PSO -> uses grouped dimenstions with stochastic scalling
"""
class StochasticScalingPSO(StandardPSO):
    def __init__(self, objective_func: Callable, dim: int, bounds: Tuple[float, float], 
                    swarm_size: int = 25, max_iterations: int = 2000,
                    w: float = 0.729844,
                    c1: float = 1.49618, c2: float = 1.49618, 
                    max_velocity_ratio: Optional[float] = None, 
                    random_seed: Optional[int] = None,
                    group_count: int = 10):    
        super().__init__(objective_func, dim, bounds, swarm_size, max_iterations, 
                        w, c1, c2, max_velocity_ratio, random_seed) 
        self.group_count = group_count
        self.dim_groups = self._create_dimension_groups()
    
    def _create_dimension_groups(self) -> List[np.ndarray]:
        indices = np.random.permutation(self.dim)
        group_size = self.dim // self.group_count
        groups = []

        for i in range(self.group_count):
            start_idx = i * group_size
            if i ==  self.group_count - 1:  # Last group takes remaining dimensions
                end_idx = self.dim
            else:
                end_idx = (i + 1) * group_size
            groups.append(indices[start_idx:end_idx])
        return groups
    
    # Update velocity using stochastic scaling - only one random group
    def update_velocity(self, particle: Particle):
        # Select one random group
        selected_group_idx = np.random.randint(0, self.group_count)
        selected_dims = self.dim_groups[selected_group_idx]
        
        # Only update selected dimensions
        r1 = np.random.random(len(selected_dims))
        r2 = np.random.random(len(selected_dims))
        
        cognitive = self.c1 * r1 * (
            particle.best_pos[selected_dims] - particle.pos[selected_dims]
        )
        social = self.c2 * r2 * (
            self.gbest_pos[selected_dims] - particle.pos[selected_dims]
        )
        
        particle.velocity[selected_dims] = (
            self.w * particle.velocity[selected_dims] + cognitive + social
        )
        
        # Apply velocity clamping if specified
        if self.max_velocity_ratio is not None:
            lower, upper = self.bounds
            max_velocity = self.max_velocity_ratio * (upper - lower)
            particle.velocity[selected_dims] = np.clip(
                particle.velocity[selected_dims], -max_velocity, max_velocity
            )

"""
Subspace Initialization PSO 
Initializes particles in a subspace around seed particles
"""
class SubspaceInitPSO(StandardPSO):
    def __init__(self, objective_func: Callable, dim: int, bounds: Tuple[float, float], 
                    swarm_size: int = 25, max_iterations: int = 2000,
                    w: float = 0.729844,
                    c1: float = 1.49618, c2: float = 1.49618, 
                    max_velocity_ratio: Optional[float] = None, 
                    seed_set_size: int = 1, pos_margin_ratio: float = 0.1,
                    pbest_margin_ratio: float = 0.1, init_zero_velocity: bool = True, 
                    random_seed: Optional[int] = None):
        super().__init__(objective_func, dim, bounds, swarm_size, max_iterations, 
                        w, c1, c2, max_velocity_ratio, random_seed)
        self.seed_set_size = seed_set_size
        self.pos_margin_ratio = pos_margin_ratio
        self.pbest_margin_ratio = pbest_margin_ratio
        self.init_zero_velocity = init_zero_velocity
    
    # Initialize using subspace initialization
    def _initialize_swarm(self):
        self.swarm = []
        lower, upper = self.bounds
        search_range = upper - lower

        # Create seed set (random initialization)
        seeds = []
        for _ in range(self.seed_set_size):
            seed_pos = np.random.uniform(lower, upper, self.dim)
            seed_fitness = self.objective_func(seed_pos)
            seeds.append((seed_pos, seed_fitness))

        # Find best seed
        best_seed_pos, best_seed_fitness = min(seeds, key=lambda x: x[1])

        # Initialize global best with best seed
        self.gbest_pos = best_seed_pos.copy()
        self.gbest_fitness = best_seed_fitness

        # Initialize swarm around best seed in subspace
        pos_margin = self.pos_margin_ratio * search_range
        pbest_margin = self.pbest_margin_ratio * search_range

        for _ in range(self.swarm_size):
            particle = Particle(self.dim, self.bounds)

            # Initialize position in subspace around best seed
            particle.pos = best_seed_pos + np.random.uniform(-pos_margin, pos_margin, self.dim)
            particle.pos = np.clip(particle.pos, lower, upper)

            # Initialize personal best in subspace
            particle.best_pos = best_seed_pos + np.random.uniform(-pbest_margin, pbest_margin, self.dim)
            particle.best_fitness = self.objective_func(particle.best_pos)

            # Initialize velocity
            if self.init_zero_velocity:
                particle.velocity = np.zeros(self.dim)
            else:
                particle.velocity = np.random.uniform(-(upper - lower), upper - lower, self.dim)

            # Evaluate initial position
            particle.evaluate(self.objective_func)

            # Update global best if necessary
            if particle.fitness < self.gbest_fitness:
                self.gbest_fitness = particle.fitness
                self.gbest_pos = particle.pos.copy()

            self.swarm.append(particle)

"""
Hybrid PSO combining subspace init + stochastic scaling 
with linearly increasing number of groups
"""
class HybridPSO(SubspaceInitPSO):
    def __init__(self, objective_func: Callable, dim: int, bounds: Tuple[float, float], 
                    swarm_size: int = 25, max_iterations: int = 2000,
                    w: float = 0.729844,
                    c1: float = 1.49618, c2: float = 1.49618, 
                    max_velocity_ratio: Optional[float] = None, 
                    seed_set_size: int = 1, pos_margin_ratio: float = 0.1,
                    pbest_margin_ratio: float = 0.0, init_zero_velocity: bool = True,
                    group_count_start: int = 1, group_count_end: int = 25,
                    random_seed: Optional[int] = None):
        super().__init__(objective_func, dim, bounds, swarm_size, max_iterations, 
                        w, c1, c2, max_velocity_ratio, seed_set_size, pos_margin_ratio,
                        pbest_margin_ratio, init_zero_velocity, random_seed)
        self.group_count_start = group_count_start
        self.group_count_end = group_count_end
        self.current_group_count = group_count_start
        self.dimension_groups = []

    # Linearly increase number of group count based on iteration
    def _update_group_count(self):
        if self.max_iterations > 1:
            progress = self.iterations / (self.max_iterations - 1)
            self.current_group_count = int(self.group_count_start + progress * (self.group_count_end - self.group_count_start))
        
        else:
            self.current_group_count = self.group_count_start
            self.current_group_count = max(1, min(self.current_group_count, self.dim))

    # Create random groups of dimensions
    def _create_dimension_groups(self) -> List[np.ndarray]:
        indices = np.random.permutation(self.dim)
        group_size = self.dim // self.current_group_count
        groups = []

        for i in range(self.current_group_count):
            start_idx = i * group_size
            if i == self.current_group_count - 1:  # Last group takes remaining dimensions
                end_idx = self.dim
            else:
                end_idx = (i + 1) * group_size
            groups.append(indices[start_idx:end_idx])
        return groups
    
    # Update velocity using stochastic scaling with current group count
    def update_velocity(self, particle):
        if len(self.dimension_groups) == 0:
            # Fallback to standard update if no groups
            super(SubspaceInitPSO, self).update_velocity(particle)
            return
        
        # select one random group
        selected_group_idx = np.random.randint(0, self.current_group_count)
        selected_dims = self.dimension_groups[selected_group_idx]

        # Only update selected dimensions
        r1 = np.random.random(len(selected_dims))
        r2 = np.random.random(len(selected_dims))

        cognitive = self.c1 * r1 * (
            particle.best_pos[selected_dims] - particle.pos[selected_dims]
        )
        social = self.c2 * r2 * (self.gbest_pos[selected_dims] - particle.pos[selected_dims])

        particle.velocity[selected_dims] = self.w * particle.velocity[selected_dims] + cognitive + social

        # Apply velocity clamping if specified
        if self.max_velocity_ratio is not None:
            lower, upper = self.bounds
            max_velocity = self.max_velocity_ratio * (upper - lower)
            particle.velocity[selected_dims] = np.clip(
                particle.velocity[selected_dims], -max_velocity, max_velocity
            )

    # Run optimization with linearly increasing groups
    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        self._initialize_swarm()
        self.convergence_curve = [self.gbest_fitness]
        self.diversity = [self._swarm_diversity()]
        self.velocities = [np.mean([p.velocity_mag() for p in self.swarm])]
        for self.iteration in range(self.max_iterations):
            # Update group count and create new groups
            self._update_group_count()
            self.dimension_groups = self._create_dimension_groups()

            for particle in self.swarm:
                # Update velocity and position
                self.update_velocity(particle)
                self._update_pos(particle)

                # Evaluate new position
                particle.evaluate(self.objective_func)

                # Update global best
                if particle.fitness < self.gbest_fitness:
                    self.gbest_fitness = particle.fitness
                    self.gbest_pos = particle.pos.copy()
            self.convergence_curve.append(self.gbest_fitness)
            self.diversity.append(self._swarm_diversity())
            self.velocities.append(np.mean([p.velocity_mag() for p in self.swarm]))

        return self.gbest_pos, self.gbest_fitness, self.convergence_curve, self.diversity, self.velocities

ITERATIONS = [100, 200, 400, 600, 800, 1000]
SWARM_SIZE = 10
DIMENSION = [5, 10, 15, 20, 25, 50, 100, 150, 200]
TRIALS = 5

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

def pso_stats(benchmark_name: str, class_name: object, output=True):
    dim = 25
    objective_function, bounds = get_benchmark_function(benchmark_name)
    convergence_data = {}
    fitness_data = {}
    diversity_data = {}
    velocity_data = {}
    
    for dim in DIMENSION:
        for max_iterations in ITERATIONS: 
            sum_fitness = 0.0
            convergence_curves = []
            diversity_curves = []
            velocity_curves = []

            for trial in range(TRIALS):
                pso = class_name(objective_function, dim, bounds, SWARM_SIZE, max_iterations, random_seed=42)
                best_pos, best_fitness, convergence_curve, diversity_curve, velocity_curve = pso.optimize()
                sum_fitness += best_fitness
                convergence_curves.append(convergence_curve)
                diversity_curves.append(diversity_curve)
                velocity_curves.append(velocity_curve)

            avg_fitness = sum_fitness / TRIALS

            if output:
                print(f"Benchmark: {benchmark_name}, Dimension: {dim}, Iterations: {max_iterations}")
                #print("Best Position:", best_pos)
                print("Average Fitness:", avg_fitness)

            avg_convergence_curve = get_avg_convergence_curve(convergence_curves)
            key = (dim, max_iterations)
            convergence_data[key] = [avg_convergence_curve]
            fitness_data[key] = avg_fitness
            diversity_data[key] = [get_avg_convergence_curve(diversity_curves)]
            velocity_data[key] = [get_avg_convergence_curve(velocity_curves)]
            if output:
                print("Convergence Curves:", avg_convergence_curve)
                print("-" * 50)
    if output:
        from plot_results import plot_convergence_by_dimension, plot_final_fitness_vs_dimension
        plot_convergence_by_dimension(
            convergence_data, benchmark_name, class_name.__name__, file_path=f"{class_name.__name__}_{benchmark_name}_convergence.png"
        )
        plot_final_fitness_vs_dimension(
            fitness_data, benchmark_name, class_name.__name__, file_path=f"{class_name.__name__}_{benchmark_name}_fitness.png"
        )
        plot_convergence_by_dimension(
            diversity_data, benchmark_name, class_name.__name__, file_path=f"{class_name.__name__}_{benchmark_name}_diversity.png"
        )
        plot_convergence_by_dimension(
            velocity_data, benchmark_name, class_name.__name__, file_path=f"{class_name.__name__}_{benchmark_name}_velocity.png"
        )
    return convergence_data, fitness_data, diversity_data, velocity_data

def run_pso_plotter(benchmark_name: str):
    data = []
    for pso_algo in PSO_ALGORITHMS.keys():
        class_name = PSO_ALGORITHMS[pso_algo]
        print("Running", pso_algo)
        convergence_data, fitness_data, diversity_data, velocity_data = pso_stats(benchmark_name, class_name, output=False)
        data.append((pso_algo, convergence_data, fitness_data, diversity_data, velocity_data))
    from plot_results import plot_multiple_curves, plot_fitness_comparison
    algorithm_names = [ele[0] for ele in data]
    convergence = [ele[1] for ele in data]
    fitness = [ele[2] for ele in data]
    diversity = [ele[3] for ele in data]
    velocity = [ele[4] for ele in data]
    plot_multiple_curves(convergence, benchmark_name, title=f'{benchmark_name.upper()} Function - PSO Convergence Comparison', x_label='Iteration', y_label='Fitness', algorithm_names=algorithm_names)
    plot_fitness_comparison(fitness, benchmark_name, title=f'{benchmark_name.upper()} Function - PSO Fitness Comparison', x_label='Dimension', y_label='Final Fitness (Mean)', algorithm_names=algorithm_names)
    # TODO:  add plots for diversity, velocity
    plot_multiple_curves(diversity, benchmark_name, title=f'{benchmark_name.upper()} Function - PSO Swarm Diversity Comparison', x_label='Iteration', y_label='Swarm Diveristy', algorithm_names=algorithm_names)
    plot_multiple_curves(velocity, benchmark_name, title=f'{benchmark_name.upper()} Function - PSO Velocity Comparison', x_label='Iteration', y_label='Average Velocity', algorithm_names=algorithm_names)

PSO_ALGORITHMS = {
    'StandardPSO': StandardPSO,
    'StochasticScalingPSO': StochasticScalingPSO,
    'SubspaceInitPSO': SubspaceInitPSO,
    'HybridPSO': HybridPSO
}
def run_pso(pso_algo: str):
   print(f"Testing {pso_algo}")
   class_name = PSO_ALGORITHMS[pso_algo]
   for bf in BENCHMARK_FUNCTIONS.keys():
        pso_stats(bf, class_name)
        return  # REMOVE
    
def main():
    if len(sys.argv) < 2:
        print("Usage: python pso_implementation.py <PSO_ALGORITHM | OPTION>")
        print("Available algorithms:", ", ".join(PSO_ALGORITHMS.keys()))
        return

    pso_algo = sys.argv[1]
    if pso_algo == "all":
        for algo in PSO_ALGORITHMS.keys():
            run_pso(algo)
    elif pso_algo in PSO_ALGORITHMS:
        run_pso(pso_algo)
    elif pso_algo in BENCHMARK_FUNCTIONS:
        run_pso_plotter(pso_algo)
    else:
        print(f"Invalid PSO algorithm: {pso_algo}")
        print("Available algorithms:", ", ".join(PSO_ALGORITHMS.keys()))
        return

if __name__ == "__main__":
    main()
