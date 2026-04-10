# add functions then add them into a dictionary to be called in the main file
# 5 OR 6 benchmarks
import numpy as np
from typing import Tuple

# Ackley function, Griewank, Rastrigin ShRot, Weierstrass Sh

def sphere_function(x):
        return np.sum(x**2)

def ackley_function(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x)) / d)
    
    return sum_sq_term + cos_term + a + np.exp(1)

def griewank_function(x):
    sum_term = np.sum(x**2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    
    return 1 + sum_term - prod_term

def rastrigin_function(x):
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def weierstrass_function(x):
    a = 0.5
    b = 3
    j_max = 20
    n = len(x)

    sum2 = 0.0
    for j in range(0, j_max + 1):
        sum2 += (a**j) * (np.cos(np.pi * (b**j)))
    sum2 *= n

    f_x = 0.0
    for i in range(n):
        sum1 = 0.0
        for j in range(0, j_max + 1):
            sum1 += (a**j) * (np.cos((2 * np.pi * (b**j)) * (x[i] + 0.5)))
        f_x += sum1 - sum2
    return f_x

def rosenbrock_function(x):
    f_x = 0.0
    n = len(x)

    for i in range(n - 1):
        f_x += 100 * (x[i + 1] - x[i]**2)**2 + (x[i] - 1)**2
    
    return f_x
        
BENCHMARK_FUNCTIONS = {
    'sphere': (sphere_function, (-100, 100)), # min = 0, x = 0
    'ackley': (ackley_function, (-32, 32)), # min = 0, x = 0
    'griewank': (griewank_function, (-600, 600)), # min = 0, x = 0
    'rastrigin': (rastrigin_function, (-5.12, 5.12)), # min = 0, x = 0
    # 'weierstrass': (weierstrass_function, (-0.5, 0.5)), # min = 4, x = 0
    'rosenbrock': (rosenbrock_function, (-30, 30)) # min = 0, x = 1
}

def get_benchmark_function(name):
    if name in BENCHMARK_FUNCTIONS:
        return BENCHMARK_FUNCTIONS[name]
    else:
        raise ValueError(f"Benchmark function '{name}' not found.")
    
