"""
Statistical Analysis for Evolutionary Algorithm Comparison

This module implements statistical testing procedures following the methodology from:
Carrasco, J., García, S., Rueda, M.M., Das, S., & Herrera, F. (2020). 
"Recent trends in the use of statistical tests for comparing swarm and evolutionary 
computing algorithms: Practical guidelines and a critical review." 
Swarm and Evolutionary Computation, 54, 100665.

Standard practice in EA comparison:
1. Friedman test for overall significance across multiple algorithms/benchmarks
2. Post-hoc tests (Nemenyi, Holm, etc.) if Friedman rejects null hypothesis  
3. Wilcoxon signed-rank test for pairwise comparisons
4. Rankings and critical difference diagrams for visualization
"""

import numpy as np
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from typing import Dict, List, Tuple
import warnings

def friedman_test(results: Dict[str, Dict[str, Dict]], benchmarks: List[str]) -> Tuple[float, float]:
    """
    Friedman test to detect significant differences among multiple algorithms.
    
    The Friedman test is a non-parametric alternative to repeated-measures ANOVA.
    It ranks algorithms for each benchmark and tests if the ranking distributions differ.
    
    Reference: Friedman, M. (1937). The use of ranks to avoid the assumption of 
    normality implicit in the analysis of variance. Journal of the American 
    Statistical Association, 32(200), 675-701.
    
    Args:
        results: Nested dict with structure results[benchmark][algorithm]['fitnesses']
        benchmarks: List of benchmark function names
        
    Returns:
        statistic: Friedman test statistic (chi-square distributed)
        p_value: p-value for the test
    """
    alg_names = sorted(list(results[benchmarks[0]].keys()))
    
    # Create matrix: rows = benchmarks, columns = algorithms
    # Each cell contains mean fitness for that algorithm on that benchmark
    data_matrix = []
    for benchmark in benchmarks:
        row = [results[benchmark][alg]['mean'] for alg in alg_names]
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    # Friedman test expects samples (algorithms) as separate arrays
    samples = [data_matrix[:, i] for i in range(len(alg_names))]
    
    statistic, p_value = friedmanchisquare(*samples)
    
    return statistic, p_value

def compute_rankings(results: Dict[str, Dict[str, Dict]], benchmarks: List[str]) -> Dict[str, List[int]]:
    """
    Compute algorithm rankings for each benchmark (1 = best, higher = worse).
    
    Rankings are fundamental in non-parametric tests because they:
    1. Remove assumptions about data distribution
    2. Handle outliers robustly
    3. Focus on relative performance rather than absolute values
    
    Args:
        results: Nested dict with structure results[benchmark][algorithm]['mean']
        benchmarks: List of benchmark function names
        
    Returns:
        Dictionary mapping algorithm names to lists of ranks
    """
    alg_names = sorted(list(results[benchmarks[0]].keys()))
    rankings = {alg: [] for alg in alg_names}
    
    for benchmark in benchmarks:
        # Get mean fitness for each algorithm on this benchmark
        fitness_values = [(alg, results[benchmark][alg]['mean']) for alg in alg_names]
        
        # Sort by fitness (lower is better for minimization)
        sorted_algs = sorted(fitness_values, key=lambda x: x[1])
        
        # Assign ranks (1 is best)
        for rank, (alg, _) in enumerate(sorted_algs, 1):
            rankings[alg].append(rank)
    
    return rankings

def average_rankings(rankings: Dict[str, List[int]]) -> Dict[str, float]:
    """
    Compute average rank for each algorithm across all benchmarks.
    
    Average ranking is a primary metric in EA comparison because it:
    1. Summarizes performance across multiple problems
    2. Enables comparison via Friedman test
    3. Feeds into post-hoc procedures like Nemenyi
    
    Args:
        rankings: Dict mapping algorithm names to lists of ranks
        
    Returns:
        Dict mapping algorithm names to average ranks
    """
    return {alg: np.mean(ranks) for alg, ranks in rankings.items()}

def nemenyi_critical_difference(num_algorithms: int, num_benchmarks: int, alpha: float = 0.05) -> float:
    """
    Compute critical difference for Nemenyi post-hoc test.
    
    The Nemenyi test is analogous to Tukey's HSD test for ANOVA. It determines
    if the difference in average ranks between two algorithms is statistically
    significant.
    
    Reference: Nemenyi, P. (1963). Distribution-free multiple comparisons. 
    Princeton University.
    
    Critical values from Demšar (2006): "Statistical comparisons of classifiers 
    over multiple data sets." Journal of Machine Learning Research, 7, 1-30.
    
    Args:
        num_algorithms: Number of algorithms being compared (k)
        num_benchmarks: Number of benchmark problems (N)
        alpha: Significance level (typically 0.05)
        
    Returns:
        Critical difference value
    """
    # Critical values for two-tailed Nemenyi test at alpha=0.05
    # From Demšar (2006) Table 1
    q_alpha_values = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 15: 3.384,
        20: 3.526
    }
    
    if num_algorithms in q_alpha_values:
        q_alpha = q_alpha_values[num_algorithms]
    elif num_algorithms > 20:
        # Approximate for large k
        q_alpha = 3.526 + (num_algorithms - 20) * 0.01
    else:
        # Linear interpolation for values not in table
        lower_k = max([k for k in q_alpha_values.keys() if k < num_algorithms])
        upper_k = min([k for k in q_alpha_values.keys() if k > num_algorithms])
        q_lower = q_alpha_values[lower_k]
        q_upper = q_alpha_values[upper_k]
        q_alpha = q_lower + (q_upper - q_lower) * (num_algorithms - lower_k) / (upper_k - lower_k)
    
    # Critical difference formula
    cd = q_alpha * np.sqrt((num_algorithms * (num_algorithms + 1)) / (6.0 * num_benchmarks))
    
    return cd

def pairwise_wilcoxon(results: Dict[str, Dict[str, Dict]], 
                     benchmarks: List[str],
                     alg1: str, 
                     alg2: str) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test for pairwise comparison of two algorithms.
    
    The Wilcoxon signed-rank test is a non-parametric alternative to the paired t-test.
    It's appropriate when:
    1. Data is paired (same benchmarks for both algorithms)
    2. Distribution may not be normal
    3. We want to test if one algorithm consistently outperforms another
    
    Reference: Wilcoxon, F. (1945). Individual comparisons by ranking methods. 
    Biometrics Bulletin, 1(6), 80-83.
    
    Args:
        results: Nested dict with structure results[benchmark][algorithm]['fitnesses']
        benchmarks: List of benchmark function names
        alg1: Name of first algorithm
        alg2: Name of second algorithm
        
    Returns:
        statistic: Wilcoxon test statistic
        p_value: Two-tailed p-value
    """
    # Extract mean fitness for each algorithm on each benchmark
    alg1_scores = [results[benchmark][alg1]['mean'] for benchmark in benchmarks]
    alg2_scores = [results[benchmark][alg2]['mean'] for benchmark in benchmarks]
    
    # Wilcoxon signed-rank test (two-tailed)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Exact p-value calculation.*')
        statistic, p_value = wilcoxon(alg1_scores, alg2_scores, alternative='two-sided')
    
    return statistic, p_value

def bonferroni_holm_correction(p_values: List[Tuple[str, str, float]], alpha: float = 0.05) -> List[Tuple[str, str, float, bool]]:
    """
    Holm-Bonferroni step-down procedure for controlling family-wise error rate (FWER).
    
    When performing multiple pairwise comparisons, the probability of making at least
    one Type I error increases. The Holm procedure controls this by adjusting the
    significance threshold for each test.
    
    Reference: Holm, S. (1979). A simple sequentially rejective multiple test procedure. 
    Scandinavian Journal of Statistics, 6(2), 65-70.
    
    Why use this? From Carrasco et al. (2020):
    "A statistical test for two samples, like the Wilcoxon's test, has an estimated error, 
    but this error increases with each pair of comparisons. Thus, when simultaneously 
    comparing the results of our proposal with those attained by several other algorithms, 
    the application of Wilcoxon's test (or others such as the t-test) is totally discouraged, 
    because it cannot ensure that the proposal is statistically better than all the other 
    reference algorithms."
    
    Args:
        p_values: List of tuples (alg1, alg2, p_value)
        alpha: Overall significance level (typically 0.05)
        
    Returns:
        List of tuples (alg1, alg2, p_value, is_significant)
    """
    # Sort p-values in ascending order
    sorted_tests = sorted(p_values, key=lambda x: x[2])
    
    k = len(p_values)
    results = []
    
    for i, (alg1, alg2, p_val) in enumerate(sorted_tests, 1):
        # Holm's adjusted threshold: alpha / (k - i + 1)
        adjusted_alpha = alpha / (k - i + 1)
        is_significant = p_val < adjusted_alpha
        
        results.append((alg1, alg2, p_val, is_significant, adjusted_alpha))
        
        # If we fail to reject at this step, stop (step-down procedure)
        if not is_significant:
            # Add remaining tests as not significant
            for j in range(i, len(sorted_tests)):
                alg1, alg2, p_val = sorted_tests[j]
                results.append((alg1, alg2, p_val, False, adjusted_alpha))
            break
    
    return results

def print_statistical_analysis(results: Dict[str, Dict[str, Dict]], benchmarks: List[str], alpha: float = 0.05):
    """
    Comprehensive statistical analysis following Carrasco et al. (2020) methodology.
    
    This function performs the complete statistical workflow recommended for 
    evolutionary algorithm comparison:
    
    1. Friedman test: Overall test for significance across all algorithms
    2. Nemenyi post-hoc: If Friedman is significant, determine which pairs differ
    3. Wilcoxon pairwise: Detailed pairwise comparisons with correction
    4. Rankings: Central metric in non-parametric EA comparison
    
    Args:
        results: Nested dict with structure results[benchmark][algorithm]['fitnesses']
        benchmarks: List of benchmark function names
        alpha: Significance level (default 0.05)
    """
    alg_names = sorted(list(results[benchmarks[0]].keys()))
    num_algorithms = len(alg_names)
    
    print("\n" + "="*100)
    print("STATISTICAL ANALYSIS (following Carrasco et al., 2020 methodology)")
    print("="*100)
    
    # Check if we have enough algorithms for Friedman test
    if num_algorithms < 3:
        print(f"\nNote: Friedman test requires at least 3 algorithms. Currently running {num_algorithms} algorithm(s).")
        print("Statistical tests will be skipped. Showing rankings only.\n")
        friedman_p = 1.0  # Not significant
    else:
        # 1. Friedman Test
        print("\n1. FRIEDMAN TEST (Overall Significance)")
        print("-" * 100)
        print("Purpose: Detect if there are statistically significant differences among algorithms")
        print("Null Hypothesis: All algorithms perform equivalently")
        print("-" * 100)
        
        friedman_stat, friedman_p = friedman_test(results, benchmarks)
        print(f"Friedman statistic: {friedman_stat:.4f}")
        print(f"p-value: {friedman_p:.6f}")
        
        if friedman_p < alpha:
            print(f"REJECT null hypothesis (p < {alpha}): Significant differences exist among algorithms")
        else:
            print(f"FAIL TO REJECT null hypothesis (p >= {alpha}): No significant differences detected")
    
    # 2. Rankings
    print("\n2. ALGORITHM RANKINGS")
    print("-" * 100)
    print("Average rank across all benchmarks (lower is better)")
    print("-" * 100)
    
    rankings = compute_rankings(results, benchmarks)
    avg_ranks = average_rankings(rankings)
    
    # Sort by average rank
    sorted_algs = sorted(avg_ranks.items(), key=lambda x: x[1])
    
    print(f"{'Rank':<6} {'Algorithm':<20} {'Avg Rank':<12} {'Individual Ranks'}")
    print("-" * 100)
    for overall_rank, (alg, avg_rank) in enumerate(sorted_algs, 1):
        individual = ', '.join([f"{r}" for r in rankings[alg]])
        print(f"{overall_rank:<6} {alg:<20} {avg_rank:<12.2f} [{individual}]")
    
    # 3. Nemenyi Post-hoc Test (if Friedman is significant and we have enough algorithms)
    if num_algorithms >= 3 and friedman_p < alpha:
        print("\n3. NEMENYI POST-HOC TEST")
        print("-" * 100)
        print("Purpose: Determine critical difference for pairwise rank comparisons")
        print("Two algorithms are significantly different if their avg rank difference > CD")
        print("-" * 100)
        
        cd = nemenyi_critical_difference(len(alg_names), len(benchmarks), alpha)
        print(f"Critical Difference (CD) at α={alpha}: {cd:.4f}")
        print()
        
        print("Pairwise Rank Differences:")
        print(f"{'Algorithm 1':<20} {'Algorithm 2':<20} {'Rank Diff':<12} {'Significant?'}")
        print("-" * 100)
        
        for i, (alg1, rank1) in enumerate(sorted_algs):
            for alg2, rank2 in sorted_algs[i+1:]:
                rank_diff = abs(rank1 - rank2)
                is_sig = rank_diff > cd
                sig_marker = "YES" if is_sig else "No"
                print(f"{alg1:<20} {alg2:<20} {rank_diff:<12.4f} {sig_marker}")
    
    # 4. Pairwise Wilcoxon Tests with Holm-Bonferroni Correction (only if we have at least 2 algorithms)
    if num_algorithms >= 2:
        print("\n4. PAIRWISE WILCOXON SIGNED-RANK TESTS")
        print("-" * 100)
        print("Purpose: Detailed pairwise comparisons of algorithm performance")
        print("Correction: Holm-Bonferroni step-down procedure to control family-wise error rate")
        print("-" * 100)
        
        # Collect all pairwise p-values
        pairwise_results = []
        for i, alg1 in enumerate(alg_names):
            for alg2 in alg_names[i+1:]:
                _, p_val = pairwise_wilcoxon(results, benchmarks, alg1, alg2)
                pairwise_results.append((alg1, alg2, p_val))
        
        # Apply Holm-Bonferroni correction
        corrected_results = bonferroni_holm_correction(pairwise_results, alpha)
        
        print(f"{'Algorithm 1':<20} {'Algorithm 2':<20} {'p-value':<12} {'Adj. α':<12} {'Significant?'}")
        print("-" * 100)
        
        for alg1, alg2, p_val, is_sig, adj_alpha in corrected_results:
            sig_marker = "✓ YES" if is_sig else "✗ No"
            print(f"{alg1:<20} {alg2:<20} {p_val:<12.6f} {adj_alpha:<12.6f} {sig_marker}")
    
    # 5. Summary and Recommendations
    print("\n5. SUMMARY AND CONCLUSIONS")
    print("=" * 100)
    
    best_alg = sorted_algs[0][0]
    best_rank = sorted_algs[0][1]
    
    print(f"Best performing algorithm: {best_alg} (average rank: {best_rank:.2f})")
    
    if friedman_p < alpha:
        print(f"\nStatistically significant differences found (Friedman p={friedman_p:.6f} < {alpha})")
        
        # Count significant differences for best algorithm
        sig_better_than = sum(1 for alg1, alg2, p_val, is_sig, _ in corrected_results 
                             if is_sig and best_alg in [alg1, alg2])
        
        if sig_better_than > 0:
            print(f"{best_alg} is significantly better than {sig_better_than} other algorithm(s)")
        else:
            print(f"{best_alg} has best average rank but differences may not be statistically significant")
    else:
        print(f"\nNo statistically significant differences found (Friedman p={friedman_p:.6f} >= {alpha})")
        print("All algorithms perform equivalently according to this test")
    
    print("\n" + "="*100)
