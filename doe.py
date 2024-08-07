# doe.py
import pandas as pd
from itertools import product
from sklearn.utils import resample

def create_full_factorial_design(optimal_a, optimal_b, optimal_c, optimal_d, step=0.1, max_experiments=25):
    a_range = [max(0, optimal_a - step), optimal_a, min(3, optimal_a + step)]
    b_range = [max(0, optimal_b - step), optimal_b, min(3, optimal_b + step)]
    c_range = [max(0, optimal_c - step), optimal_c, min(3, optimal_c + step)]
    d_range = [max(0, optimal_d - step), optimal_d, min(3, optimal_d + step)]
    
    full_factorial_design = list(product(a_range, b_range, c_range, d_range))
    
    # Reduce the number of experiments by resampling the full factorial design
    if len(full_factorial_design) > max_experiments:
        reduced_design = resample(full_factorial_design, n_samples=max_experiments, random_state=42)
    else:
        reduced_design = full_factorial_design
    
    df_fractional_factorial_design = pd.DataFrame(reduced_design, columns=['Ingredient A', 'Ingredient B', 'Ingredient C', 'Ingredient D'])
    
    return df_fractional_factorial_design