# doe.py
import pandas as pd
from itertools import product

def create_full_factorial_design(optimal_a, optimal_b, optimal_c, optimal_d, step=0.1):
    a_range = [max(0, optimal_a - step), optimal_a, min(3, optimal_a + step)]
    b_range = [max(0, optimal_b - step), optimal_b, min(3, optimal_b + step)]
    c_range = [max(0, optimal_c - step), optimal_c, min(3, optimal_c + step)]
    d_range = [max(0, optimal_d - step), optimal_d, min(3, optimal_d + step)]
    
    full_factorial_design = list(product(a_range, b_range, c_range, d_range))
    df_full_factorial_design = pd.DataFrame(full_factorial_design, columns=['Ingredient A', 'Ingredient B', 'Ingredient C', 'Ingredient D'])
    
    return df_full_factorial_design
