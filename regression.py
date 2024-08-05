# regression.py
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor

def train_models(X, y_initial_fiber_tear_1, y_initial_fiber_tear_2, y_initial_fast_load, y_initial_slow_load, y_after_500_fiber_tear_1, y_after_500_fiber_tear_2, y_after_500_fast_load, y_after_500_slow_load, model_choice):
    if model_choice == "Ridge":
        model_initial_fiber_tear_1 = Ridge().fit(X, y_initial_fiber_tear_1)
        model_initial_fiber_tear_2 = Ridge().fit(X, y_initial_fiber_tear_2)
        model_initial_fast_load = Ridge().fit(X, y_initial_fast_load)
        model_initial_slow_load = Ridge().fit(X, y_initial_slow_load)
        model_after_500_fiber_tear_1 = Ridge().fit(X, y_after_500_fiber_tear_1)
        model_after_500_fiber_tear_2 = Ridge().fit(X, y_after_500_fiber_tear_2)
        model_after_500_fast_load = Ridge().fit(X, y_after_500_fast_load)
        model_after_500_slow_load = Ridge().fit(X, y_after_500_slow_load)
    elif model_choice == "Lasso":
        model_initial_fiber_tear_1 = Lasso().fit(X, y_initial_fiber_tear_1)
        model_initial_fiber_tear_2 = Lasso().fit(X, y_initial_fiber_tear_2)
        model_initial_fast_load = Lasso().fit(X, y_initial_fast_load)
        model_initial_slow_load = Lasso().fit(X, y_initial_slow_load)
        model_after_500_fiber_tear_1 = Lasso().fit(X, y_after_500_fiber_tear_1)
        model_after_500_fiber_tear_2 = Lasso().fit(X, y_after_500_fiber_tear_2)
        model_after_500_fast_load = Lasso().fit(X, y_after_500_fast_load)
        model_after_500_slow_load = Lasso().fit(X, y_after_500_slow_load)
    elif model_choice == "ElasticNet":
        model_initial_fiber_tear_1 = ElasticNet().fit(X, y_initial_fiber_tear_1)
        model_initial_fiber_tear_2 = ElasticNet().fit(X, y_initial_fiber_tear_2)
        model_initial_fast_load = ElasticNet().fit(X, y_initial_fast_load)
        model_initial_slow_load = ElasticNet().fit(X, y_initial_slow_load)
        model_after_500_fiber_tear_1 = ElasticNet().fit(X, y_after_500_fiber_tear_1)
        model_after_500_fiber_tear_2 = ElasticNet().fit(X, y_after_500_fiber_tear_2)
        model_after_500_fast_load = ElasticNet().fit(X, y_after_500_fast_load)
        model_after_500_slow_load = ElasticNet().fit(X, y_after_500_slow_load)
    else:
        model_initial_fiber_tear_1 = KNeighborsRegressor().fit(X, y_initial_fiber_tear_1)
        model_initial_fiber_tear_2 = KNeighborsRegressor().fit(X, y_initial_fiber_tear_2)
        model_initial_fast_load = KNeighborsRegressor().fit(X, y_initial_fast_load)
        model_initial_slow_load = KNeighborsRegressor().fit(X, y_initial_slow_load)
        model_after_500_fiber_tear_1 = KNeighborsRegressor().fit(X, y_after_500_fiber_tear_1)
        model_after_500_fiber_tear_2 = KNeighborsRegressor().fit(X, y_after_500_fiber_tear_2)
        model_after_500_fast_load = KNeighborsRegressor().fit(X, y_after_500_fast_load)
        model_after_500_slow_load = KNeighborsRegressor().fit(X, y_after_500_slow_load)
    
    return (
        model_initial_fiber_tear_1, model_initial_fiber_tear_2, model_initial_fast_load, model_initial_slow_load, 
        model_after_500_fiber_tear_1, model_after_500_fiber_tear_2, model_after_500_fast_load, model_after_500_slow_load
    )
