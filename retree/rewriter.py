# rewriter
from typing import Callable, Any
from .tree import TreeEnsembleRegressor

def rewrite_model(model: TreeEnsembleRegressor, func: Callable[[Any], bool]):
    for r in model.regressors:
        for i in range(len(r.target_weights)):
            r.target_weights[i] = float(func(r.target_weights[i]))
