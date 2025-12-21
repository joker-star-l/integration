import numpy as np
from pandas import DataFrame, Series
from sympy import And, Symbol
from numpy import array

from craftsman.base.operator import CON_C_CAT
from craftsman.base.defs import OperatorName
import craftsman.base.defs as defs

class KBinsDiscretizerSQLOperator(CON_C_CAT):

    def __init__(self, featrues: list[str], fitted_transform):
        super().__init__(OperatorName.KBINSDISCRETIZER)
        self.features = featrues
        self._extract(fitted_transform)


    def _extract(self, fitted_transform) -> None:  
        self.strategy = fitted_transform.strategy
        self.bin_distribution = fitted_transform.bin_distribution
        for feature in self.features:
            self.features_out.append(feature)
            feature_idx = fitted_transform.feature_names_in_.tolist().index(feature)
            self.bin_edges.append(fitted_transform.bin_edges_[feature_idx])
            self.n_bins.append(fitted_transform.n_bins_[feature_idx])
            self.categories.append(array(list(range(fitted_transform.n_bins_[feature_idx]))))
            intervals = [
                (self.bin_edges[0][i], self.bin_edges[0][i + 1])
                for i in range(self.n_bins[0])
            ]
            
            if defs.AUTO_RULE_GEN:
                self.inequations[feature] = []
                self.inequations_mappings[feature] = []
                x = Symbol('x')
                for idx, interval in enumerate(intervals):
                    self.inequations[feature].append(And(x > interval[0], x < interval[1]))
                    self.inequations_mappings[feature].append(idx)
                
            self.mappings.append(
                Series(
                    self.categories[-1],
                    index=intervals
                )
            )
        

    @staticmethod
    def trans_feature_names_in(input_data: DataFrame):
        return input_data.columns
