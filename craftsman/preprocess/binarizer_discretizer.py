import numpy as np
from pandas import DataFrame, Series
from sympy import And, Symbol
from numpy import array

from craftsman.base.operator import CON_C_CAT
from craftsman.base.defs import OperatorName

class BinarizerDiscretizerSQLOperator(CON_C_CAT):

    def __init__(self, featrues: list[str], fitted_transform):
        super().__init__(OperatorName.BINARIZERDISCRETIZER)
        self.features = featrues
        self._extract(fitted_transform)


    def _extract(self, fitted_transform) -> None:  
        self.categories = []
        self.bin_edges = []
        self.n_bins = []
        self.threshold = fitted_transform.threshold
        for feature in self.features:
            self.features_out.append(feature)
            self.n_bins.append(2)
            feature_idx = fitted_transform.feature_names_in_.tolist().index(feature)
            self.bin_edges.append([-float("inf"), self.threshold, float("inf")])
            self.categories.append([0, 1])
            intervals = [
                (self.bin_edges[0][i], self.bin_edges[0][i + 1])
                for i in range(self.n_bins[0])
            ]
            
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
