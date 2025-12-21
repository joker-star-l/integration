from pandas import DataFrame, Series
from sympy import symbols, Eq, log

from craftsman.base.operator import CON_S_CON
from craftsman.base.defs import  OperatorName


class PowerTransformerSQLOperator(CON_S_CON):
    """
    This class implements the SQL wrapper for a Sklearn PowerTransformer object.
    """

    def __init__(self, featrues: list[str], fitted_transform):
        super().__init__(OperatorName.POWERTRANSFORMER)
        self.features = featrues
        self._extract(fitted_transform)

    def _extract(self, fitted_transform) -> None:
        lambdas_list = fitted_transform.lambdas_
        method = fitted_transform.method
        symbols_list = symbols("y x lambdas")
        y, x, lambdas = symbols_list
        self.symbols = {symb.name: symb for symb in symbols_list}
        for feature in self.features:
            self.features_out.append(feature)
            feature_idx = fitted_transform.feature_names_in_.tolist().index(feature)
            
        
            if method == "yeo-johnson":
                self.intervals = [(-float("inf"), 0), (0, float("inf"))]
                self.equations = []
                if lambdas_list[feature_idx] != 2:
                    self.equations.append(Eq(y, -((-x+1)**(2-lambdas) - 1)/(2-lambdas)))
                else:
                    self.equations.append(Eq(y, -log(-x+1)))
                if lambdas_list[feature_idx] != 0:
                    self.equations.append(Eq(y, ((x+1)**lambdas - 1)/lambdas))
                else:
                    self.equations.append(Eq(y, log(x+1)))

                self.parameter_values.append(
                    {
                        "lambdas": [lambdas_list[feature_idx]] * 2
                    }
                )
                
            elif method == "box-cox":
                self.intervals = [(-float("inf"), float("inf"))]
                self.equations = []
                if lambdas_list[feature_idx] != 0:
                    self.equations.append(Eq(y, (x**lambdas - 1)/lambdas))
                else:
                    self.equations.append(Eq(y, log(x)))
                
                self.parameter_values.append(
                    {
                        "lambdas": [lambdas_list[feature_idx]]
                    }
                )
            
            self.mappings.append(Series(self.equations, index=self.intervals))

    @staticmethod
    def trans_feature_names_in(input_data: DataFrame):
        return input_data.columns
