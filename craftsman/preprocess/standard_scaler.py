from pandas import DataFrame
from sympy import symbols, Eq

from craftsman.base.operator import CON_A_CON
from craftsman.base.defs import  OperatorName


class StandardScalerSQLOperator(CON_A_CON):
    """
    This class implements the SQL wrapper for a Sklearn StardardScaler object.
    """

    def __init__(self, featrues: list[str], fitted_transform):
        super().__init__(OperatorName.STANDARDSCALER)
        self.features = featrues
        self._extract(fitted_transform)

    def _extract(self, fitted_transform) -> None:
        for feature in self.features:
            self.features_out.append(feature)
            feature_idx = fitted_transform.feature_names_in_.tolist().index(feature)
            self.parameter_values.append(
                {
                    "mean": fitted_transform.mean_[feature_idx],
                    "std": fitted_transform.scale_[feature_idx],
                }
            )

        symbols_list = symbols("y x mean std")
        y, x, mean, std = symbols_list
        self.symbols = {symb.name: symb for symb in symbols_list}
        self.equation = Eq(y, (x - mean) / std)

    @staticmethod
    def trans_feature_names_in(input_data: DataFrame):
        return input_data.columns