from pandas import DataFrame
from sympy import symbols, Eq

from craftsman.base.operator import CON_A_CON
from craftsman.base.defs import  OperatorName


class RobustScalerSQLOperator(CON_A_CON):
    """
    This class implements the SQL wrapper for a Sklearn RobustScaler object.
    """

    def __init__(self, featrues: list[str], fitted_transform):
        super().__init__(OperatorName.ROBUSTSCALER)
        self.features = featrues
        self._extract(fitted_transform)

    def _extract(self, fitted_transform) -> None:
        center_list = fitted_transform.center_
        scale_list = fitted_transform.scale_
        
        for feature in self.features:
            self.features_out.append(feature)
            feature_idx = fitted_transform.feature_names_in_.tolist().index(feature)
            self.parameter_values.append(
                {
                    "center": center_list[feature_idx],
                    "scale": scale_list[feature_idx]
                }
            )

        symbols_list = symbols("y x center scale")
        y, x, center, scale = symbols_list
        self.symbols = {symb.name: symb for symb in symbols_list}
        self.equation = Eq(y, (x - center) / scale)

    @staticmethod
    def trans_feature_names_in(input_data: DataFrame):
        return input_data.columns
