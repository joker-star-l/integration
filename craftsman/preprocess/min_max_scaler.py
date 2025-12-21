from pandas import DataFrame
from sympy import symbols, Eq

from craftsman.base.operator import CON_A_CON
from craftsman.base.defs import  OperatorName


class MinMaxScalerSQLOperator(CON_A_CON):
    """
    This class implements the SQL wrapper for a Sklearn MinMaxScaler object.
    """

    def __init__(self, featrues: list[str], fitted_transform):
        super().__init__(OperatorName.MINMAXSCALER)
        self.features = featrues
        self._extract(fitted_transform)

    def _extract(self, fitted_transform) -> None:
        data_min_list = fitted_transform.data_min_
        scale_list = fitted_transform.scale_
        range_min_list = fitted_transform.min_ + fitted_transform.data_min_ * fitted_transform.scale_
        for feature in self.features:
            self.features_out.append(feature)
            feature_idx = fitted_transform.feature_names_in_.tolist().index(feature)
            self.parameter_values.append(
                {
                    "data_min": data_min_list[feature_idx],
                    "scale": scale_list[feature_idx],
                    "range_min": range_min_list[feature_idx]
                }
            )

        symbols_list = symbols("y x data_min scale range_min")
        y, x, data_min, scale, range_min = symbols_list
        self.symbols = {symb.name: symb for symb in symbols_list}
        self.equation = Eq(y, (x - data_min) * scale + range_min)

    @staticmethod
    def trans_feature_names_in(input_data: DataFrame):
        return input_data.columns
