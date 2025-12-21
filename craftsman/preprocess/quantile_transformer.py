from pandas import DataFrame, Series
from sympy import symbols, Eq

from craftsman.base.operator import CON_S_CON
from craftsman.base.defs import OperatorName


class QuantileTransformerSQLOperator(CON_S_CON):
    """
    This class implements the SQL wrapper for a Sklearn QuantileTransformer object.
    """

    def __init__(self, featrues: list[str], fitted_transform):
        super().__init__(OperatorName.QUANTILETRANSFORMER)
        self.features = featrues
        self._extract(fitted_transform)

    def _extract(self, fitted_transform) -> None:
        n_quantiles = fitted_transform.n_quantiles_
        quantiles_list = fitted_transform.quantiles_
        references_list = fitted_transform.references_
        symbols_list = symbols("y x scale base")
        y, x, scale, base = symbols_list
        self.symbols = {symb.name: symb for symb in symbols_list}
        for feature in self.features:
            self.features_out.append(feature)
            feature_idx = fitted_transform.feature_names_in_.tolist().index(feature)
            intervals = []
            scales = []
            bases = []
            for i in range(n_quantiles - 1):
                intervals.append(
                    f"({quantiles_list[feature_idx][i]}, {quantiles_list[feature_idx][i+1]})"
                )
                scales.append(
                    f"({references_list[feature_idx][i+1]} - {references_list[feature_idx][i]}) / ({quantiles_list[feature_idx][i+1]} - {quantiles_list[feature_idx][i]})"
                )
                bases.append(
                    f"{references_list[feature_idx][i]} - {quantiles_list[feature_idx][i]} * {scales[i]}"
                )
            equations = [Eq(y, x * scale + base)] * (n_quantiles - 1)

            self.parameter_values.append({"scale": scales, "base": bases})

            self.mappings.append(Series(equations, index=intervals))

    @staticmethod
    def trans_feature_names_in(input_data: DataFrame):
        return input_data.columns
