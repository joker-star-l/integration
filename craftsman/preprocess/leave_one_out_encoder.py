from pandas import DataFrame

from craftsman.base.operator import CAT_C_CAT
from craftsman.base.defs import OperatorName

class LeaveOneOutEncoderSQLOperator(CAT_C_CAT):

    def __init__(self, featrues: list[str], fitted_transform):
        super().__init__(OperatorName.LEAVEONEOUTENCODER)
        self.features = featrues
        self._extract(fitted_transform)


    def _extract(self, fitted_transform) -> None:
        self.value_counts = fitted_transform.value_counts
        for feature in self.features:
            self.features_out.append(feature)
            loo_mapping = fitted_transform.mapping[feature]
            self.mappings.append(loo_mapping['sum'] / loo_mapping['count'])


    @staticmethod
    def trans_feature_names_in(input_data: DataFrame):
        return input_data.columns
