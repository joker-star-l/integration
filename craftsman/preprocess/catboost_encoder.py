from pandas import DataFrame

from craftsman.base.operator import CAT_C_CAT
from craftsman.base.defs import OperatorName

class CatBoostEncoderSQLOperator(CAT_C_CAT):

    def __init__(self, featrues: list[str], fitted_transform):
        super().__init__(OperatorName.CATBOOSTENCODER)
        self.features = featrues
        self._extract(fitted_transform)


    def _extract(self, fitted_transform) -> None:
        self.a = fitted_transform.a
        self.mean = fitted_transform._mean
        cat_info = fitted_transform.mapping
        for feature in self.features:
            self.features_out.append(feature)
            colmap = cat_info[feature]
            level_notunique = colmap['count'] > 1
            level_means = ((colmap['sum'] + self.mean * self.a) / (colmap['count'] + self.a)).where(level_notunique, self.mean)
            self.mappings.append(level_means)


    @staticmethod
    def trans_feature_names_in(input_data: DataFrame):
        return input_data.columns
