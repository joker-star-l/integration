from pandas import DataFrame, Series
import numpy as np

from craftsman.base.operator import CAT_C_CAT
from craftsman.base.defs import OperatorName

class OrdinalEncoderSQLOperator(CAT_C_CAT):

    def __init__(self, featrues: list[str], fitted_transform):
        super().__init__(OperatorName.ORDINALENCODER)
        self.features = featrues
        self._extract(fitted_transform)


    def _extract(self, fitted_transform) -> None:
        self.value_counts = fitted_transform.value_counts
        for feature in self.features:
            self.features_out.append(feature)
            idx = fitted_transform.feature_names_in_.tolist().index(feature)
            categories = fitted_transform.categories_[idx]
            # encs = fitted_transform.transform(categories.reshape(-1, 1))
            self.mappings.append(Series(np.arange(1, len(categories) + 1)  ,index=categories))


    @staticmethod
    def trans_feature_names_in(input_data: DataFrame):
        return input_data.columns
    
