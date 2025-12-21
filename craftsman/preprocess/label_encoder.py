from pandas import DataFrame, Series

from craftsman.base.operator import CAT_C_CAT
from craftsman.base.defs import OperatorName


class LabelEncoderSQLOperator(CAT_C_CAT):

    def __init__(self, featrues: list[str], fitted_transform):
        super().__init__(OperatorName.LABELENCODER)
        self.features = featrues
        self._extract(fitted_transform)

    def _extract(self, fitted_transform) -> None:
        self.value_counts = fitted_transform.value_counts
        for idx, feature in enumerate(self.features):
            self.features_out.append(feature)
            label_mapping = Series(
                fitted_transform.transform(fitted_transform.classes_).index,
                index=fitted_transform.classes_
            )
            self.mappings.append(label_mapping)

    @staticmethod
    def trans_feature_names_in(input_data: DataFrame):
        return input_data.columns
