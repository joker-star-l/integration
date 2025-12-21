from pandas import DataFrame
from numpy import zeros
from sklearn.preprocessing import OneHotEncoder

from craftsman.base.operator import  EXPAND
from craftsman.base.defs import  OperatorName

class OneHotEncoderSQLOperator(EXPAND):
    """
    This class implements the SQL Operator for a Sklearn OneHotEncoder object.
    """

    def __init__(self, featrue: list[str], fitted_transform):
        super().__init__(OperatorName.ONEHOTENCODER)
        self.features = featrue
        self._extract(fitted_transform)


    def _extract(self, fitted_transform) -> None:
        self.value_counts = fitted_transform.value_counts
        feature = self.features[0]
        feature_idx = fitted_transform.feature_names_in_.tolist().index(feature)
        categories = fitted_transform.categories_[feature_idx]
        df = DataFrame(zeros((len(categories), len(categories)), dtype=int), index=categories, columns=categories)
        for category in categories:
            df.loc[category, category] = 1
        df.columns = [feature + f'_{i}' for i in range(len(df.columns))]
        self.features_out.extend(df.columns.tolist())
        self.mapping = df


    @staticmethod
    def trans_feature_names_in(input_data: DataFrame):
        feature_names_out = []

        one_hot_encoder = OneHotEncoder()
        one_hot_encoder.fit(input_data)
        feature_names_in = one_hot_encoder.feature_names_in_
        categories_list = one_hot_encoder.categories_

        for feature_name, categories in zip (feature_names_in, categories_list):
            feature_names_out.extend([feature_name + f'_{i}' for i in range(len(categories))])

        return feature_names_out