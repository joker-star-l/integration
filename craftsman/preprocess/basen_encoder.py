from pandas import DataFrame, Series

from category_encoders import BaseNEncoder
from craftsman.base.operator import EXPAND
from craftsman.base.defs import OperatorName

class BaseNEncoderSQLOperator(EXPAND):
    def __init__(self, featrue: list[str], fitted_transform):
        super().__init__(OperatorName.BASENENCODER)
        self.features = featrue
        self._extract(fitted_transform)

    
    def _extract(self, fitted_transform) -> None:
        self.value_counts = fitted_transform.value_counts
        feature = self.features[0]
        for m in fitted_transform.mapping:
            if m['col'] == feature:
                self.features_out.extend(m['mapping'].columns.tolist())
                self.mapping = m['mapping']
                for enc_mapping in fitted_transform.ordinal_encoder.category_mapping:
                    if enc_mapping['col'] == feature:
                        reversed_ordinal_enc = Series(data=enc_mapping['mapping'].index, index=enc_mapping['mapping'])
                        reversed_ordinal_enc[-1] = 'NaN'
                        break
                if not self.mapping.index.dtype == object:
                    self.mapping.index = [reversed_ordinal_enc[idx] for idx in self.mapping.index]
                self.mapping = self.mapping[~self.mapping.index.isnull()]
                self.mapping = self.mapping.loc[[idx for idx in self.mapping.index if idx != 'NaN']]
                break
    

    @staticmethod
    def trans_feature_names_in(input_data: DataFrame):
        feature_names_out = []

        base_n_encoder = BaseNEncoder(cols=input_data.columns)
        base_n_encoder.fit(input_data)
        base_n_encoder.transform(input_data)

        for feature in input_data.columns:
            for m in base_n_encoder.mapping:
                if m['col'] == feature:
                    feature_names_out.extend(m['mapping'].columns.tolist())
                    break 

        return feature_names_out
