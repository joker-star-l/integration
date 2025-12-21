from pandas import DataFrame, Series

import numpy as np
from craftsman.base.operator import CAT_C_CAT
from craftsman.base.defs import OperatorName

class TargetEncoderSQLOperator(CAT_C_CAT):

    def __init__(self, featrues: list[str], fitted_transform):
        super().__init__(OperatorName.TARGETENCODER)
        self.features = featrues
        self._extract(fitted_transform)

    def _extract(self, fitted_transform) -> None:
        self.non_string_columns = fitted_transform.non_string_columns
        self.value_counts = fitted_transform.value_counts
        for feature in self.features:
            self.features_out.append(feature)
            target_mapping = fitted_transform.mapping[feature]
            oe = fitted_transform.ordinal_encoder
            for m in oe.mapping:
                if m['col'] == feature:
                    oe_mapping = m['mapping']
                    oe_mapping = oe_mapping[~oe_mapping.index.isnull()]
                    oe_mapping = oe_mapping[[idx for idx in oe_mapping.index if idx != 'NaN']]
                    break
            if feature in self.non_string_columns:
                self.mappings.append(
                    Series(
                        target_mapping[oe_mapping].tolist(),
                        index=np.array(oe_mapping.index.tolist()).astype(
                            self.non_string_columns[feature]
                        ),
                    )
                )
            else:
                self.mappings.append(Series(target_mapping[oe_mapping].tolist(), index=oe_mapping.index.tolist()))

    @staticmethod
    def trans_feature_names_in(input_data: DataFrame):
        return input_data.columns
