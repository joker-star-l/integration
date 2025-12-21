from pandas import DataFrame
import hashlib
import sys
import pandas as pd
from craftsman.base.operator import EXPAND
from craftsman.base.defs import OperatorName
import craftsman.base.defs as defs

class HashingEncoderSQLOperator(EXPAND):
    def __init__(self, featrue: list[str], fitted_transform):
        super().__init__(OperatorName.HASHINGENCODER)
        self.features = featrue
        self._extract(fitted_transform)

    
    def _extract(self, fitted_transform) -> None:
        self.value_counts = fitted_transform.value_counts
        self.n_components = fitted_transform.n_components
        defs.HASHING_ENCODER_N_COMPONENTS = self.n_components
        defs.HASHING_ENCODER_FEATURE = self.features[0]
        self.hashing_method = fitted_transform.hash_method
        self.mapping = self.hashing_trick(DataFrame(self.x_unique), self.hashing_method, self.n_components, self.features)
        self.mapping = self.x_unique.values.tolist()

        
    def hashing_trick(self, X_in, hashing_method='md5', N=2, cols=None, make_copy=False):
        if hashing_method not in hashlib.algorithms_available:
            raise ValueError(f"Hashing Method: {hashing_method} not Available. "
                             f"Please use one from: [{', '.join([str(x) for x in hashlib.algorithms_available])}]")

        if make_copy:
            X = X_in.copy(deep=True)
        else:
            X = X_in

        if cols is None:
            cols = X.columns

        def hash_fn(x):
            tmp = [0 for _ in range(N)]
            for val in x.array:
                if val is not None:
                    hasher = hashlib.new(hashing_method)
                    if sys.version_info[0] == 2:
                        hasher.update(str(val))
                    else:
                        hasher.update(bytes(str(val), 'utf-8'))
                    tmp[int(hasher.hexdigest(), 16) % N] += 1
            return tmp

        new_cols = [f'{self.features[0]}_{d}' for d in range(N)]

        X_cat = X.loc[:, cols]
        X_num = X.loc[:, [x for x in X.columns if x not in cols]]

        X_cat = X_cat.apply(hash_fn, axis=1, result_type='expand')
        X_cat.columns = new_cols

        X = pd.concat([X_cat, X_num], axis=1)

        return X


    @staticmethod
    def trans_feature_names_in(input_data: DataFrame):
        feature_names_out = [f'{defs.HASHING_ENCODER_FEATURE}_{d}' for d in range(defs.HASHING_ENCODER_N_COMPONENTS)]
        return feature_names_out
