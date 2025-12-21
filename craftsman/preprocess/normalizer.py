from pandas import DataFrame
from sympy import symbols, Eq
import numpy as np

from craftsman.base.operator import CON_A_CON
from craftsman.base.defs import  OperatorName
from craftsman.utility.dbms_utils import DBMSUtils
import craftsman.base.defs as defs


class NormalizerSQLOperator(CON_A_CON):
    """
    This class implements the SQL wrapper for a Sklearn Normalizer object.
    """

    def __init__(self, featrues: list[str], fitted_transform):
        super().__init__(OperatorName.NORMALIZER)
        self.features = featrues
        self._extract(fitted_transform)

    def _extract(self, fitted_transform) -> None:
        self.norm = fitted_transform.norm
        for feature in self.features:
            self.features_out.append(feature)
            feature_idx = fitted_transform.feature_names_in_.tolist().index(feature)
            if self.norm == 'l1':
                scale = np.linalg.norm(DBMSUtils.fetch_data_as_numpy(defs.DBMS, defs.TABLE_NAME, feature), ord=1)
            elif self.norm == 'l2':
                scale = np.linalg.norm(DBMSUtils.fetch_data_as_numpy(defs.DBMS, defs.TABLE_NAME, feature), ord=2)
            elif self.norm == 'max':
                scale = np.max(DBMSUtils.fetch_data_as_numpy(defs.DBMS, defs.TABLE_NAME, feature))
            self.parameter_values.append(
                {
                    "scale": scale
                }
            )

        symbols_list = symbols("y x scale")
        y, x, scale = symbols_list
        self.symbols = {symb.name: symb for symb in symbols_list}
        self.equation = Eq(y, x / scale)

    @staticmethod
    def trans_feature_names_in(input_data: DataFrame):
        return input_data.columns
