from sklearn.linear_model import LinearRegression

from craftsman.model.base_model import LinearModel
from craftsman.base.defs import ModelName
from craftsman.utility.dbms_utils import DBMSUtils

class LinearRegressionSQLModel(LinearModel):
    
    def __init__(self, trained_model: LinearRegression) -> None:
        super().__init__()
        self.model_name = ModelName.LINEARREGRESSION
        self.trained_model = trained_model
        self.input_features = trained_model.feature_names_in_
        self.weights = trained_model.coef_.ravel()
        self.bias = trained_model.intercept_
        
    def query(self, input_table: str, dbms: str) -> str:
        query = "SELECT "
        plus_items = []
        for i in range(len(self.input_features)):
            col = DBMSUtils.get_delimited_col(dbms, self.input_features[i])
            plus_item = " {} * {} ".format(col, self.weights[i], col)
            plus_items.append(plus_item)
            
        query += ('+'.join(plus_items) + f'+ {self.bias}')
        query += " FROM {}".format(input_table)
        
        return query