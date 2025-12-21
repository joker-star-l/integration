from sklearn.linear_model import LogisticRegression

from craftsman.model.base_model import LinearModel
from craftsman.base.defs import ModelName
from craftsman.utility.dbms_utils import DBMSUtils

class LogisticRegressionSQLModel(LinearModel):
    
    def __init__(self, trained_model: LogisticRegression) -> None:
        super().__init__()
        self.model_name = ModelName.LOGISTICREGRESSION
        self.trained_model = trained_model
        self.input_features = trained_model.feature_names_in_
        
        self.weights = trained_model.coef_
        self.bias = trained_model.intercept_
        self.classes = trained_model.classes_
        
    def query(self, input_table: str, dbms: str) -> str:
        if len(self.classes) == 2:
            query = "SELECT CASE WHEN "
            plus_items = []
            for i in range(len(self.input_features)):
                col = DBMSUtils.get_delimited_col(dbms, self.input_features[i])
                plus_item = " {} * {} ".format(col, self.weights[0][i], col)
                plus_items.append(plus_item)
            
            sum_sql =  '(' + '+'.join(plus_items) + f'+ {self.bias[0]}' + ')'   
            query += f'1 / (1 + EXP(-{sum_sql})) <= 0.5 THEN {self.classes[0]} ELSE {self.classes[1]} END ' 
            query += " FROM {}".format(input_table)
            return query
        
        else:
            # sum every class's sql
            query = '(SELECT '
            for i in range(len(self.classes)):
                plus_items = []
                for j in range(len(self.input_features)):
                    col = DBMSUtils.get_delimited_col(dbms, self.input_features[j])
                    plus_item = " {} * {} ".format(col, self.weights[i][j], col)
                    plus_items.append(plus_item)
                sum_sql =  '+'.join(plus_items) + f'+ {self.bias[i]}' 
                query += f'{sum_sql} AS sum_{i},'
            query = query[:-1]
            query += f' FROM {input_table}) as data'
            
            input_table = query
            
            sum_exp_sql = "("
            for i in range(len(self.classes)):
                sum_exp_sql += f"EXP(sum_{i})+"
            sum_exp_sql = sum_exp_sql[:-1]  # remove the last '+'
            sum_exp_sql += ")"
            
            query = '(SELECT '
            for i in range(len(self.classes)):
                query += "(" + "EXP({}{}) / {} ) AS {}{},".format('sum_', i, sum_exp_sql, 'class_', i)
            query = query[:-1]  # remove the last ','
            query = "{}\n FROM {}) as data".format(query, input_table)

            case_stm = "CASE"
            for i in range(len(self.classes)):
                case_stm += " WHEN "
                for j in range(len(self.classes)):
                    if j == i:
                        continue
                    case_stm += "{}{} >= {}{} AND ".format('class_', i, 'class_', j)
                case_stm = case_stm[:-5] # remove the last ' AND '
                case_stm += " THEN {}\n".format(self.classes[i])
            case_stm += "END AS Score"

            final_query = "SELECT {} FROM {}".format(case_stm, query)
            
            return final_query