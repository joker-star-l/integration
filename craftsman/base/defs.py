from enum import Enum
from pandas import DataFrame

DBMS = 'duckdb'
TABLE_NAME = 'car_price'
JUST_PUSH_FLAG = False

def set_JUST_PUSH_FLAG(flag):
    global JUST_PUSH_FLAG
    JUST_PUSH_FLAG = flag
SAMPLE_RATE = 1
PREPROCESS_PACKAGE_PATH = 'craftsman.preprocess.'
MODEL_PACKAGE_PATH = 'craftsman.model.'

class DataType(Enum):
    CON = 1
    CAT = 2


class CalculationType(Enum):
    ARITHMETIC = 1
    COMPARISON = 2
    MIXED = 3


class OperatorType(Enum):
    # CON_A_CON = 1
    # CON_C_CAT = 2
    # CON_M_CON = 3
    # CAT_C_CAT = 4
    # EXPAND = 5
    # IMPUTE = 6
    # TEMP = 7 # con-c-cat fuse expand
    CON_A_CON = 'con_a_con'
    CON_C_CAT = 'con_c_cat'
    CON_M_CON = 'con_m_con'
    CAT_C_CAT = 'cat_c_cat'
    EXPAND = 'expand'
    IMPUTE = 'impute'
    TEMP = 'temp' # con-c-cat fuse expand
    CON_S_CON = 'con_s_con'


class OperatorName(Enum):
    # CON_A_CON
    MINMAXSCALER = 'MinMaxScaler'
    STANDARDSCALER = 'StandardScaler'
    ROBUSTSCALER = 'RobustScaler'
    NORMALIZER = 'Normalizer'

    # CON_S_CON
    QUANTILETRANSFORMER = 'QuantileTransformer'
    POWERTRANSFORMER = 'PowerTransformer'

    # CON_C_CAT
    KBINSDISCRETIZER = 'KBinsDiscretizer'
    BINARIZERDISCRETIZER = 'BinarizerDiscretizer'
    CATBOOSTENCODER = 'CatBoostEncoder'
    # CON_M_CON


    # CAT_C_CAT
    COUNTENCODER = 'CountEncoder'
    LEAVEONEOUTENCODER = 'LeaveOneOutEncoder'
    ORDINALENCODER = 'OrdinalEncoder'
    TARGETENCODER = 'TargetEncoder'
    LABELENCODER = 'LabelEncoder'

    # EXPAND
    ONEHOTENCODER = 'OneHotEncoder'
    BINARYENCODER = 'BinaryEncoder'
    BASENENCODER = 'BaseNEncoder'
    HASHINGENCODER = 'HashingEncoder'

    # IMPUTE
    SIMPLEIMPUTER = 'SimpleImputer'

    # Merged Operator
    CON_C_CAT_Merged_OP = 'con_c_cat_merged_op'
    EXPAND_Merged_OP = 'expand_merged_op'
    CON_A_CON_Merged_OP = 'con_a_con_merged_op'
    CAT_C_CAT_Merged_OP = 'cat_c_cat_merged_op'


class ModelName(Enum):
    RANDOMFORESTCLASSIFIER = 'RandomForestClassifier'
    DECISIONTREECLASSIFIER = 'DecisionTreeClassifier'
    DECISIONTREEREGRESSOR = 'DecisionTreeRegressor'
    RANDOMFORESTREGRESSOR = 'RandomForestRegressor'
    LINEARREGRESSION = 'LinearRegression'
    LOGISTICREGRESSION = 'LogisticRegression'


EXPAND_JOIN_POSTNAME = '_expand'
CAT_C_CAT_JOIN_POSTNAME = '_cat_c_cat'
CAT_C_CAT_JOIN_COL_POSTNAME = '_cat_c_cat_col'


class DBDataType(Enum):
    VARCHAR = 'VARCHAR'
    VARCHAR512 = 'VARCHAR(512)'
    SMALLINT = 'SMALLINT'
    INT = 'INT'
    FLOAT = 'FLOAT'
    BOOLEAN = 'BOOLEAN'
    
    
class SQLPlanType(Enum):
    CASE = 'case'
    FUSION = 'fusion'
    JOIN = 'join'
    PUSH = 'push'
    
class PrimitiveType(Enum):
    IN = 'in'
    OR = 'or'
    LE = 'le'
    # CASE = 'case'
    EQUAL = 'equal'
    INEQUAL = 'inequal'
    GE = 'ge'
    LE_EQ = 'le_eq'
    GE_EQ = 'ge_eq'
    

def in_cost_func_pg(length):
    if length == 1:
        return 3.4
    else:
        return 1.7 * length
    
def in_cost_func_duckdb(length):
    if length <= 4:
        return 0.0000116 * length + 0.000017833
    else:
        return 0.0000031144 * length + 0.0030752
    
def in_cost_func_monetdb(length):
    return 0.098055 * length + 0.4922019

class PrimitiveCost(Enum):
    

    # pg
    # IN = in_cost_func_pg
    # OR = 25.5 # OR = GE_EQ + LE
    # LE = 12.75
    # EQUAL = 3.4
    # INEQUAL = 3.4
    # GE = 12.75
    # LE_EQ = 12.75
    # GE_EQ = 12.75
    
    
    # duckdb
    IN = in_cost_func_duckdb
    OR = 0.00000991 # OR = GE_EQ + LE
    LE = 0.00000485
    EQUAL = 0.00000843
    INEQUAL = 0.00000846
    GE = 0.0000051
    LE_EQ = 0.000005126
    GE_EQ = 0.00000506
    
    # monetdb
    # IN = in_cost_func_monetdb
    # OR = 0.419532
    # LE = 0.211766
    # EQUAL = 0.241633
    # INEQUAL = 0.278633
    # GE = 0.214733
    # LE_EQ = 0.2356
    # GE_EQ = 0.207766

# class PrimitiveCost(Enum):
#     IN = None
#     OR = None
#     LE = None
#     EQUAL = None
#     INEQUAL = None
#     GE = None
#     LE_EQ = None
#     GE_EQ = None


# if DBMS == 'postgresql':
#     PrimitiveCost.IN = in_cost_func_pg
#     PrimitiveCost.OR = 25.5
#     PrimitiveCost.LE = 12.75
#     PrimitiveCost.EQUAL = 3.4
#     PrimitiveCost.INEQUAL = 3.4
#     PrimitiveCost.GE = 12.75
#     PrimitiveCost.LE_EQ = 12.75
#     PrimitiveCost.GE_EQ = 12.75
# elif DBMS == 'duckdb':
#     PrimitiveCost.IN = in_cost_func_duckdb
#     PrimitiveCost.OR = 0.00000991
#     PrimitiveCost.LE = 0.00000485
#     PrimitiveCost.EQUAL = 0.00000843
#     PrimitiveCost.INEQUAL = 0.00000846
#     PrimitiveCost.GE = 0.0000051
#     PrimitiveCost.LE_EQ = 0.000005126
#     PrimitiveCost.GE_EQ = 0.00000506
    
ORDER_WHEN = True
EXPRIMENT_METHOD = None
EXPRIMENT_COL = None

CURRENT_PYTHON_FILE = 'input_your_python_file'

PUSH_USE_AVERAGE = False

COST_DIR_PATH = '/home/postgres/craftsman_experiments/sql_result_cost/push_not_use_average/duckdb'
# COST_DIR_PATH = '/home/postgres/craftsman_experiments/sql_result_cost/push_not_use_average/pg2'
# COST_DIR_PATH = '/home/postgres/craftsman_experiments/sql_result_cost/push_not_use_average/monetdb'

TIMER = 'OFF' # ON/OFF

GROUP = None

MASQ = False

AUTO_RULE_GEN = False

PIPELINE_FEATURES_IN = []

HASHING_ENCODER_N_COMPONENTS = 1000
HASHING_ENCODER_FEATURE = 'col'

value_counts_sum_cache = {}
tree_node_in_length = {}
tree_node_or_length = {}

rule_table_content = [
    ["uncertain", "uncertain", "disable", "disable", "disable", "disable", "uncertain"],
    ["apply", "apply", "apply", "apply", "uncertain", "disable", "uncertain"],
    ["apply", "apply", "apply", "apply", "simply", "simply", "uncertain"],
    ["apply", "apply", "apply", "apply", "simply", "simply", "disable"],
    ["apply", "apply", "apply", "apply", "disable", "disable", "uncertain"],
    ["apply", "apply", "apply", "apply", "disable", "disable", "disable"],
    ["disable", "disable", "disable", "disable", "disable", "disable", "disable"]
]

implementation_table_content = [
    [SQLPlanType.CASE, SQLPlanType.CASE, None, None, None, None, None],
    [SQLPlanType.CASE, SQLPlanType.CASE, SQLPlanType.CASE, SQLPlanType.CASE, SQLPlanType.CASE, None, None],
    [SQLPlanType.CASE, SQLPlanType.CASE, SQLPlanType.CASE, SQLPlanType.CASE, SQLPlanType.CASE, SQLPlanType.JOIN, None],
    [SQLPlanType.JOIN, SQLPlanType.JOIN, SQLPlanType.JOIN, SQLPlanType.JOIN, SQLPlanType.CASE, SQLPlanType.JOIN, None],
    [SQLPlanType.CASE, SQLPlanType.CASE, SQLPlanType.CASE, SQLPlanType.CASE, None, None, None],
    [SQLPlanType.JOIN, SQLPlanType.JOIN, SQLPlanType.JOIN, SQLPlanType.JOIN, None, None, None],
    [None, None, None, None, None, None, None]
]

rule_table_index_columns = [
    OperatorType.CON_A_CON.value + SQLPlanType.CASE.value,
    OperatorType.CON_C_CAT.value + SQLPlanType.CASE.value,
    OperatorType.CAT_C_CAT.value + SQLPlanType.CASE.value,
    OperatorType.CAT_C_CAT.value + SQLPlanType.JOIN.value,
    OperatorType.EXPAND.value + SQLPlanType.CASE.value,
    OperatorType.EXPAND.value + SQLPlanType.JOIN.value,
    'Tree'
]

rule_table = DataFrame(
    rule_table_content,
    index=rule_table_index_columns,
    columns=rule_table_index_columns
)

implementation_table = DataFrame(
    implementation_table_content,
    index=rule_table_index_columns,
    columns=rule_table_index_columns
)