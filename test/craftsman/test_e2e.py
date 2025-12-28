import os
import sys
import time
import pandas as pd
from loguru import logger
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from craftsman.utility.loader import save_model
from craftsman.base.defs import OperatorName, ModelName
import craftsman.base.defs as defs
from craftsman.transformer_manager import TransformerManager
# #{use craftsman
# from craftsman.utility.training_helper import *
# #}
##{do not use craftsman
from sklearn.compose import ColumnTransformer as CraftsmanColumnTransformer
from sklearn.impute import SimpleImputer as CraftsmanSimpleImputer
from sklearn.preprocessing import \
    LabelEncoder as CraftsmanLabelEncoder,\
    KBinsDiscretizer as CraftsmanKBinsDiscretizer,\
    OrdinalEncoder as CraftsmanOrdinalEncoder,\
    OneHotEncoder as CraftsmanOneHotEncoder
from category_encoders import \
    TargetEncoder as CraftsmanTargetEncoder,\
    CountEncoder as CraftsmanCountEncoder,\
    LeaveOneOutEncoder as CraftsmanLeaveOneOutEncoder,\
    BinaryEncoder as CraftsmanBinaryEncoder,\
    CatBoostEncoder as CraftsmanCatBoostEncoder,\
    BaseNEncoder as CraftsmanBaseNEncoder,\
    HashingEncoder as CraftsmanHashingEncoder
##}

def test_e2e_car_price():
    data_root = f'{os.path.dirname(os.path.abspath(__file__))}/data/car_price/'
    train_data_path = data_root + 'train.csv'
    test_data_path = data_root + 'test.csv'
    pipeline_save_path = data_root + 'car_price.joblib'
    generated_file_path = data_root + 'car_price.sql'

    # load dataset
    data = pd.read_csv(train_data_path)
    y = data["Price"]
    X = data.drop(["Price"], axis=1)

    columns = X.columns.tolist()

    ordinal_cols = ['Owner_Type']
    binary_cols = ['Location', 'Fuel_Type', 'Transmission', 'Name']
    kbins_cols = ['Year', 'Kilometers_Driven', 'Engine', 'Power', 'Mileage', 'Seats']
    count_cols = ['Brand']

    all_cols =  ordinal_cols + binary_cols + kbins_cols + count_cols
    X = X[all_cols]

    # define preprocessors
    type_categories = ["First", "Second", "Third", "Fourth & Above"]
    ordinal_encoder = CraftsmanOrdinalEncoder(categories=[type_categories])
    kbins = CraftsmanKBinsDiscretizer(encode="ordinal", n_bins=15)
    binary_encoder = CraftsmanBinaryEncoder()
    imputer = CraftsmanSimpleImputer(strategy="most_frequent")
    count_encoder = CraftsmanCountEncoder()
    kbins2 = CraftsmanKBinsDiscretizer(encode="ordinal", n_bins=15)

    # define model
    dt = DecisionTreeRegressor(max_depth=6, random_state=24)

    # define steps
    X_copy = X.copy()

    # X_copy = imputer.fit_transform(X_copy)

    transformer1 = CraftsmanColumnTransformer(
        remainder="passthrough",
        transformers=[
            (
                OperatorName.ORDINALENCODER.value,
                ordinal_encoder,
                ordinal_cols,
            ),
            (
                OperatorName.BINARYENCODER.value,
                binary_encoder,
                binary_cols,
            ),
            (
                OperatorName.KBINSDISCRETIZER.value,
                kbins,
                kbins_cols,
            ),
            (
                OperatorName.COUNTENCODER.value,
                count_encoder,
                count_cols,
            ),
        ],
        verbose_feature_names_out=False # must set to False!
        # input_data=X_copy
    )

    # X_copy = transformer1.fit_transform(X_copy, y)

    transformer2 = CraftsmanColumnTransformer(
        remainder="passthrough",
        transformers=[
            (
                OperatorName.KBINSDISCRETIZER.value,
                kbins2,
                ['Year','Kilometers_Driven','Engine'],
            )
        ],
        verbose_feature_names_out=False
        # input_data=X_copy
    )

    # compose pipline
    imputer.set_output(transform='pandas')
    transformer1.set_output(transform='pandas')
    transformer2.set_output(transform='pandas')
    pipeline = Pipeline(
        steps=[
            ('Imputer', imputer),
            ("step2", transformer1),
            ("step3", transformer2),        
            (ModelName.DECISIONTREEREGRESSOR.value, dt)
        ]
    )

    # which database dialect
    pipeline.data_rows = len(X)
    # training dataset amount
    defs.DBMS = 'duckdb'

    # train model
    pipeline.fit(X, y)

    # save model to the file
    save_model(pipeline, pipeline_save_path)
    logger.info(f'Pipeline has been saved at: {pipeline_save_path}')

    # test model
    data_test = pd.read_csv(test_data_path)
    y_test = data_test["Price"]
    X_test = data_test.drop("Price", axis=1)
    X_test = X_test[all_cols]

    # evaluate the test result
    y_predict = pipeline.predict(X_test)

    manager = TransformerManager()
    table_name = "car_price"
    dbms = 'duckdb'
    pre_sql = "EXPLAIN ANALYZE "
    group = 'prune'

    t1 = time.time()
    query = manager.generate_query(
        pipeline_save_path,
        table_name,
        dbms,
        pre_sql=pre_sql,
        group=group,
        cost_model='craftsman'
    )
    t2 = time.time()
    logger.info(f'total compile time: {(t2-t1):.2f}s')

    with open(generated_file_path, "w") as sql_file:
        sql_file.write(query)
    logger.info(f'Generate SQL file have been saved at: {generated_file_path}')


logger.remove()
logger.add(sys.stdout, level='INFO')

test_e2e_car_price()
