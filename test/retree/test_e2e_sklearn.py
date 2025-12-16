import os
import sys
import pandas as pd
from loguru import logger
from retree.util import timer
from sklearn.model_selection import train_test_split
from retree.convertor import *
from timer_funcs import *

_DT_RG = '_dt_rg'
_DT_CF = '_dt_cf'
_RF_RG = '_rf_rg'
_RF_CF = '_rf_cf'

def train(X_train, y_train, model_type):
    if model_type == _DT_RG:
        model = SklearnDecisionTreeRegressor(max_depth=10)
    elif model_type == _DT_CF:
        model = SklearnDecisionTreeClassifier(max_depth=10)
    elif model_type == _RF_RG:
        model = SklearnRandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=100)
    elif model_type == _RF_CF:
        model = SklearnRandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=100)
    pipeline = SklearnPipeline([('model', model)])
    pipeline.fit(X_train, y_train)
    return pipeline

@timer
def execute_retree(pipeline, func, detail=True):
    model = SklearnConvertor_find_model(pipeline)
    assert model is not None
    ensemble = SklearnConvertor_from_model(model, func)
    ensemble = T_process(ensemble, 4, detail)
    out_model = SklearnConvertor_to_model(ensemble, model)
    out_pipeline = SklearnConvertor_to_pipeline(pipeline, out_model)
    return out_pipeline

@timer
def test_e2e_DT_RG():
    logger.info(f'test_e2e{_DT_RG}')
    data_path = f'{os.path.dirname(os.path.abspath(__file__))}/data/nyc-taxi-green-dec-2016.csv'
    df = pd.read_csv(data_path)
    X = df.drop(columns=['tipamount'])
    y = df['tipamount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=42)
    y_train = y_train.values
    y_test = y_test.values
    func = lambda x: x > 0.99
    pipeline0 = train(X_train, y_train, _DT_RG)
    pred0 = func(pipeline0.predict(X_test)).astype(np.float64)
    pipeline1 = execute_retree(pipeline0, func)
    pred1 = pipeline1.predict(X_test)
    for (p0, p1) in zip(pred0, pred1):
        assert p0 == p1

@timer
def test_e2e_DT_CF():
    logger.info(f'test_e2e{_DT_CF}')
    data_path = f'{os.path.dirname(os.path.abspath(__file__))}/data/wine_quality.csv'
    df = pd.read_csv(data_path)
    X = df.drop(columns=['quality'])
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
    y_train = y_train.values
    y_test = y_test.values
    func = lambda x: x == 6
    pipeline0 = train(X_train, y_train, _DT_CF)
    pred0 = pipeline0.predict(X_test)
    pred0 = func(pred0).astype(np.float64)
    pipeline1 = execute_retree(pipeline0, func)
    pred1 = pipeline1.predict(X_test)
    for (p0, p1) in zip(pred0, pred1):
        assert p0 == p1

@timer
def test_e2e_RF_RG():
    logger.info(f'test_e2e{_RF_RG}')
    data_path = f'{os.path.dirname(os.path.abspath(__file__))}/data/nyc-taxi-green-dec-2016.csv'
    df = pd.read_csv(data_path)
    X = df.drop(columns=['tipamount'])
    y = df['tipamount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=42)
    y_train = y_train.values
    y_test = y_test.values
    func = lambda x: x > 0.99
    pipeline0 = train(X_train, y_train, _RF_RG)
    pred0 = [func(e.predict(X_test.values)).astype(np.float64) for e in pipeline0.steps[-1][-1].estimators_]
    pipeline1 = execute_retree(pipeline0, func, False)
    pred1 = [e.predict(X_test.values) for e in pipeline1.steps[-1][-1].estimators_]
    for (pr0, pr1) in zip(pred0, pred1):
        for (p0, p1) in zip(pr0, pr1):
            assert int(p0) == int(p1)

logger.remove()
logger.add(sys.stdout, level='INFO')

test_e2e_DT_RG()
test_e2e_DT_CF()
test_e2e_RF_RG()
