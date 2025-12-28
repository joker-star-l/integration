import sys
import numpy as np
import pandas as pd
from loguru import logger
from category_encoders import\
    BaseNEncoder, BinaryEncoder, CatBoostEncoder,\
    CountEncoder, HashingEncoder
from sklearn.preprocessing import OneHotEncoder
from craftsman.utility.training_helper import \
    CraftsmanBaseNEncoder, CraftsmanBinaryEncoder, CraftsmanCatBoostEncoder,\
    CraftsmanCountEncoder, CraftsmanHashingEncoder, CraftsmanOneHotEncoder
from craftsman.utility.feature_name_utils import get_set_feature_names_in_out

data = {
    'name': ['Alice',    'Bob',    'Charlie'],
    'city': ['New York', 'London', 'Tokyo'  ],
    'age':  [25,         30,       35       ],
}
features = ['name',
            'city',
            'age',
]
y = [1, 2, 3]
df = pd.DataFrame(data)

def test_CraftsmanBaseNEncoder():
    e0 = CraftsmanBaseNEncoder(base=2)
    p0 = e0.fit_transform(df)
    out00 = list(get_set_feature_names_in_out(e0))
    out01 = list(get_set_feature_names_in_out(e0, features))
    assert out00 == out01
    logger.info(out00)
    assert hasattr(e0, 'feature_names_in_')
    assert hasattr(e0, 'feature_names_out_')
    # print('e0:', e0, f'p0: {type(p0)}', p0, sep='\n')

    e1 = BaseNEncoder(base=2)
    p1 = e1.fit_transform(df)
    out10 = list(get_set_feature_names_in_out(e1))
    out11 = list(get_set_feature_names_in_out(e1, features))
    assert out10 == out11
    logger.info(out10)
    assert hasattr(e1, 'feature_names_in_')
    assert hasattr(e1, 'feature_names_out_')
    # print('e1:', e1, f'p0: {type(p1)}', p1, sep='\n')

def test_CraftsmanBinaryEncoder():
    e0 = CraftsmanBinaryEncoder()
    p0 = e0.fit_transform(df)
    out00 = list(get_set_feature_names_in_out(e0))
    out01 = list(get_set_feature_names_in_out(e0, features))
    assert out00 == out01
    logger.info(out00)
    assert hasattr(e0, 'feature_names_in_')
    assert hasattr(e0, 'feature_names_out_')
    # print('e0:', e0, f'p0: {type(p0)}', p0, sep='\n')

    e1 = BinaryEncoder()
    p1 = e1.fit_transform(df)
    out10 = list(get_set_feature_names_in_out(e1))
    out11 = list(get_set_feature_names_in_out(e1, features))
    assert out10 == out11
    logger.info(out10)
    assert hasattr(e1, 'feature_names_in_')
    assert hasattr(e1, 'feature_names_out_')
    # print('e1:', e1, f'p0: {type(p1)}', p1, sep='\n')

def test_CraftsmanCatBoostEncoder():
    e0 = CraftsmanCatBoostEncoder()
    p0 = e0.fit_transform(df, y)
    out00 = list(get_set_feature_names_in_out(e0))
    out01 = list(get_set_feature_names_in_out(e0, features))
    assert out00 == out01
    logger.info(out00)
    assert hasattr(e0, 'feature_names_in_')
    assert hasattr(e0, 'feature_names_out_')
    # print('e0:', e0, f'p0: {type(p0)}', p0, sep='\n')

    e1 = CatBoostEncoder()
    p1 = e1.fit_transform(df, y)
    out10 = list(get_set_feature_names_in_out(e1))
    out11 = list(get_set_feature_names_in_out(e1, features))
    assert out10 == out11
    logger.info(out10)
    assert hasattr(e1, 'feature_names_in_')
    assert hasattr(e1, 'feature_names_out_')
    # print('e1:', e1, f'p0: {type(p1)}', p1, sep='\n')

def test_CraftsmanCountEncoder():
    e0 = CraftsmanCountEncoder()
    p0 = e0.fit_transform(df)
    out00 = list(get_set_feature_names_in_out(e0))
    out01 = list(get_set_feature_names_in_out(e0, features))
    assert out00 == out01
    logger.info(out00)
    assert hasattr(e0, 'feature_names_in_')
    assert hasattr(e0, 'feature_names_out_')
    # print('e0:', e0, f'p0: {type(p0)}', p0, sep='\n')

    e1 = CountEncoder()
    p1 = e1.fit_transform(df)
    out10 = list(get_set_feature_names_in_out(e1))
    out11 = list(get_set_feature_names_in_out(e1, features))
    assert out10 == out11
    logger.info(out10)
    assert hasattr(e1, 'feature_names_in_')
    assert hasattr(e1, 'feature_names_out_')
    # print('e1:', e1, f'p0: {type(p1)}', p1, sep='\n')

def test_CraftsmanHashingEncoder():
    e0 = CraftsmanHashingEncoder(cols=features)
    p0 = e0.fit_transform(df)
    out00 = list(get_set_feature_names_in_out(e0))
    out01 = list(get_set_feature_names_in_out(e0, features))
    assert out00 == out01
    logger.info(out00)
    assert hasattr(e0, 'feature_names_in_')
    assert hasattr(e0, 'feature_names_out_')
    # print('e0:', e0, f'p0: {type(p0)}', p0, sep='\n')

    e1 = HashingEncoder(cols=features)
    p1 = e1.fit_transform(df)
    out10 = list(get_set_feature_names_in_out(e1))
    out11 = list(get_set_feature_names_in_out(e1, features))
    assert out10 == out11
    logger.info(out10)
    assert hasattr(e1, 'feature_names_in_')
    assert hasattr(e1, 'feature_names_out_')
    # print('e1:', e1, f'p0: {type(p1)}', p1, sep='\n')

logger.remove()
logger.add(sys.stdout, level='INFO')

test_CraftsmanBaseNEncoder()
test_CraftsmanBinaryEncoder()
test_CraftsmanCatBoostEncoder()
test_CraftsmanCountEncoder()
test_CraftsmanHashingEncoder()
