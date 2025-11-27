import os
import sys
from multiprocessing import Pool
import onnx
import onnxruntime as ort
import pandas as pd
import numpy as np
from loguru import logger
from retree.convertor import ONNXConvertor, SklearnConvertor
from retree.rewriter import rewrite_model
from retree.collapse import collapse_single_tree
from retree.recombination import recombine_single_tree
from retree.util import timer

# 随机森林回归器（3 棵树）
_pipeline_1_name = 'nyc-taxi-green-dec-2016_t3_d2_l4_n7.onnx'
_func_1 = lambda x: x > 0.99
# 随机森林回归器（100 棵树）
_pipeline_2_name = 'nyc-taxi-green-dec-2016_t100_d10_l842_n1684.onnx'
_func_2 = lambda x: x > 0.99
# 决策树回归器
_pipeline_3_name = 'nyc-taxi-green-dec-2016_d10_l376_n751.onnx'
_pipeline_3_out_name = 'nyc-taxi-green-dec-2016_d10_l376_n751_out.onnx'
_func_3 = lambda x: x > 0.99

@timer
def ONNXConvertor_find_model(pipeline):
    return ONNXConvertor.find_model(pipeline)

@timer
def ONNXConvertor_from_model(model):
    return ONNXConvertor.from_model(model)

@timer
def ONNXConvertor_to_model(ensemble, model):
    return ONNXConvertor.to_model(ensemble, model)

@timer
def ONNXConvertor_to_pipeline(pipeline, model):
    return ONNXConvertor.to_pipeline(pipeline, model)

@timer
def _rewrite_model(model, func):
    return rewrite_model(model, func)

@timer
def _collapse_single_tree(model):
    return collapse_single_tree(model)

@timer
def _recombine_single_tree(model):
    return recombine_single_tree(model)

@timer
def _process_single_tree(model):
    model = _collapse_single_tree(model)
    model = _recombine_single_tree(model)
    return model

@timer
def _process(input_model, threads_count):
    threads_count = min(threads_count, len(input_model.regressors))
    with Pool(threads_count) as pool:
        input_model.regressors = pool.map(_process_single_tree, input_model.regressors)
    return input_model

@timer
def _check_node(model, pipeline):
    ctx = onnx.checker.DEFAULT_CONTEXT
    ctx.opset_imports = {opset.domain: opset.version for opset in pipeline.opset_import}
    return onnx.checker.check_node(model)

@timer
def _check_model(pipeline):
    return onnx.checker.check_model(pipeline)

@timer
def test_e2e_1():
    logger.info('test_e2e_1')
    pipeline_path = f'{os.path.dirname(os.path.abspath(__file__))}/data/{_pipeline_1_name}'
    pipeline = onnx.load(pipeline_path)
    model = ONNXConvertor_find_model(pipeline)
    ensemble = ONNXConvertor_from_model(model)
    _rewrite_model(ensemble, _func_1)
    ensemble = _process(ensemble, 4)
    out_model = ONNXConvertor_to_model(ensemble, model)
    _check_node(out_model, pipeline)
    out_pipeline = ONNXConvertor_to_pipeline(pipeline, out_model)
    _check_model(out_pipeline)
    out_model = ONNXConvertor_find_model(out_pipeline)
    ensemble = ONNXConvertor_from_model(out_model)
    assert len(ensemble.regressors) == 3
    tree0 = ensemble.regressors[0]
    assert tree0.nodes_falsenodeids == [2, 0, 0]
    assert tree0.nodes_featureids == [2, 0, 0]
    assert tree0.nodes_hitrates == [331273.0, 201728.0, 129545.0]
    assert tree0.nodes_missing_value_tracks_true == [0, 0, 0]
    assert tree0.nodes_modes == ['BRANCH_LEQ', 'LEAF', 'LEAF']
    assert tree0.nodes_nodeids == [0, 1, 2]
    assert tree0.nodes_treeids == [0, 0, 0]
    assert tree0.nodes_truenodeids == [1, 0, 0]
    assert [round(v, 3) for v in tree0.nodes_values] == [15.325, 0.0, 0.0]
    assert tree0.target_ids == [0, 0]
    assert tree0.target_nodeids == [1, 2]
    assert tree0.target_treeids == [0, 0]
    assert [round(v, 3) for v in tree0.target_weights] == [0.0, 1.0]

@timer
def test_e2e_2():
    logger.info('test_e2e_2')
    pipeline_path = f'{os.path.dirname(os.path.abspath(__file__))}/data/{_pipeline_2_name}'
    pipeline = onnx.load(pipeline_path)
    model = ONNXConvertor_find_model(pipeline)
    ensemble = ONNXConvertor_from_model(model)
    _rewrite_model(ensemble, _func_2)
    ensemble = _process(ensemble, 4)
    out_model = ONNXConvertor_to_model(ensemble, model)
    _check_node(out_model, pipeline)
    out_pipeline = ONNXConvertor_to_pipeline(pipeline, out_model)
    _check_model(out_pipeline)

@timer
def test_e2e_3():
    logger.info('test_e2e_3')
    pipeline_path = f'{os.path.dirname(os.path.abspath(__file__))}/data/{_pipeline_3_name}'
    pipeline = onnx.load(pipeline_path)
    model = ONNXConvertor_find_model(pipeline)
    ensemble = ONNXConvertor_from_model(model)
    _rewrite_model(ensemble, _func_3)
    ensemble = _process(ensemble, 1)
    out_model = ONNXConvertor_to_model(ensemble, model)
    _check_node(out_model, pipeline)
    out_pipeline = ONNXConvertor_to_pipeline(pipeline, out_model)
    _check_model(out_pipeline)
    out_pipeline_path = f'{os.path.dirname(os.path.abspath(__file__))}/data/{_pipeline_3_out_name}'
    onnx.save_model(out_pipeline, out_pipeline_path)

    data_path = f'{os.path.dirname(os.path.abspath(__file__))}/data/nyc-taxi-green-dec-2016.csv'
    df = pd.read_csv(data_path)
    df.drop('tipamount', axis=1, inplace=True)
    data = df.values.astype(np.float32)

    input_name = 'float_input'
    output_name = 'variable'
    ses = ort.InferenceSession(pipeline_path, providers=['CPUExecutionProvider'])
    result0 = ses.run([output_name], {input_name: data})[0].reshape(-1)
    ses = ort.InferenceSession(out_pipeline_path, providers=['CPUExecutionProvider'])
    result1 = ses.run([output_name], {input_name: data})[0].reshape(-1)
    sum0 = _func_3(result0).sum()
    sum1 = int(result1.sum())
    logger.info(f'sum0:{sum0}, sum1:{sum1}')
    assert sum0 == sum1, f'sum0:{sum0} != sum1:{sum1}'

logger.remove()
logger.add(sys.stdout, level='INFO')

test_e2e_1()
test_e2e_2()
test_e2e_3()
