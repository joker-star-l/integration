from multiprocessing import Pool
import onnx
from retree.util import timer
from retree.convertor import SklearnConvertor, ONNXConvertor
from retree.collapse import collapse_single_tree
from retree.recombination import recombine_single_tree

@timer
def SklearnConvertor_find_model(pipeline):
    return SklearnConvertor.find_model(pipeline)

@timer
def SklearnConvertor_from_model(model, func):
    return SklearnConvertor.from_model(model, func)

@timer
def SklearnConvertor_to_model(ensemble, model):
    return SklearnConvertor.to_model(ensemble, model)

@timer
def SklearnConvertor_to_pipeline(pipeline, model):
    return SklearnConvertor.to_pipeline(pipeline, model)

@timer
def ONNXConvertor_find_model(pipeline):
    return ONNXConvertor.find_model(pipeline)

@timer
def ONNXConvertor_from_model(model, func):
    return ONNXConvertor.from_model(model, func)

@timer
def ONNXConvertor_to_model(ensemble, model):
    return ONNXConvertor.to_model(ensemble, model)

@timer
def ONNXConvertor_to_pipeline(pipeline, model):
    return ONNXConvertor.to_pipeline(pipeline, model)

@timer
def T_collapse_single_tree(model):
    return collapse_single_tree(model)

@timer
def T_recombine_single_tree(model):
    return recombine_single_tree(model)

@timer
def T_process_single_tree(model):
    model = T_collapse_single_tree(model)
    model = T_recombine_single_tree(model)
    return model

def _process_single_tree(model):
    model = collapse_single_tree(model)
    model = recombine_single_tree(model)
    return model

@timer
def T_process(input_model, threads_count, detail=True):
    threads_count = min(threads_count, len(input_model.regressors))
    with Pool(threads_count) as pool:
        func = T_process_single_tree if detail else _process_single_tree
        input_model.regressors = pool.map(func, input_model.regressors)
    return input_model

@timer
def T_check_node(model, pipeline):
    ctx = onnx.checker.DEFAULT_CONTEXT
    ctx.opset_imports = {opset.domain: opset.version for opset in pipeline.opset_import}
    return onnx.checker.check_node(model)

@timer
def T_check_model(pipeline):
    return onnx.checker.check_model(pipeline)
