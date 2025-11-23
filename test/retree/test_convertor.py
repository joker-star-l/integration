import os
import onnx
from retree.convertor import ONNXConvertor, SklearnConvertor

def test_new_onnx_convertor():
    convertor = ONNXConvertor()

def test_load_onnx():
    model_path = f'{os.path.dirname(os.path.abspath(__file__))}/data/nyc-taxi-green-dec-2016_t3_d2_l4_n7.onnx'
    model = onnx.load(model_path)
    # model.graph.node[0] = None # does not support assignment 

def test_new_sklearn_convertor():
    convertor = SklearnConvertor()
