# model convertors
from typing import List, Tuple, Callable, Any
import onnx
import numpy as np
from .tree import TreeEnsembleRegressor, DecisionTreeRegressor
from . import sklearn_utils as skutils
from sklearn.tree import _tree as sklearn_tree
from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
SklearnTreeModel = SklearnDecisionTreeRegressor | SklearnDecisionTreeClassifier |\
    SklearnRandomForestRegressor | SklearnRandomForestClassifier

class ONNXConvertor:

    @staticmethod
    def find_model(input_pipeline: onnx.ModelProto) -> onnx.NodeProto | None:
        # 1. 找到所有的 TreeEnsembleRegressor nodes 和 TreeEnsembleClassifier nodes
        nodes = input_pipeline.graph.node
        models = [node for node in nodes if node.op_type in ['TreeEnsembleRegressor', 'TreeEnsembleClassifier']]
        if not models:
            return None

        # 2. 判断 node 的输出是否是 graph 的输出
        pipeline_outputs = [o.name for o in input_pipeline.graph.output]
        models = [
            r for r in models 
            if pipeline_outputs == r.output
        ]
        if not models:
            return None

        # 3. 模型细节判断：非 GBDT、加性模型
        # TODO 排除随机森林多分类器
        assert len(models) == 1
        model = models[0]
        attributes_map = {attr.name: attr for attr in model.attribute}
        if model.op_type == 'TreeEnsembleRegressor':
            if attributes_map.get('base_values') is None and \
                attributes_map['post_transform'].s == b'NONE' and \
                (attributes_map.get('aggregate_function') is None or attributes_map['aggregate_function'].s == b'SUM'):
                return model
            else:
                return None
        if model.op_type == 'TreeEnsembleClassifier':
            if attributes_map.get('base_values') is None and \
                attributes_map['post_transform'].s == b'NONE' and \
                attributes_map.get('classlabels_int64s') is not None:
                return model
            else:
                return None

    @staticmethod
    def from_model(input_model: onnx.NodeProto, func: Callable[[Any], bool]) -> TreeEnsembleRegressor:
        attributes_map = {attr.name: attr for attr in input_model.attribute}
        if input_model.op_type == 'TreeEnsembleRegressor':
            assert attributes_map.get('base_values') is None  # 不是 GBDT
            assert attributes_map['post_transform'].s == b'NONE' # 无后处理
            assert (attributes_map.get('aggregate_function') is None or attributes_map['aggregate_function'].s == b'SUM')  # 加性模型
        elif input_model.op_type == 'TreeEnsembleClassifier':
            assert attributes_map.get('base_values') is None  # 不是 GBDT
            assert attributes_map['post_transform'].s == b'NONE' # 无后处理
            assert attributes_map.get('classlabels_int64s') is not None # 类别为数字
        else:
            raise Exception(f'illegal op type: {input_model.op_type}')

        nodes_falsenodeids = attributes_map['nodes_falsenodeids'].ints
        nodes_featureids = attributes_map['nodes_featureids'].ints
        nodes_hitrates = attributes_map['nodes_hitrates'].floats
        nodes_missing_value_tracks_true = attributes_map['nodes_missing_value_tracks_true'].ints
        nodes_modes = [s.decode() for s in attributes_map['nodes_modes'].strings]
        nodes_nodeids = attributes_map['nodes_nodeids'].ints
        nodes_treeids = attributes_map['nodes_treeids'].ints
        nodes_truenodeids = attributes_map['nodes_truenodeids'].ints
        nodes_values = attributes_map['nodes_values'].floats        
        nodes_tree_intervals = ONNXConvertor.get_tree_intervals(nodes_treeids)
        
        tree_count = len(nodes_tree_intervals)

        target_ids = []
        target_nodeids = []
        target_treeids = []
        target_weights = []

        if input_model.op_type == 'TreeEnsembleRegressor':
            target_ids = attributes_map['target_ids'].ints
            target_nodeids = attributes_map['target_nodeids'].ints
            target_treeids = attributes_map['target_treeids'].ints
            target_weights = [float(func(w * tree_count)) for w in attributes_map['target_weights'].floats]
        elif input_model.op_type == 'TreeEnsembleClassifier':            
            class_ids = attributes_map['class_ids'].ints
            class_nodeids = attributes_map['class_nodeids'].ints
            class_treeids = attributes_map['class_treeids'].ints
            class_weights = [w * tree_count for w in attributes_map['class_weights'].floats]
            classlabels_int64s = attributes_map['classlabels_int64s'].ints
            stride = len(classlabels_int64s)
            if stride == 2:
                stride = 1
            for i in range(0, len(class_ids), stride):
                target_ids.append(class_ids[i])
                target_nodeids.append(class_nodeids[i])
                target_treeids.append(class_treeids[i])
                if stride == 1:
                    label = classlabels_int64s[1] if class_weights[i] > 0.5 else classlabels_int64s[0]                    
                else:
                    label = classlabels_int64s[np.argmax(class_weights[i:i+stride])]
                target_weights.append(float(func(label)))

        target_tree_intervals = ONNXConvertor.get_tree_intervals(target_treeids)
        ensemble = TreeEnsembleRegressor()
        for i in range(tree_count):
            nodes_tree_start, nodes_tree_end = nodes_tree_intervals[i]
            target_tree_start, target_tree_end = target_tree_intervals[i]
            regressor = DecisionTreeRegressor()
            regressor.nodes_falsenodeids = nodes_falsenodeids[nodes_tree_start:nodes_tree_end]
            regressor.nodes_featureids = nodes_featureids[nodes_tree_start:nodes_tree_end]
            regressor.nodes_hitrates = nodes_hitrates[nodes_tree_start:nodes_tree_end]
            regressor.nodes_missing_value_tracks_true = nodes_missing_value_tracks_true[nodes_tree_start:nodes_tree_end]
            regressor.nodes_modes = nodes_modes[nodes_tree_start:nodes_tree_end]
            regressor.nodes_nodeids = nodes_nodeids[nodes_tree_start:nodes_tree_end]
            regressor.nodes_treeids = nodes_treeids[nodes_tree_start:nodes_tree_end]
            regressor.nodes_truenodeids = nodes_truenodeids[nodes_tree_start:nodes_tree_end]
            regressor.nodes_values = nodes_values[nodes_tree_start:nodes_tree_end]
            regressor.target_ids = target_ids[target_tree_start:target_tree_end]
            regressor.target_nodeids = target_nodeids[target_tree_start:target_tree_end]
            regressor.target_treeids = target_treeids[target_tree_start:target_tree_end]
            regressor.target_weights = target_weights[target_tree_start:target_tree_end]
            ensemble.regressors.append(regressor)
        return ensemble

    @staticmethod
    def get_tree_intervals(nodes_treeids: List[int]) -> List[Tuple[int, int]]:
        # 获取每棵树在数组中的区间, 左闭右开
        tree_roots: List[int] = []
        # nodes_treeids is ordered
        next_tree_id = 0
        for i, tree_id in enumerate(nodes_treeids):
            if tree_id == next_tree_id:
                next_tree_id += 1
                tree_roots.append(i)

        tree_intervals: List[Tuple[int, int]] = []
        for i, root in enumerate(tree_roots):
            if i == len(tree_roots) - 1:
                end = len(nodes_treeids)
            else:
                end = tree_roots[i + 1]
            tree_intervals.append((root, end))
        return tree_intervals

    @staticmethod
    def to_model(output_model: TreeEnsembleRegressor, input_model: onnx.NodeProto) -> onnx.NodeProto:
        nodes_falsenodeids = []
        nodes_featureids = []
        nodes_hitrates = []
        nodes_missing_value_tracks_true = []
        nodes_modes = []
        nodes_nodeids = []
        nodes_treeids = []
        nodes_truenodeids = []
        nodes_values = []
        target_ids = []
        target_nodeids = []
        target_treeids = []
        target_weights = []

        for r in output_model.regressors:
            nodes_falsenodeids.extend(r.nodes_falsenodeids)
            nodes_featureids.extend(r.nodes_featureids)
            nodes_hitrates.extend(r.nodes_hitrates)
            nodes_missing_value_tracks_true.extend(r.nodes_missing_value_tracks_true)
            nodes_modes.extend([mode.encode() for mode in r.nodes_modes])
            nodes_nodeids.extend(r.nodes_nodeids)
            nodes_treeids.extend(r.nodes_treeids)
            nodes_truenodeids.extend(r.nodes_truenodeids)
            nodes_values.extend(r.nodes_values)
            target_ids.extend(r.target_ids)
            target_nodeids.extend(r.target_nodeids)
            target_treeids.extend(r.target_treeids)
            target_weights.extend([weight / len(output_model.regressors) for weight in r.target_weights])

        outputs = None
        if input_model.op_type == 'TreeEnsembleRegressor':
            outputs = input_model.output
        elif input_model.op_type == 'TreeEnsembleClassifier':
            outputs = [input_model.output[0]]

        node = onnx.helper.make_node(
            op_type='TreeEnsembleRegressor',
            inputs=input_model.input,
            outputs=outputs,
            name=input_model.name,
            domain='ai.onnx.ml',
            # attributes
            n_targets=1,
            nodes_falsenodeids=nodes_falsenodeids,
            nodes_featureids=nodes_featureids,
            nodes_hitrates=nodes_hitrates,
            nodes_missing_value_tracks_true=nodes_missing_value_tracks_true,
            nodes_modes=nodes_modes,
            nodes_nodeids=nodes_nodeids,
            nodes_treeids=nodes_treeids,
            nodes_truenodeids=nodes_truenodeids,
            nodes_values=nodes_values,
            post_transform=b'NONE',
            target_ids=target_ids,
            target_nodeids=target_nodeids,
            target_treeids=target_treeids,
            target_weights=target_weights
        )
        return node

    @staticmethod
    def to_pipeline(input_pipeline: onnx.ModelProto, output_model: onnx.NodeProto) -> onnx.NodeProto:
        nodes = list(input_pipeline.graph.node)
        for i, node in enumerate(nodes):
            if node.name == output_model.name:
                nodes[i] = output_model
                break
        outputs = [onnx.helper.make_tensor_value_info(
            name=output_model.output[0],
            elem_type=onnx.TensorProto.FLOAT,
            shape=[None, 1]
        )]
        graph = onnx.helper.make_graph(
            nodes=nodes,
            name=input_pipeline.graph.name,
            inputs=input_pipeline.graph.input,
            outputs=outputs,
            initializer=input_pipeline.graph.initializer,
            doc_string=input_pipeline.graph.doc_string,
            value_info=input_pipeline.graph.value_info,
            sparse_initializer=input_pipeline.graph.sparse_initializer
        )
        output_pipeline = onnx.helper.make_model(
            graph=graph,
            opset_imports=input_pipeline.opset_import
        )
        output_pipeline.ir_version = input_pipeline.ir_version
        return output_pipeline


class SklearnConvertor:
    @staticmethod
    def find_model(input_pipeline: SklearnPipeline) -> SklearnTreeModel | None:
        model = input_pipeline.steps[-1][-1]
        if not isinstance(model, SklearnTreeModel):
            return None
        if type(model) in [SklearnDecisionTreeClassifier, SklearnRandomForestClassifier] and not np.issubdtype(model.classes_.dtype, np.number):
            return None
        if type(model) is SklearnRandomForestClassifier and model.n_classes_ != 2:
            return None
        return model

    @staticmethod
    def from_model(input_model: SklearnTreeModel, func: Callable[[Any], bool]) -> TreeEnsembleRegressor:
        ensemble = TreeEnsembleRegressor()
        if type(input_model) in [SklearnDecisionTreeRegressor, SklearnDecisionTreeClassifier]:
            regressor = SklearnConvertor.from_model_single_tree(input_model, func, 0)
            ensemble.regressors.append(regressor)
        elif type(input_model) in [SklearnRandomForestRegressor, SklearnRandomForestClassifier]:
            for i, model in enumerate(input_model.estimators_):
                regressor = SklearnConvertor.from_model_single_tree(model, func, i)
                ensemble.regressors.append(regressor)
        else:
            raise Exception(f'illegal model type: {type(input_model)}')
        return ensemble

    @staticmethod
    def from_model_single_tree(
        input_model: SklearnDecisionTreeRegressor | SklearnDecisionTreeClassifier,
        func: Callable[[Any], bool],
        tree_id: int
    ) -> DecisionTreeRegressor:
        tree = input_model.tree_
        node_count = len(tree.feature)
        leaf_count = len([id for id in tree.feature if id == -2])
        regressor = DecisionTreeRegressor()
        regressor.nodes_falsenodeids = [0 if id == -1 else id for id in tree.children_right]
        regressor.nodes_featureids = [0 if id == -2 else id for id in tree.feature]
        regressor.nodes_hitrates = tree.n_node_samples
        regressor.nodes_missing_value_tracks_true = tree.missing_go_to_left
        regressor.nodes_modes = ['LEAF' if id == -2 else 'BRANCH_LEQ' for id in tree.feature]
        regressor.nodes_nodeids = [ii for ii in range(node_count)]
        regressor.nodes_treeids = [tree_id] * node_count
        regressor.nodes_truenodeids = [0 if id == -1 else id for id in tree.children_left]
        regressor.nodes_values = [0.0 if regressor.nodes_modes[ii] == 'LEAF' else v for (ii, v) in enumerate(tree.threshold)]
        regressor.target_ids = [0] * leaf_count
        regressor.target_nodeids = [ii for (ii, id) in enumerate(tree.feature) if id == -2]
        regressor.target_treeids = [tree_id] * leaf_count
        if type(input_model) is SklearnDecisionTreeRegressor:
            regressor.target_weights = [float(func(tree.value[id][0][0])) for id in regressor.target_nodeids]
        elif type(input_model) is SklearnDecisionTreeClassifier:
            labels = input_model.classes_
            regressor.target_weights = [float(func(labels[np.argmax(tree.value[id][0])])) for id in regressor.target_nodeids]
        return regressor

    @staticmethod
    def to_model(output_model: TreeEnsembleRegressor, input_model: SklearnTreeModel) -> SklearnTreeModel:
        if type(input_model) in [SklearnDecisionTreeRegressor, SklearnDecisionTreeClassifier]:
            return SklearnConvertor.to_model_single_tree(output_model.regressors[0], input_model)
        if type(input_model) in [SklearnRandomForestRegressor, SklearnRandomForestClassifier]:
            estimator_count = input_model.n_estimators
            ensemble = SklearnRandomForestRegressor(n_estimators=estimator_count)
            ensemble.n_outputs_ = 1
            ensemble.n_features_in_ = input_model.n_features_in_
            ensemble.feature_names_in_ = input_model.feature_names_in_
            ensemble.estimators_ = [None] * estimator_count
            for i in range(estimator_count):
                ensemble.estimators_[i] = SklearnConvertor.to_model_single_tree(output_model.regressors[i], input_model.estimators_[i])
            return ensemble
        raise Exception(f'illegal model type: {type(input_model)}')

    @staticmethod
    def to_model_single_tree(
        regressor: DecisionTreeRegressor, 
        input_model: SklearnDecisionTreeRegressor | SklearnDecisionTreeClassifier
    ) -> SklearnDecisionTreeRegressor:
        node_count = len(regressor.nodes_modes)
        assert input_model.tree_.n_outputs == 1
        sktree = sklearn_tree.Tree(input_model.tree_.n_features, np.array([1]), 1)  # Tree(n_features, n_classes, n_outputs)
        sknodes = np.ndarray(shape=node_count, dtype=skutils.Node)
        parents: List[Tuple[int, str] | None] = [None] * node_count
        for (pid, lcid) in enumerate(regressor.nodes_truenodeids):
            parents[lcid] = (pid, 'L')
        for (pid, rcid) in enumerate(regressor.nodes_falsenodeids):
            parents[rcid] = (pid, 'R')
        node_to_weight = {k : v for (k, v) in zip(regressor.target_nodeids, regressor.target_weights)}
        for i in range(node_count):
            sknodes[i] = skutils.Node(
                sklearn_tree.TREE_UNDEFINED if parents[i] is None else parents[i][0],
                parents[i] is not None and parents[i][1] == 'L',
                regressor.nodes_modes[i] == 'LEAF',
                regressor.nodes_featureids[i],
                regressor.nodes_values[i],
                0.0,
                regressor.nodes_hitrates[i],
                float(regressor.nodes_hitrates[i]),
                bool(regressor.nodes_missing_value_tracks_true[i]),
                node_to_weight.get(i, 0.0)
            )
        skutils.init_tree(sktree, node_count, input_model.max_depth, sknodes)
        result_model = SklearnDecisionTreeRegressor()
        result_model.tree_ = sktree
        result_model.n_outputs_ = 1
        result_model.n_features_in_ = input_model.n_features_in_
        # 随机森林中的单个决策树模型没有 feature_names_in_ 属性
        if hasattr(input_model, 'feature_names_in_'):
            result_model.feature_names_in_ = input_model.feature_names_in_
        return result_model

    @staticmethod
    def to_pipeline(input_pipeline: SklearnPipeline, output_model: SklearnTreeModel) -> SklearnPipeline:
        output_pipeline = SklearnPipeline([e for e in input_pipeline.steps])
        name = output_pipeline.steps[-1][0]
        output_pipeline.steps[-1] = (name, output_model)
        return output_pipeline
