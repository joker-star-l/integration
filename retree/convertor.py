# model convertors
from typing import List, Tuple
import onnx
from .tree import TreeEnsembleRegressor, DecisionTreeRegressor
from sklearn import tree as sklearn_tree, ensemble as sklearn_ensemble, pipeline as sklearn_pipeline
SklearnPipeline = sklearn_pipeline.Pipeline
SklearnTreeModel = sklearn_tree.DecisionTreeClassifier | sklearn_tree.DecisionTreeRegressor |\
    sklearn_ensemble.RandomForestClassifier | sklearn_ensemble.RandomForestRegressor

class ONNXConvertor:

    @staticmethod
    def find_model(input_pipeline: onnx.ModelProto) -> onnx.NodeProto | None:
        # 1. 找到所有的 TreeEnsembleRegressor nodes
        nodes = input_pipeline.graph.node
        models = [node for node in nodes if node.op_type == 'TreeEnsembleRegressor']
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
        model = models[0]
        attributes_map = {attr.name: attr for attr in model.attribute}
        if attributes_map.get('base_values') == None and attributes_map.get('aggregate_function') in [None, 'SUM']:
            return model
        else:
            return None

    @staticmethod
    def from_model(input_model: onnx.NodeProto) -> TreeEnsembleRegressor:
        assert input_model.op_type == 'TreeEnsembleRegressor'
        attributes_map = {attr.name: attr for attr in input_model.attribute}
        assert attributes_map.get('base_values') == None  # 不是 GBDT
        assert attributes_map.get('aggregate_function') in [None, 'SUM']  # 加性模型

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

        target_ids = attributes_map['target_ids'].ints
        target_nodeids = attributes_map['target_nodeids'].ints
        target_treeids = attributes_map['target_treeids'].ints
        target_weights = [w * tree_count for w in attributes_map['target_weights'].floats]
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

        node = onnx.helper.make_node(
            op_type='TreeEnsembleRegressor',
            inputs=input_model.input,
            outputs=input_model.output,
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
        graph = onnx.helper.make_graph(
            nodes=nodes,
            name=input_pipeline.graph.name,
            inputs=input_pipeline.graph.input,
            outputs=input_pipeline.graph.output,
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
        pass


    @staticmethod
    def from_model(input_model: SklearnTreeModel) -> TreeEnsembleRegressor:
        pass

    @staticmethod
    def to_model(output_model: TreeEnsembleRegressor, input_model: SklearnTreeModel) -> SklearnTreeModel:
        pass

    @staticmethod
    def to_pipeline(input_pipeline: SklearnPipeline, output_model: SklearnTreeModel) -> SklearnPipeline:
        pass
