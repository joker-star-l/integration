# model convertors
import onnx
from .tree import TreeEnsembleRegressor
from sklearn import tree as sklearn_tree, ensemble as sklearn_ensemble, pipeline as sklearn_pipeline
SklearnPipeline = sklearn_pipeline.Pipeline
SklearnTreeModel = sklearn_tree.DecisionTreeClassifier | sklearn_tree.DecisionTreeRegressor |\
    sklearn_ensemble.RandomForestClassifier | sklearn_ensemble.RandomForestRegressor

class ONNXConvertor:

    @staticmethod
    def find_model(input_pipeline: onnx.ModelProto) -> onnx.NodeProto | None:
        pass

    @staticmethod
    def from_model(input_model: onnx.NodeProto) -> TreeEnsembleRegressor:
        pass

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
        onnx.checker.check_node(node)
        return node

    @staticmethod
    def to_pipeline(input_pipeline: onnx.ModelProto, output_model: onnx.NodeProto) -> onnx.NodeProto:
        pass


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
