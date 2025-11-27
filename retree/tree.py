# internal data structures
from typing import List, Dict
from multiprocessing import Pool

class Node:
    def __init__(
            self,
            id: int,                        # 节点 id
            feature_id: int,                # 特征 id
            mode: str,                      # 节点类型，LEAF 表示叶子节点，BRANCH_LEQ 表示非叶子节点
            value: float,                   # 阈值，叶子节点的值为 0
            target_idx: int | None,         # target_nodeids 数组的下标
            target_weight: float | None,    # 叶子节点的权重，即预测值
            samples: int                    # 节点的样本数
            ):
        self.id: int = id
        self.feature_id: int = feature_id
        self.mode: str = mode
        self.value: float = value
        self.target_idx: int | None = target_idx
        self.target_weight: float | None = target_weight
        self.samples: int = samples

        self.left: Node | None = None
        self.right: Node | None  = None

    def to_model(self, tree_no: int) -> 'DecisionTreeRegressor':
        regressor = DecisionTreeRegressor()
        self.to_model_internal(tree_no, regressor)
        id_map = {old_id: new_id for new_id, old_id in enumerate(regressor.nodes_nodeids)}
        is_leaf = [mode == 'LEAF' for mode in regressor.nodes_modes]
        regressor.nodes_falsenodeids = [(0 if is_leaf[i] else id_map[old_id]) for i, old_id in enumerate(regressor.nodes_falsenodeids)]
        regressor.nodes_truenodeids = [(0 if is_leaf[i] else id_map[old_id]) for i, old_id in enumerate(regressor.nodes_truenodeids)]
        regressor.nodes_nodeids = [id_map[old_id] for old_id in regressor.nodes_nodeids]
        regressor.target_nodeids = [id_map[old_id] for old_id in regressor.target_nodeids]
        return regressor

    def to_model_internal(self, tree_no: int, regressor: 'DecisionTreeRegressor'):
        is_leaf = self.mode == 'LEAF'

        regressor.nodes_falsenodeids.append(self.right.id if not is_leaf else 0)
        regressor.nodes_featureids.append(self.feature_id)
        regressor.nodes_hitrates.append(float(self.samples))
        regressor.nodes_missing_value_tracks_true.append(0)
        regressor.nodes_modes.append(self.mode)
        regressor.nodes_nodeids.append(self.id)
        regressor.nodes_treeids.append(tree_no)
        regressor.nodes_truenodeids.append(self.left.id if not is_leaf else 0)
        regressor.nodes_values.append(self.value)
        
        if is_leaf:
            regressor.target_ids.append(0)
            regressor.target_nodeids.append(self.id)
            regressor.target_treeids.append(tree_no)
            regressor.target_weights.append(self.target_weight)

        if not is_leaf:
            self.left.to_model_internal(tree_no, regressor)
            self.right.to_model_internal(tree_no, regressor)


class TreeEnsembleRegressor:
    def __init__(self):
        self.regressors: List[DecisionTreeRegressor] = []       # 决策树

    def to_trees(self, thread_count: int) -> List[Node]:
        threads_count = min(threads_count, len(self.regressors))
        trees: List[Node] = []
        with Pool(thread_count) as pool:
            trees = pool.map(_to_tree, self.regressors)
        return trees


class DecisionTreeRegressor:
    def __init__(self):
        self.nodes_falsenodeids: List[int] = []                 # 右侧分支
        self.nodes_featureids: List[int] = []                   # 特征 id
        self.nodes_hitrates: List[float] = []                   # 样本数
        self.nodes_missing_value_tracks_true: List[int] = []    # 缺失值标记
        self.nodes_modes: List[str] = []                        # 节点类型，LEAF 表示叶子节点，BRANCH_LEQ 表示非叶子节点
        self.nodes_nodeids: List[int] = []                      # 节点 id
        self.nodes_treeids: List[int] = []                      # 树 id
        self.nodes_truenodeids: List[int] = []                  # 左侧分支
        self.nodes_values: List[float] = []                     # 阈值，叶子节点的值为 0
        self.target_ids: List[int] = []                         # 叶子节点的 target id
        self.target_nodeids: List[int] = []                     # 叶子节点的节点 id
        self.target_treeids: List[int] = []                     # 叶子节点的树 id
        self.target_weights: List[float] = []                   # 叶子节点的权重，即预测值（不除以树的数量）

    def to_tree(self) -> Node:
        node_to_target: Dict[int, int] = {}
        for ti, ni in enumerate(self.target_nodeids):
            node_to_target[ni] = ti
        return self.to_tree_internal(0, node_to_target)

    def to_tree_internal(
        self, 
        node_id: int, 
        node_to_target: Dict[int, int]
    ) -> Node:
        id = node_id
        feature_id = self.nodes_featureids[id]
        mode = self.nodes_modes[id]
        value = self.nodes_values[id]
        target_idx = node_to_target.get(id, None)
        target_weight = self.target_weights[target_idx] if target_idx is not None else None
        samples = int(self.nodes_hitrates[id])

        node = Node(
            id=id,
            feature_id=feature_id,
            mode=mode,
            value=value,
            target_idx=target_idx,
            target_weight=target_weight,
            samples=samples
        )

        if mode != 'LEAF':
            left_node_id = self.nodes_truenodeids[id]
            node.left = self.to_tree_internal(left_node_id, node_to_target)
            right_node_id = self.nodes_falsenodeids[id]
            node.right = self.to_tree_internal(right_node_id, node_to_target)

        return node

def _to_tree(r: DecisionTreeRegressor) -> Node:
    return r.to_tree()
