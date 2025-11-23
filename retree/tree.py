# internal data structures
from typing import List

class Node:
    def __init__(
            self,
            id: int,                        # 节点 id
            feature_id: int,                # 特征 id
            mode: str,                      # 节点类型，LEAF 表示叶子节点，BRANCH_LEQ 表示非叶子节点
            value: float,                   # 阈值，叶子节点的值为 0
            target_id: int | None,          # 叶子节点的 taget id
            target_weight: float | None,    # 叶子节点的权重，即预测值
            samples: int | None             # 节点的样本数
            ):
        self.id: int = id
        self.feature_id: int = feature_id
        self.mode: str = mode
        self.value: float = value
        self.target_id: int | None = target_id
        self.target_weight: float | None = target_weight
        self.samples: int = samples

        self.parent: Node | None = None
        self.left: Node | None = None
        self.right: Node | None  = None


class TreeEnsembleRegressor:
    def __init__(self):
        self.regressors: List[DecisionTreeRegressor]            # 决策树


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
