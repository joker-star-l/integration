# merging-based subtree collapse
from typing import List, Callable, Any
from .tree import TreeEnsembleRegressor, DecisionTreeRegressor

def collapse():
    pass

def collpase_single_tree(
    tree_no: int,                        # 树 id
    node_id: int,                        # 节点 id
    result_nodes: List[str],             # 节点状态，包括 BRANCH_LEQ、LEAF_FALSE、LEAF_TRUE、REMOVED
    input_model: DecisionTreeRegressor,  # 输入模型
    
) -> int:   # 0: leaf_false, 1: leaf_true, 2: inner
    left_nodes = input_model.nodes_truenodeids
    right_nodes = input_model.nodes_falsenodeids
    node_types = input_model.nodes_modes

    target_treeids = input_model.target_treeids
    target_nodeids = input_model.target_nodeids
    target_weights = input_model.target_weights

    result_nodes[node_id] = node_types[node_id]
    is_leaf = node_types[node_id] == 'LEAF'

    if is_leaf:
        target_idx = -1
