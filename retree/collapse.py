# merging-based subtree collapse
from typing import List, Dict
from multiprocessing import Pool
from loguru import logger
from .tree import TreeEnsembleRegressor, DecisionTreeRegressor

def collapse_single_tree_internal(
    node_id: int,                        # 节点 id
    result_nodes: List[str],             # 节点状态，包括 BRANCH_LEQ、LEAF_FALSE、LEAF_TRUE、REMOVED
    input_model: DecisionTreeRegressor,  # 输入模型
    node_to_target: Dict[int, int],      # target_nodeids 数组的值与下标的映射
) -> int:                                # 0: leaf_false, 1: leaf_true, 2: inner
    left_nodes = input_model.nodes_truenodeids
    right_nodes = input_model.nodes_falsenodeids
    node_types = input_model.nodes_modes
    target_weights = input_model.target_weights

    result_nodes[node_id] = node_types[node_id]
    is_leaf = node_types[node_id] == 'LEAF'

    if is_leaf:
        target_idx = node_to_target[node_id]
        result = int(target_weights[target_idx])
        result_nodes[node_id] = 'LEAF_TRUE' if result == 1 else 'LEAF_FALSE'
        logger.debug(f'node_id: {node_id}, is_leaf: {is_leaf}, result: {result}')
        return result

    if not is_leaf:
        left_node_id = left_nodes[node_id]
        left_result = collapse_single_tree_internal(left_node_id, result_nodes, input_model, node_to_target)
        right_node_id = right_nodes[node_id]
        right_result = collapse_single_tree_internal(right_node_id, result_nodes, input_model, node_to_target)

    if left_result == 0 and right_result == 0:
        result_nodes[node_id] = 'LEAF_FALSE'
        result_nodes[left_node_id] = 'REMOVED'
        result_nodes[right_node_id] = 'REMOVED'
        logger.debug(f'node_id: {node_id}, is_leaf: {is_leaf}, result: 0')
        return 0
    
    if left_result == 1 and right_result == 1:
        result_nodes[node_id] = 'LEAF_TRUE'
        result_nodes[left_node_id] = 'REMOVED'
        result_nodes[right_node_id] = 'REMOVED'
        logger.debug(f'node_id: {node_id}, is_leaf: {is_leaf}, result: 1')
        return 1
    
    logger.debug(f'node_id: {node_id}, is_leaf: {is_leaf}, result: 2')
    return 2

def collapse_single_tree(
    input_model: DecisionTreeRegressor,  # 输入模型
) -> DecisionTreeRegressor:
    result_nodes: List[str] = [''] * len(input_model.nodes_nodeids)  # 节点状态，包括 BRANCH_LEQ、LEAF_FALSE、LEAF_TRUE、REMOVED
    node_to_target: Dict[int, int] = {}
    for ti, ni in enumerate(input_model.target_nodeids):
        node_to_target[ni] = ti
    collapse_single_tree_internal(0, result_nodes, input_model, node_to_target)

    leaf_count = result_nodes.count('LEAF_FALSE') + result_nodes.count('LEAF_TRUE')

    new_ids: List[int] = []
    curr_id = 0
    for result in result_nodes:
        if (result == 'LEAF_FALSE' or result == 'LEAF_TRUE'):
            new_ids.append(curr_id)
            curr_id += 1
        elif result == 'BRANCH_LEQ':
            new_ids.append(curr_id)
            curr_id += 1
        else:
            new_ids.append(-1)
    
    nodes_falsenodeids = [
        new_ids[ii] if new_ids[ii] != -1 else 0
        for i, ii in enumerate(input_model.nodes_falsenodeids)
        if new_ids[i] != -1
    ]
    nodes_featureids = [
        ii if result_nodes[i] == 'BRANCH_LEQ' else 0
        for i, ii in enumerate(input_model.nodes_featureids)
        if new_ids[i] != -1
    ]
    nodes_hitrates = [
        ii
        for i, ii in enumerate(input_model.nodes_hitrates)
        if new_ids[i] != -1
    ]
    nodes_missing_value_tracks_true = [
        ii
        for i, ii in enumerate(input_model.nodes_missing_value_tracks_true)
        if new_ids[i] != -1
    ]
    nodes_modes = [
        'BRANCH_LEQ' if result_nodes[i] == 'BRANCH_LEQ' else 'LEAF'
        for i, _ in enumerate(input_model.nodes_modes)
        if new_ids[i] != -1
    ]
    nodes_nodeids = [
        new_ids[i]
        for i, _ in enumerate(input_model.nodes_nodeids)
        if new_ids[i] != -1
    ]
    nodes_treeids = [
        ii
        for i, ii in enumerate(input_model.nodes_treeids)
        if new_ids[i] != -1
    ]
    nodes_truenodeids = [
        new_ids[ii] if new_ids[ii] != -1 else 0
        for i, ii in enumerate(input_model.nodes_truenodeids)
        if new_ids[i] != -1
    ]
    nodes_values = [
        ii if result_nodes[i] == 'BRANCH_LEQ' else 0
        for i, ii in enumerate(input_model.nodes_values)
        if new_ids[i] != -1
    ]
    target_ids = [input_model.target_ids[0]] * leaf_count
    target_nodeids = [
        ii
        for i, ii in enumerate(new_ids)
        if (result_nodes[i] == 'LEAF_FALSE' or result_nodes[i] == 'LEAF_TRUE')
    ]
    target_treeids = [input_model.target_treeids[0]] * leaf_count
    target_weights = [
        float(result_nodes[i] == 'LEAF_TRUE')
        for i, _ in enumerate(new_ids)
        if (result_nodes[i] == 'LEAF_FALSE' or result_nodes[i] == 'LEAF_TRUE')
    ]

    input_model.nodes_falsenodeids = nodes_falsenodeids
    input_model.nodes_featureids = nodes_featureids
    input_model.nodes_hitrates = nodes_hitrates
    input_model.nodes_missing_value_tracks_true = nodes_missing_value_tracks_true
    input_model.nodes_modes = nodes_modes
    input_model.nodes_nodeids = nodes_nodeids
    input_model.nodes_treeids = nodes_treeids
    input_model.nodes_truenodeids = nodes_truenodeids
    input_model.nodes_values = nodes_values
    input_model.target_ids = target_ids
    input_model.target_nodeids = target_nodeids
    input_model.target_treeids = target_treeids
    input_model.target_weights = target_weights
    return input_model

def collapse(input_model: TreeEnsembleRegressor, threads_count: int):
    threads_count = min(threads_count, len(input_model.regressors))
    with Pool(threads_count) as pool:
        input_model.regressors = pool.map(collapse_single_tree, input_model.regressors)
