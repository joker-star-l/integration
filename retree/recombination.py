# sliding-based subtree recombination
# 两侧必须有紧挨着超平面的纯色区域才可以滑动
# 与当前节点特征不同的节点两侧都要遍历
# 与当前节点特征相同的节点只需要遍历靠近当前节点的那一侧
# 删除最靠近当前节点的同特征节点
from typing import List, Tuple
from multiprocessing import Pool
from loguru import logger
from .tree import Node, TreeEnsembleRegressor, DecisionTreeRegressor

def recombine_single_tree(input_model: DecisionTreeRegressor) -> DecisionTreeRegressor:
    root = input_model.to_tree()
    dfs(root)
    return root.to_model(input_model.nodes_treeids[0])

_S_FALSE = 0
_S_TRUE = 1
_S_NO = 2

def find_silde_nodes(
    node: Node,                             # 当前节点
    parent: Node,                           # 当前节点的父节点
    path_length: int,                       # 从 root 到当前节点的长度
    root: Node,                             # 待确定的 slide node
    left_branch: bool,                      # node 是否在 root 的左侧
    result: List[Tuple[Node, Node, int]],   # boundary node 的相关信息（node、parent、path_length），列表长度恒为 1
    paths_length: List[int]                 # 从 root 到所有叶子节点的长度列表
) -> int:
    if node.mode == 'LEAF':
        paths_length.append(path_length)
        return _S_FALSE if node.target_weight == 0 else _S_TRUE
    
    same_feature = node.feature_id == root.feature_id

    # 特征不相同或特征相同但是 node 在 root 的右侧时, 需要递归遍历左子树
    if not same_feature or (same_feature and not left_branch):
        left_slide_stats = find_silde_nodes(node.left, node, path_length + 1, root, 
                                            left_branch, result, paths_length)
        if left_slide_stats == _S_NO:
            return _S_NO

    # 特征不相同或特征相同但是 node 在 root 的左侧时, 需要递归遍历右子树
    if not same_feature or (same_feature and left_branch):
        right_slide_stats = find_silde_nodes(node.right, node, path_length + 1, root, 
                                             left_branch, result, paths_length)
        if right_slide_stats == _S_NO:
            return _S_NO
    
    if not same_feature:
        # 特征不同, 左右两侧的状态不同, 无法合并
        return _S_NO if left_slide_stats != right_slide_stats else left_slide_stats
    
    if same_feature and not left_branch:
        if left_slide_stats != _S_NO:
            set_result(node, parent, path_length, root, result)
        return left_slide_stats
    
    if same_feature and left_branch:
        if right_slide_stats != _S_NO:
            set_result(node, parent, path_length, root, result)
        return right_slide_stats

def set_result(
    node: Node,
    parent: Node,
    path_length: int,
    root: Node,
    result: List[Tuple[Node, Node, int]]
):
    if node.mode == 'LEAF':
        return
    
    assert node.feature_id == root.feature_id

    if not result:
        result.append((node, parent, path_length))
        return

def dfs(node: Node):
    if node.mode == 'LEAF':
        return
    dfs(node.left)
    dfs(node.right)

    left_boundary_node = []
    left_paths_length = []
    left_slide_stats = find_silde_nodes(node.left, node, 1, node, True, 
                                        left_boundary_node, left_paths_length)
    if left_slide_stats == _S_NO:
        logger.debug(f'{node.id} cannot slide because of (left, {left_slide_stats})')
        return
    
    right_boundary_node = []
    right_paths_length = []
    right_slide_stats = find_silde_nodes(node.right, node, 1, node, False, 
                                         right_boundary_node, right_paths_length)
    if right_slide_stats == _S_NO:
        logger.debug(f'{node.id} cannot slide because of (right, {right_slide_stats})')
        return
    
    if left_slide_stats != right_slide_stats:
        logger.debug(f'{node.id} cannot slide because of (left, {left_slide_stats}) and (right, {right_slide_stats})')
        return
    
    logger.debug(f'{node.id} can slide because of (left, {left_slide_stats}) and (right, {right_slide_stats})')

    # 左侧分支长度的最小值 - 右侧分支长度的最大值
    left2right = min(left_paths_length) - max(right_paths_length)
    # 右侧分支长度的最大值 - 左侧分支长度的最小值
    right2left = min(right_paths_length) - max(left_paths_length)

    if left_boundary_node and right_boundary_node:
        logger.debug(f'{node.id} can slide to both sides')
        if left2right >= 0:
            logger.debug(f'{node.id} slide to left is definitely benificial')
        if right2left >= 0:
            logger.debug(f'{node.id} slide to right is definitely benificial')
        
        if left2right >= 0 and left2right >= right2left:
            logger.debug(f'{node.id} slide to left')
            slide(node, left_boundary_node, True)
            return
        
        if right2left >= 0 and right2left >= left2right:
            logger.debug(f'{node.id} slide to right')
            slide(node, right_boundary_node, False)
            return
        
        logger.debug(f'{node.id} does not slide')
        return
    
    if left_boundary_node:
        logger.debug(f'{node.id} slide to left')
        slide(node, left_boundary_node, True)
        return
    
    if right_boundary_node:
        logger.debug(f'{node.id} slide to right')
        slide(node, right_boundary_node, False)
        return

def slide(
    slide_node: Node,
    boundary_node_info: List[Tuple[Node, Node, int]],
    left_branch: bool
):
    boundary_node, boundary_parent, _ = boundary_node_info[0]
    assert slide_node.mode == 'BRANCH_LEQ'
    assert boundary_node.mode == 'BRANCH_LEQ'
    assert slide_node.feature_id == boundary_node.feature_id

    # change slide_node properties
    slide_node.value = boundary_node.value

    # change links
    if left_branch:
        boundary_left = boundary_node.left
        #link
        if boundary_node == boundary_parent.left:
            boundary_parent.left = boundary_left
        else:
            boundary_parent.right = boundary_left
        # TODO: update node.samples
    else:
        boundary_right = boundary_node.right
        # link
        if boundary_node == boundary_parent.left:
            boundary_parent.left = boundary_right
        else:
            boundary_parent.right = boundary_right
        # TODO: update node.samples

def recombine(input_model: TreeEnsembleRegressor, threads_count: int):
    threads_count = min(threads_count, len(input_model.regressors))
    with Pool(threads_count) as pool:
        input_model.regressors = pool.map(recombine_single_tree, input_model.regressors)
