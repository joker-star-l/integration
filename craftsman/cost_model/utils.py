import craftsman.base.defs as defs
import subprocess
import time

def tree_paths(tree, node=0, path=None):
    """
    从决策树提取所有路径
    """
    if path is None:
        path = []
    path.append(node)
    if tree.children_left[node] == tree.children_right[node]:  # 叶子节点
        yield path
    else:
        yield from tree_paths(tree, tree.children_left[node], path.copy())
        yield from tree_paths(tree, tree.children_right[node], path.copy())


def calc_join_cost_by_train_data(data_rows, join_table_rows, join_table_columns):
    # pg
    if defs.DBMS == 'postgresql':
        # return data_rows * (9.08197566e-02)
        return data_rows * (4.43204396e-05 * join_table_rows + 9.08197566e-02 * join_table_columns)

    # duckdb
    elif defs.DBMS == 'duckdb':
        # return data_rows * ( 1.02417616e-05)
        return data_rows * (1.02122041e-08 * join_table_rows + 1.02417616e-05 * join_table_columns)

    # monetdb
    elif defs.DBMS == 'monetdb':
        # return data_rows * (1.67787246e-03)
        return data_rows * (4.58751378e-07 * join_table_rows + 1.67787246e-03 * join_table_columns)

    # TODO: choose the accurate join cost model


def get_pg_sql_cost(sql: str):
    # print(sql)
    try:
        output = subprocess.check_output(
            [
                "psql",
                "-U",
                "postgres",
                "-d",
                "postgres",
                "-c",
                "set max_parallel_workers_per_gather=3;" + sql,
            ],
            stderr=subprocess.STDOUT
        )
    except Exception as e:
        print(sql)
        exit
    
    execution_cost = None
    for line in output.decode().split("\n"):
        if "cost=" in line:
            execution_cost = float(line.split("..")[1].split()[0])
            break

    
    return execution_cost


def get_craftsman_graph_cost(graph, data_rows):
    cost = 0
    timer = False
    
    if timer:
        t1 = time.time()
    
    # op cost
    for feature, chain in graph.chains.items():
        # if feature == 'nom_5':
        #     pass
        implements = graph.implements[feature]
        for idx, op in enumerate(chain.prep_operators):
            # case op cost
            if implements[idx] == defs.SQLPlanType.CASE:
                for feature_out in op.features_out:
                    cost += op._get_op_cost(feature_out)
                    
            # join op cost
            elif implements[idx] == defs.SQLPlanType.JOIN:
                for feature in op.features:
                    cost += op._get_join_cost_without_tree(feature, graph, data_rows)
    if timer:
        t2 = time.time()
        print(f'op cost time: {t2 - t1}s')
        
    if timer:
        t1 = time.time()
    
    # tree cost
    # for feature in graph.model.input_features:
    #     if graph.model.model_name in (
    #         defs.ModelName.DECISIONTREECLASSIFIER,
    #         defs.ModelName.RANDOMFORESTCLASSIFIER,
    #         defs.ModelName.DECISIONTREEREGRESSOR,
    #         defs.ModelName.RANDOMFORESTREGRESSOR,
    #     ):
    #         # if 'nom_5' in feature:
    #         #     pass
    #         tree_costs = graph.model.get_tree_costs_static(feature)
    #         total_tree_cost = sum(
    #             [tree_cost.calculate_tree_cost() for tree_cost in tree_costs]
    #         )
    #         cost += total_tree_cost
    
    if graph.model.model_name in (
        defs.ModelName.DECISIONTREECLASSIFIER,
        defs.ModelName.RANDOMFORESTCLASSIFIER,
        defs.ModelName.DECISIONTREEREGRESSOR,
        defs.ModelName.RANDOMFORESTREGRESSOR,
    ):

        tree_costs = graph.model.get_tree_costs_static()
        total_tree_cost = sum(
            [tree_cost.calculate_tree_cost() for tree_cost in tree_costs]
        )
        cost += total_tree_cost
   
    if timer:
        t2 = time.time()
        print(f'tree cost time: {t2 - t1}s')
    
    # if timer:
    #     t1 = time.time()
        
    # # join op cost
    # for op in graph.join_operators:
    #     for feature in op.features:
    #         cost += op._get_join_cost_without_tree(feature, graph, data_rows)
    
    # if timer:
    #     t2 = time.time()
    #     print(f'join op cost time: {t2 - t1}s')    
    
    return cost
        
    
