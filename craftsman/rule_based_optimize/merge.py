import copy
import time
from pandas import DataFrame

from craftsman.model.base_model import TreeModel
from craftsman.base.graph import PrepGraph, PrepChain
from craftsman.base.plan import ChainImplementPlan, ChainFusionPlan
from craftsman.base.operator import *
from craftsman.rule_based_optimize.merge_property import PropertyManager
import craftsman.base.defs as defs


# def merge_sql_operator_by_rules(preprocessing_graph: PrepGraph) -> PrepGraph:
#     new_prep_graph = preprocessing_graph.get_empty_chains_graph()

#     for feature, chain in preprocessing_graph.chains.items():
#         if chain.prep_operators:
#             first_op = chain.prep_operators[0]
#             op_idx = 1
#             while op_idx < len(chain.prep_operators):
#                 second_op = chain.prep_operators[op_idx]
#                 op_idx += 1
#                 merged_op = _merge(first_op, second_op)
#                 if merged_op is None:
#                     new_prep_graph.chains[feature].prep_operators.append(first_op)
#                     first_op = second_op
#                 else:
#                     first_op = merged_op

#             if (
#                 not new_prep_graph.chains[feature].prep_operators
#                 or first_op != new_prep_graph.chains[feature].prep_operators[-1]
#             ):
#                 new_prep_graph.chains[feature].prep_operators.append(first_op)

#     return new_prep_graph


# def _merge(first_op: Operator, second_op: Operator):
#     rule_table_content = [
#         ["simply", "simply", "disable", "disable"],
#         ["apply", "apply", "apply", "simply"],#"disable"
#         ["apply", "apply", "apply", "simply"],
#         ["apply", "apply", "apply", "disable"],
#     ]

#     rule_table_index_columns = [
#         OperatorType.CON_A_CON.value,
#         OperatorType.CON_C_CAT.value,
#         OperatorType.CAT_C_CAT.value,
#         OperatorType.EXPAND.value,
#     ]

#     rule_table = DataFrame(
#         rule_table_content,
#         index=rule_table_index_columns,
#         columns=rule_table_index_columns,
#     )

#     choice = rule_table.loc[first_op.op_type.value, second_op.op_type.value]

#     if choice == "apply":
#         return second_op.apply(first_op)
#     elif choice == "simply":
#         return first_op.simply(second_op)
#     elif choice == "disable":
#         return None

def _merge_by_implement_method(first_op, second_op, first_implementaion, second_implementation, assigned_rule=None):
    if not defs.AUTO_RULE_GEN:
        rule_table = defs.rule_table
        implementation_table = defs.implementation_table
        if assigned_rule != None:
            rule_table = rule_table.applymap(lambda x: "disable")
            for rule in assigned_rule:
                rule_table.loc[rule[0], rule[1]] = defs.rule_table.loc[rule[0], rule[1]]

        if second_implementation != 'Tree' and second_implementation != 'Not-Tree':
            if defs.MASQ:
                return [[[first_op, second_op], [first_implementaion, second_implementation]]]
            else:
                choice = rule_table.loc[first_op.op_type.value + first_implementaion.value, second_op.op_type.value + second_implementation.value]
                merged_implementation = implementation_table.loc[first_op.op_type.value + first_implementaion.value, second_op.op_type.value + second_implementation.value]
                if choice == "apply":
                    return [[[second_op.apply(first_op)], [merged_implementation]]]
                elif choice == "simply":
                    return [[[first_op.simply(second_op)], [merged_implementation]]]
                elif choice == "disable":
                    return [[[first_op, second_op], [first_implementaion, second_implementation]]]
                elif choice == "uncertain":
                    merged_op = second_op.apply(first_op)
                    if merged_op is None:
                        merged_op = first_op.simply(second_op)
                    # special case, directly merge
                    if ((first_op.op_name == OperatorName.STANDARDSCALER and second_op.op_name == OperatorName.MINMAXSCALER) 
                        or (first_op.op_name == OperatorName.MINMAXSCALER and second_op.op_name == OperatorName.STANDARDSCALER)
                        or (first_op.op_name == OperatorName.STANDARDSCALER and second_op.op_name == OperatorName.KBINSDISCRETIZER)
                        or (first_op.op_name == OperatorName.MINMAXSCALER and second_op.op_name == OperatorName.KBINSDISCRETIZER)):
                        return [[[merged_op], [merged_implementation]]]
                    else:
                        return [[[merged_op], [merged_implementation]],
                                [[first_op, second_op], [first_implementaion, second_implementation]]]
                    
        elif second_implementation == 'Tree':
            if defs.MASQ:
                if first_op.op_name == OperatorName.ONEHOTENCODER:
                    
                    copyed_model = copy.deepcopy(second_op)
                    first_op.fusion(copyed_model)
                    
                    return[[[copyed_model], [second_implementation]]]
                else:
                    return [[[first_op, second_op], [first_implementaion, second_implementation]]]
                
            else:
            
                choice = rule_table.loc[first_op.op_type.value + first_implementaion.value, 'Tree']
                if choice == "disable":
                    return [[[first_op, second_op], [first_implementaion, second_implementation]]]
                elif choice == "uncertain":
                    
                    copyed_model = copy.deepcopy(second_op)
                    first_op.fusion(copyed_model)
                    
                    # special case, directly fusion
                    if first_op.op_name in (OperatorName.KBINSDISCRETIZER, OperatorName.STANDARDSCALER, OperatorName.MINMAXSCALER):
                        return[[[copyed_model], [second_implementation]]]
                    else:
                        return[[[copyed_model], [second_implementation]],
                            [[first_op, second_op], [first_implementaion, second_implementation]]]
            
        elif second_implementation == 'Not-Tree':
            return [[[first_op, second_op], [first_implementaion, second_implementation]]]
    
    else:
        pm = PropertyManager()
        properties = ['property1', 'property3', 'property2']
        # properties = [method for method in dir(pm) if callable(getattr(pm, method)) and method.startswith('property')]
        for property in properties:
            merged_op = getattr(pm, property)(first_op, second_op)
            if merged_op:
                break

        if second_implementation != 'Tree' and second_implementation != 'Not-Tree':
            if defs.MASQ:
                return [[[first_op, second_op], [first_implementaion, second_implementation]]]
            else:
                if merged_op:
                    if property != 'property2':
                        return [[[merged_op], [SQLPlanType.CASE]]]
                    else:
                        # special case, directly merge
                        if ((first_op.op_name == OperatorName.STANDARDSCALER and second_op.op_name == OperatorName.MINMAXSCALER) 
                            or (first_op.op_name == OperatorName.MINMAXSCALER and second_op.op_name == OperatorName.STANDARDSCALER)
                            or (first_op.op_name == OperatorName.STANDARDSCALER and second_op.op_name == OperatorName.KBINSDISCRETIZER)
                            or (first_op.op_name == OperatorName.MINMAXSCALER and second_op.op_name == OperatorName.KBINSDISCRETIZER)):
                            return [[[merged_op], [SQLPlanType.CASE]]]
                        else:
                            return [[[merged_op], [SQLPlanType.CASE]],
                                    [[first_op, second_op], [SQLPlanType.CASE, SQLPlanType.CASE]]]
                else:
                    return [[[first_op, second_op], [SQLPlanType.CASE, SQLPlanType.CASE]]]
                        
        elif second_implementation == 'Tree':
            if defs.MASQ:
                if first_op.op_name == OperatorName.ONEHOTENCODER:
                    return[[[merged_op], ['Tree']]]
                else:
                    return [[[first_op, second_op], [first_implementaion, second_implementation]]]
                
            else:
                if not merged_op:
                    return [[[first_op, second_op], [first_implementaion, second_implementation]]]
                else:
                    if property != 'property2':
                        return [[[merged_op], ['Tree']]]
                    else:
                        # special case, directly fusion
                        if first_op.op_name in (OperatorName.KBINSDISCRETIZER, OperatorName.STANDARDSCALER, OperatorName.MINMAXSCALER):
                            return[[[merged_op], ['Tree']]]
                        else:
                            return[[[merged_op], ['Tree']],
                                [[first_op, second_op], [first_implementaion, second_implementation]]]
            
        elif second_implementation == 'Not-Tree':
            return [[[first_op, second_op], [first_implementaion, second_implementation]]]

def merge_sql_operator_by_chain_plan(
    preprocessing_graph: PrepGraph,
    chain_implement_plan: ChainImplementPlan,
    chain_fusion_plan: ChainFusionPlan,
    feature: str,
    assigned_rule = None):
    # aim to maintain a list of graph, to contain all possible plan
    graph_list: list[PrepGraph] = []
    
    # original graph
    new_prep_graph = preprocessing_graph.copy_prune_graph(feature)
    new_prep_graph.chains[feature] = PrepChain(feature)
    new_prep_graph.implements[feature] = []
    graph_list.append(new_prep_graph)
    
    # consider the special chain
    chain = preprocessing_graph.chains[feature]
    
    
    first_index = chain_fusion_plan.begin_op_index
        
    # if the chain don't have any operators
    if first_index == -1:
        return graph_list
    
    
    second_index = first_index
    fusion_directions = chain_fusion_plan.fusion_directions    
    # next_direction = fusion_directions[0]

    # initial the chain for every graph
    for graph in graph_list:
        graph.chains[feature].prep_operators.append(chain.prep_operators[first_index])
        graph.implements[feature].append(chain_implement_plan.chain_implement_plan[first_index])
    
    # fusion ops for every chain   
    new_prep_graph = preprocessing_graph.copy_prune_graph(feature)
    step = 0
    while(step < len(fusion_directions)):
        next_direction = fusion_directions[step]
        new_graph_list = []
        if next_direction == 1:
            second_index = second_index + 1
        elif next_direction == 0:
            first_index = first_index - 1
            
        for graph in graph_list:
            if next_direction == 1:
                first_op = graph.chains[feature].prep_operators[-1]
                first_implementaion = graph.implements[feature][-1]
                if second_index == len(chain.prep_operators):
                    second_op = graph.model
                    if isinstance(second_op, TreeModel):
                        second_implementaion = 'Tree'
                    else:
                        second_implementaion = 'Not-Tree'
                else:
                    second_op = chain.prep_operators[second_index]
                    second_implementaion = chain_implement_plan.chain_implement_plan[second_index]
                    
            elif next_direction == 0:
                first_op = chain.prep_operators[first_index]
                first_implementaion = chain_implement_plan.chain_implement_plan[first_index]
                if len(graph.chains[feature].prep_operators) == 0:
                    second_op = graph.model
                    if isinstance(second_op, TreeModel):
                        second_implementaion = 'Tree'
                    else:
                        second_implementaion = 'Not-Tree'
                else:
                    second_op = graph.chains[feature].prep_operators[0]
                    second_implementaion = graph.implements[feature][0]

            merged_res = _merge_by_implement_method(first_op, second_op, first_implementaion, second_implementaion, assigned_rule)
            
            res_graphs = [graph]
            if len(merged_res) == 2:
                twin_graph = graph.copy_prune_graph(feature)
                res_graphs.append(twin_graph)
            
            for situation, graph in zip(merged_res, res_graphs):
                if len(situation[0]) == 1:
                    if situation[1][0] == 'Tree' or situation[1][0] == 'Not-Tree':
                        if graph.chains[feature].prep_operators and first_op == graph.chains[feature].prep_operators[-1]:
                            graph.chains[feature].prep_operators.pop()
                            graph.implements[feature].pop()
                        graph.model = situation[0][0]
                    else:
                        if next_direction == 1:
                            graph.chains[feature].prep_operators[-1] = situation[0][0]
                            graph.implements[feature][-1] = situation[1][0]
                        else:
                            graph.chains[feature].prep_operators[0] = situation[0][0]
                            graph.implements[feature][0] = situation[1][0]
                            
                elif len(situation[0]) == 2:
                    if situation[1][1] == 'Tree' or situation[1][1] == 'Not-Tree':
                        if len(graph.chains[feature].prep_operators) == 0:
                            graph.chains[feature].prep_operators.append(situation[0][0])
                            graph.implements[feature].append(situation[1][0])
                    else:
                        if next_direction == 1:
                            graph.chains[feature].prep_operators.append(situation[0][1])
                            graph.implements[feature].append(situation[1][1])
                        else:
                            graph.chains[feature].prep_operators.insert(0, situation[0][0])
                            graph.implements[feature].insert(0, situation[1][0])
                new_graph_list.append(graph)   
        
        step = step + 1            
        graph_list = new_graph_list 
    # join the op to model if the implement is join
    # for graph in graph_list:
    #     chain = graph.chains[feature]
    #     new_prep_operators = []
    #     for i, op in enumerate(chain.prep_operators):
    #         if graph.implements[feature][i] == SQLPlanType.JOIN:
    #             op.join(graph)
    #         else:
    #             new_prep_operators.append(op)
    #     chain.prep_operators = new_prep_operators
                
    return graph_list


def merge_sql_operator_by_graph_plan(
    preprocessing_graph: PrepGraph,
    graph_implement_plan: list[ChainImplementPlan],
    graph_fusion_plan: list[ChainFusionPlan],
):
    # aim to maintain a list of graph, to contain all possible plan
    graph_list: list[PrepGraph] = []
    
    # original graph
    new_prep_graph = preprocessing_graph.get_empty_chains_graph()
    graph_list.append(new_prep_graph)
    
    
    
    # consider the iteration, chains by chains
    for index, item in enumerate(preprocessing_graph.chains.items()):
        feature, chain = item
        chain_implement_plan = graph_implement_plan[index]
        chain_fusion_plan = graph_fusion_plan[index]
        first_index = chain_fusion_plan.begin_op_index
        
        # if the chain don't have any operators
        if first_index == -1:
            continue
        
        
        second_index = first_index
        fusion_directions = chain_fusion_plan.fusion_directions    
        # next_direction = fusion_directions[0]
    
        # initial the chain for every graph
        for graph in graph_list:
            graph.chains[feature].prep_operators.append(chain.prep_operators[first_index])
            graph.implements[feature].append(chain_implement_plan.chain_implement_plan[first_index])
        
        # fusion ops for every chain   
        step = 0
        while(step < len(fusion_directions)):
            next_direction = fusion_directions[step]
            new_graph_list = []
            if next_direction == 1:
                second_index = second_index + 1
            elif next_direction == 0:
                first_index = first_index - 1
                
            for graph in graph_list:
                if next_direction == 1:
                    first_op = graph.chains[feature].prep_operators[-1]
                    first_implementaion = graph.implements[feature][-1]
                    if second_index == len(chain.prep_operators):
                        second_op = graph.model
                        if isinstance(second_op, TreeModel):
                            second_implementaion = 'Tree'
                        else:
                            second_implementaion = 'Not-Tree'
                    else:
                        second_op = chain.prep_operators[second_index]
                        second_implementaion = chain_implement_plan.chain_implement_plan[second_index]
                        
                elif next_direction == 0:
                    first_op = chain.prep_operators[first_index]
                    first_implementaion = chain_implement_plan.chain_implement_plan[first_index]
                    if len(graph.chains[feature].prep_operators) == 0:
                        second_op = graph.model
                        if isinstance(second_op, TreeModel):
                            second_implementaion = 'Tree'
                        else:
                            second_implementaion = 'Not-Tree'
                    else:
                        second_op = graph.chains[feature].prep_operators[0]
                        second_implementaion = graph.implements[feature][0]

                merged_res = _merge_by_implement_method(first_op, second_op, first_implementaion, second_implementaion)
                
                res_graphs = [graph]
                if len(merged_res) == 2:
                    # twin_graph = graph.copy_graph()
                    twin_graph = graph.copy_prune_graph(feature)
                    res_graphs.append(twin_graph)
                
                for situation, graph in zip(merged_res, res_graphs):
                    if len(situation[0]) == 1:
                        if situation[1][0] == 'Tree' or situation[1][0] == 'Not-Tree':
                            if graph.chains[feature].prep_operators and first_op == graph.chains[feature].prep_operators[-1]:
                                graph.chains[feature].prep_operators.pop()
                                graph.implements[feature].pop()
                            graph.model = situation[0][0]
                        else:
                            if next_direction == 1:
                                graph.chains[feature].prep_operators[-1] = situation[0][0]
                                graph.implements[feature][-1] = situation[1][0]
                            else:
                                graph.chains[feature].prep_operators[0] = situation[0][0]
                                graph.implements[feature][0] = situation[1][0]
                                
                    elif len(situation[0]) == 2:
                        if situation[1][1] == 'Tree' or situation[1][1] == 'Not-Tree':
                            if len(graph.chains[feature].prep_operators) == 0:
                                graph.chains[feature].prep_operators.append(situation[0][0])
                                graph.implements[feature].append(situation[1][0])
                        else:
                            if next_direction == 1:
                                graph.chains[feature].prep_operators.append(situation[0][1])
                                graph.implements[feature].append(situation[1][1])
                            else:
                                graph.chains[feature].prep_operators.insert(0, situation[0][0])
                                graph.implements[feature].insert(0, situation[1][0])
                    new_graph_list.append(graph)   
            
            step = step + 1            
            graph_list = new_graph_list
    
    # join the op to model if the implement is join
    # for graph in graph_list:
    #     for feature, chain in graph.chains.items():
    #         new_prep_operators = []
    #         for i, op in enumerate(chain.prep_operators):
    #             if graph.implements[feature][i] == SQLPlanType.JOIN:
    #                 op.join(graph)
    #             else:
    #                 new_prep_operators.append(op)
    #         chain.prep_operators = new_prep_operators
                
    return graph_list


def join_the_operators(preprocessing_graph: PrepGraph):
    new_prep_graph = preprocessing_graph.get_empty_chains_graph()    
    for feature, chain in preprocessing_graph.chains.items():
        for i, op in enumerate(chain.prep_operators):
            if preprocessing_graph.implements[feature][i] == SQLPlanType.JOIN:
                op.join(new_prep_graph)
            else:
                new_prep_graph.chains[feature].prep_operators.append(op)
                new_prep_graph.implements[feature].append(preprocessing_graph.implements[feature][i])
    return new_prep_graph
            
            
def implement_operator_by_plan(preprocessing_graph: PrepGraph, graph_implement_plan):
    new_prep_graph = preprocessing_graph.get_empty_chains_graph()
    for index, item in enumerate(preprocessing_graph.chains.items()):
        feature, chain = item
        for op, implement in zip(chain.prep_operators, graph_implement_plan[index].chain_implement_plan):
            # if implement == SQLPlanType.JOIN:
            #     op.join(new_prep_graph)
            new_prep_graph.chains[feature].prep_operators.append(op)
            new_prep_graph.implements[feature].append(implement)
    return new_prep_graph

def merge_sql_operator_by_benifit_rules(graph: PrepGraph):
    new_prep_graph = graph.get_empty_chains_graph()
    for index, item in enumerate(graph.chains.items()):
        feature, chain = item
        implements = graph.implements[feature]
        if chain.prep_operators:
            first_op = chain.prep_operators[0]
            first_implement = implements[0]
            op_idx = 1
            while op_idx < len(chain.prep_operators):
                second_op = chain.prep_operators[op_idx]
                second_implement = implements[op_idx]
                op_idx += 1
                merged_res = _merge_by_implement_method(first_op, second_op, first_implement, second_implement)
                if len(merged_res) == 2 or len(merged_res[0][0]) == 2:
                    new_prep_graph.chains[feature].prep_operators.append(first_op)
                    new_prep_graph.implements[feature].append(first_implement)
                    first_op = second_op
                    first_implement = second_implement
                else:
                    first_op = merged_res[0][0][0]
                    first_implement = merged_res[0][1][0]

            if (
                not new_prep_graph.chains[feature].prep_operators
                or first_op != new_prep_graph.chains[feature].prep_operators[-1]
            ):
                new_prep_graph.chains[feature].prep_operators.append(first_op)
                new_prep_graph.implements[feature].append(first_implement)

    if isinstance(new_prep_graph.model, TreeModel):
        for index, (feature, chain) in enumerate(new_prep_graph.chains.items()):
            if chain.prep_operators:
                op = chain.prep_operators[-1]
                implement = new_prep_graph.implements[feature][-1]
                merged_res = _merge_by_implement_method(op, new_prep_graph.model, implement, 'Tree')
                if len(merged_res) == 1 and len(merged_res[0][0]) == 1:
                    new_prep_graph.model = merged_res[0][0][0]
                    new_prep_graph.implements[feature].pop()
                    chain.prep_operators.pop()
    return new_prep_graph

def merge_sql_operator_by_uncertain_rules(graph: PrepGraph):
    new_prep_graph = graph.get_empty_chains_graph()
    for index, item in enumerate(graph.chains.items()):
        feature, chain = item
        implements = graph.implements[feature]
        if chain.prep_operators:
            first_op = chain.prep_operators[0]
            first_implement = implements[0]
            op_idx = 1
            while op_idx < len(chain.prep_operators):
                second_op = chain.prep_operators[op_idx]
                second_implement = implements[op_idx]
                op_idx += 1
                merged_res = _merge_by_implement_method(first_op, second_op, first_implement, second_implement)
                if len(merged_res) == 2 or (len(merged_res) == 1 and len(merged_res[0][0]) == 1):
                    first_op = merged_res[0][0][0]
                    first_implement = merged_res[0][1][0]
                else:
                    new_prep_graph.chains[feature].prep_operators.append(first_op)
                    new_prep_graph.implements[feature].append(first_implement)
                    first_op = second_op
                    first_implement = second_implement

            if (
                not new_prep_graph.chains[feature].prep_operators
                or first_op != new_prep_graph.chains[feature].prep_operators[-1]
            ):
                new_prep_graph.chains[feature].prep_operators.append(first_op)
                new_prep_graph.implements[feature].append(first_implement)
    
    if isinstance(new_prep_graph.model, TreeModel):
        for index, (feature, chain) in enumerate(new_prep_graph.chains.items()):
            if chain.prep_operators:
                op = chain.prep_operators[-1]
                implement = new_prep_graph.implements[feature][-1]
                merged_res = _merge_by_implement_method(op, new_prep_graph.model, implement, 'Tree')
                if len(merged_res) == 2 or len(merged_res[0][0]) == 1:
                    new_prep_graph.model = merged_res[0][0][0]
                    new_prep_graph.implements[feature].pop()
                    chain.prep_operators.pop()
    
    return new_prep_graph