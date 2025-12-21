import pandas as pd

from craftsman.base.graph import PrepGraph
from craftsman.base.defs import *
import craftsman.base.defs as defs
from craftsman.model.base_model import TreeModel
from craftsman.preprocess.k_bins_discretizer import KBinsDiscretizerSQLOperator


def merge_by_cost_model(graph: PrepGraph, data_rows, merge_flag, cost_flag, masq):
    new_prep_graph = graph.get_empty_chains_graph()

    if cost_flag:
        cost_results = []

    for feature, chain in graph.chains.items():
        if chain.prep_operators:
            if len(chain.prep_operators) == 1:
                op = chain.prep_operators[0]

                # --------------------!!!!!!!!!!!DEBUG CODE!!!!!!!!!!------------------------------
                # if 'all' == defs.EXPRIMENT_COL or feature == defs.EXPRIMENT_COL:
                #     if defs.EXPRIMENT_METHOD == SQLPlanType.CASE:
                #         new_prep_graph.chains[feature] = chain
                #     elif defs.EXPRIMENT_METHOD == SQLPlanType.FUSION:
                #         op.fusion(new_prep_graph)
                #     elif defs.EXPRIMENT_METHOD == SQLPlanType.PUSH:
                #         op.push(new_prep_graph)
                #     elif defs.EXPRIMENT_METHOD == SQLPlanType.JOIN:
                #         if hasattr(op, 'join'):
                #             op.join(new_prep_graph)
                #         else:
                #            new_prep_graph.chains[feature] = chain 
                #     continue
                # --------------------!!!!!!!!!!!DEBUG CODE!!!!!!!!!!------------------------------

                # masq method: push all operator, but fusion the onehot encoder
                if masq and isinstance(graph.model, TreeModel):
                    if op.op_name == OperatorName.ONEHOTENCODER:
                        op.fusion(new_prep_graph)
                    else:
                        op.push(new_prep_graph)

                # con-a-con is always can fusion by the rule
                elif not masq and merge_flag and isinstance(graph.model, TreeModel) and (
                    op.op_type == OperatorType.CON_A_CON
                    or op.op_name == OperatorName.ONEHOTENCODER
                    or isinstance(op, KBinsDiscretizerSQLOperator)
                ):
                    op.fusion(new_prep_graph)
                else:
                    # common case: (cat-c-cat | expand | con-c-cat) + Tree
                    if cost_flag:
                        # if use cost model
                        sql_plan, costs_line = op.get_best_plan(graph, data_rows)
                        cost_results.append(costs_line)
                        if sql_plan == SQLPlanType.FUSION:
                            op.fusion(new_prep_graph)
                        elif sql_plan == SQLPlanType.JOIN:
                            op.join(new_prep_graph)
                        elif sql_plan == SQLPlanType.PUSH:
                            op.push(new_prep_graph)
                        elif sql_plan == SQLPlanType.CASE:
                            new_prep_graph.chains[feature] = chain
                    else:
                        new_prep_graph.chains[feature] = chain

            # special case: con-c-cat + expand + tree
            elif len(chain.prep_operators) >= 2:
                new_prep_graph.chains[feature] = chain

    # save costs to csv file
    # if cost_flag:
    #     pd.DataFrame(cost_results, columns=["op_name", "feature", "case", "fusion", "join", "push"]).to_csv(
    #         f"{defs.COST_DIR_PATH}/{defs.CURRENT_PYTHON_FILE}_{'ordered' if defs.ORDER_WHEN else 'unordered'}.csv",
    #         index=False
    #     )

    return new_prep_graph
