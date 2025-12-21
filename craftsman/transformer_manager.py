import time
import copy
import pickle
import os
from concurrent.futures import ProcessPoolExecutor

from craftsman.base.graph import PrepGraph
from craftsman.base.plan import ChainCandidateFusionPlans, ChainCandidateImplementPlans
from craftsman.rule_based_optimize.merge import *
from craftsman.cost_model.merge import merge_by_cost_model
from craftsman.cost_model.utils import get_pg_sql_cost, get_craftsman_graph_cost
from craftsman.utility.loader import load_model
from craftsman.utility.dbms_utils import DBMSUtils
import craftsman.base.defs as defs


def task(
    all_implements_plans,
    all_fusion_plans,
    preprocessing_graph,
    cost_model,
    data_rows,
    table_name,
    dbms,
    pre_sql,
    pipeline,
):
    min_cost = float('inf')
    plan_num = 0
    min_cost_preprocessing_graph = None
    for graph_fusion_plan in all_fusion_plans:
        for graph_implement_plan in all_implements_plans:
            preprocessing_graph_list = merge_sql_operator_by_graph_plan(preprocessing_graph, graph_implement_plan, graph_fusion_plan)
            plan_num = plan_num + len(preprocessing_graph_list)
            for graph in preprocessing_graph_list:
                if cost_model == 'craftsman':
                    cost = get_craftsman_graph_cost(graph, data_rows)
                elif cost_model == 'postgresql':
                    mng = TransformerManager()
                    query_str = mng.__compose_sql(graph, table_name, dbms, pre_sql, pipeline)
                    cost = get_pg_sql_cost(query_str)
                if cost < min_cost:
                    min_cost_preprocessing_graph = graph
                    min_cost = cost
    return min_cost_preprocessing_graph, min_cost, plan_num


class TransformerManager(object):

    def __extract_pipeline(self, pipeline):
        steps = pipeline.steps
        column_transformer_start_idx = 0

        fitted_imputer = None
        # if contain imputer extract it
        if 'Imputer' == steps[0][0]:
            fitted_imputer = steps[0][1]
            column_transformer_start_idx = 1

        # extract model
        model_name, trained_model = steps[-1]

        # extract preprocessors
        transforms = []
        for i in range(column_transformer_start_idx, len(steps) - 1):
            _, pipeline_transformers = steps[i]
            for idx in range(len(pipeline_transformers.transformers)):
                a, b, c = pipeline_transformers.transformers_[idx]
                a = a.split('_')[0]
                transforms.append({
                    'transform_name': a,
                    'fitted_transform': b,
                    'transform_features': c,
                })

        return {
            'imputer': {
                'filled_values': fitted_imputer.statistics_ if fitted_imputer else [],
                'missing_cols': fitted_imputer.missing_cols if fitted_imputer else [],
                'missing_col_indexs': fitted_imputer.missing_col_indexs if fitted_imputer else []
            },
            'transforms': transforms,
            'model': {
                'model_name': model_name,
                'trained_model': trained_model
            }
        }

    def generate_query(
        self,
        model_file,
        table_name,
        dbms,
        *,
        just_push_flag=False,
        masq=False,
        pre_sql=None,
        order_when=False,
        group='prune',
        cost_model='craftsman',
        max_process_num=1,
        assigned_rule=None
    ):
        assert group in ('org', 'pos', 'uncertain', 'enum', 'prune'), "group must in ('org', 'pos', 'uncertain', 'enum', 'prune')"
        assert cost_model in ('craftsman', 'postgresql'), "cost_model must in ('craftsman', 'postgresql')"

        # some load and extract tasks
        defs.DBMS = dbms
        defs.set_JUST_PUSH_FLAG(just_push_flag)
        defs.ORDER_WHEN = order_when
        defs.GROUP = group
        defs.MASQ = masq
        defs.TABLE_NAME = table_name

        model = load_model(model_file)
        pipeline_features_in = model.feature_names_in_.tolist()
        defs.PIPELINE_FEATURES_IN = pipeline_features_in
        pipeline = self.__extract_pipeline(model)
        data_rows = model.data_rows
        model_name = pipeline['model']['model_name']
        # build the graph of the preprocessing operators
        preprocessing_graph = PrepGraph(pipeline_features_in, pipeline)

        # for every implement plan, use different fusion plan, get the execute plan
        # calculate the cost of every execture plan, get the min cost plan
        min_cost = float("inf")

        if group == 'org':
            # reuse_flag = False
            # if reuse_flag and os.path.exists(f'{table_name}_{model_name}_{dbms}_org.pkl'):
            #     with open(f'{table_name}_{model_name}_{dbms}_org.pkl', 'rb') as f:
            #         min_cost_preprocessing_graph = pickle.load(f)

            # else:
            # enumerate the chain implement plans
            all_chain_candidate_implement_plans: list[ChainCandidateImplementPlans] = []
            for feature, chain in preprocessing_graph.chains.items():
                all_chain_candidate_implement_plans.append(ChainCandidateImplementPlans(feature, chain))

            def enumerate_graph_implement_plans(index=0, current_graph_implement_plan:list = []):
                if index == len(all_chain_candidate_implement_plans):
                    yield copy.deepcopy(current_graph_implement_plan) 
                    return
                for chain_implement_plan in all_chain_candidate_implement_plans[index].candidate_implement_plans:
                    current_graph_implement_plan.append(chain_implement_plan)
                    yield from enumerate_graph_implement_plans(index + 1, current_graph_implement_plan)
                    current_graph_implement_plan.pop()

            for graph_implement_plan in enumerate_graph_implement_plans():
                preprocessing_graph = implement_operator_by_plan(preprocessing_graph, graph_implement_plan)
                if cost_model == 'craftsman':
                    cost = get_craftsman_graph_cost(preprocessing_graph, data_rows)
                elif cost_model == 'postgresql':
                    query_str = self.__compose_sql(join_the_operators(preprocessing_graph), table_name, dbms, pre_sql, pipeline)
                    cost = get_pg_sql_cost(query_str)
                if cost < min_cost:
                    min_cost_preprocessing_graph = preprocessing_graph
                    min_cost = cost

            # with open(f'{table_name}_{model_name}_{dbms}_org.pkl', 'wb') as f:
            #     pickle.dump(min_cost_preprocessing_graph, f)

            t1 = time.time()
            min_cost_query_str = self.__compose_sql(join_the_operators(min_cost_preprocessing_graph), table_name, dbms, pre_sql, pipeline)
            t2 = time.time()
            print(f'org sql generate time: {(t2 - t1):.4f} s', flush=True)    

        if group == 'pos':
            # if os.path.exists(f'{table_name}_{model_name}_{dbms}_org.pkl'):
            #     with open(f'{table_name}_{model_name}_{dbms}_org.pkl', 'rb') as f:
            #         min_cost_preprocessing_graph = pickle.load(f)
            # else:
            # enumerate the chain implement plans
            all_chain_candidate_implement_plans: list[ChainCandidateImplementPlans] = []
            for feature, chain in preprocessing_graph.chains.items():
                all_chain_candidate_implement_plans.append(ChainCandidateImplementPlans(feature, chain))

            def enumerate_graph_implement_plans(index=0, current_graph_implement_plan:list = []):
                if index == len(all_chain_candidate_implement_plans):
                    yield copy.deepcopy(current_graph_implement_plan) 
                    return
                for chain_implement_plan in all_chain_candidate_implement_plans[index].candidate_implement_plans:
                    current_graph_implement_plan.append(chain_implement_plan)
                    yield from enumerate_graph_implement_plans(index + 1, current_graph_implement_plan)
                    current_graph_implement_plan.pop()

            for graph_implement_plan in enumerate_graph_implement_plans():
                preprocessing_graph = implement_operator_by_plan(preprocessing_graph, graph_implement_plan)
                if cost_model == 'craftsman':
                    cost = get_craftsman_graph_cost(preprocessing_graph, data_rows)
                elif cost_model == 'postgresql':
                    query_str = self.__compose_sql(join_the_operators(preprocessing_graph), table_name, dbms, pre_sql, pipeline)
                    cost = get_pg_sql_cost(query_str)
                if cost < min_cost:
                    min_cost_preprocessing_graph = preprocessing_graph
                    min_cost = cost

            min_cost_preprocessing_graph = merge_sql_operator_by_benifit_rules(min_cost_preprocessing_graph)
            t1 = time.time()
            min_cost_query_str = self.__compose_sql(join_the_operators(min_cost_preprocessing_graph), table_name, dbms, pre_sql, pipeline)
            t2 = time.time()
            print(f'pos sql generate time: {(t2 - t1):.4f} s', flush=True)
            # with open(f'{table_name}_{model_name}_{dbms}_pos.pkl', 'wb') as f:
            #     pickle.dump(min_cost_preprocessing_graph, f)

        if group == 'uncertain':
            # if os.path.exists(f'{table_name}_{model_name}_{dbms}_pos.pkl'):
            #     with open(f'{table_name}_{model_name}_{dbms}_pos.pkl', 'rb') as f:
            #         min_cost_preprocessing_graph = pickle.load(f)
            # elif os.path.exists(f'{table_name}_{model_name}_{dbms}_org.pkl'):
            #     with open(f'{table_name}_{model_name}_{dbms}_org.pkl', 'rb') as f:
            #         min_cost_preprocessing_graph = pickle.load(f)
            #     min_cost_preprocessing_graph = merge_sql_operator_by_benifit_rules(min_cost_preprocessing_graph)
            # else:
            # enumerate the chain implement plans
            all_chain_candidate_implement_plans: list[ChainCandidateImplementPlans] = []
            for feature, chain in preprocessing_graph.chains.items():
                all_chain_candidate_implement_plans.append(ChainCandidateImplementPlans(feature, chain))

            def enumerate_graph_implement_plans(index=0, current_graph_implement_plan:list = []):
                if index == len(all_chain_candidate_implement_plans):
                    yield copy.deepcopy(current_graph_implement_plan) 
                    return
                for chain_implement_plan in all_chain_candidate_implement_plans[index].candidate_implement_plans:
                    current_graph_implement_plan.append(chain_implement_plan)
                    yield from enumerate_graph_implement_plans(index + 1, current_graph_implement_plan)
                    current_graph_implement_plan.pop()

            for graph_implement_plan in enumerate_graph_implement_plans():
                preprocessing_graph = implement_operator_by_plan(preprocessing_graph, graph_implement_plan)
                if cost_model == 'craftsman':
                    cost = get_craftsman_graph_cost(preprocessing_graph, data_rows)
                elif cost_model == 'postgresql':
                    query_str = self.__compose_sql(preprocessing_graph, table_name, dbms, pre_sql, pipeline)
                    cost = get_pg_sql_cost(query_str)
                if cost < min_cost:
                    min_cost_preprocessing_graph = preprocessing_graph
                    min_cost = cost
            min_cost_preprocessing_graph = merge_sql_operator_by_benifit_rules(min_cost_preprocessing_graph)

            min_cost_preprocessing_graph = merge_sql_operator_by_uncertain_rules(min_cost_preprocessing_graph)
            t1 = time.time()
            min_cost_query_str = self.__compose_sql(join_the_operators(min_cost_preprocessing_graph), table_name, dbms, pre_sql, pipeline)
            t2 = time.time()
            print(f'uncertain sql generate time: {(t2 - t1):.4f} s', flush=True)

        if group == 'enum':
            concurrent_flag = True

            # enumerate the chain implement plans
            all_chain_candidate_implement_plans: list[ChainCandidateImplementPlans] = []
            for feature, chain in preprocessing_graph.chains.items():
                all_chain_candidate_implement_plans.append(ChainCandidateImplementPlans(feature, chain))

            def enumerate_graph_implement_plans(index=0, current_graph_implement_plan:list = []):
                if index == len(all_chain_candidate_implement_plans):
                    yield copy.deepcopy(current_graph_implement_plan) 
                    return
                for chain_implement_plan in all_chain_candidate_implement_plans[index].candidate_implement_plans:
                    current_graph_implement_plan.append(chain_implement_plan)
                    yield from enumerate_graph_implement_plans(index + 1, current_graph_implement_plan)
                    current_graph_implement_plan.pop()

            # enumerate the chain fusion plans
            all_chain_candidate_fusion_plans: list[ChainCandidateFusionPlans] = []
            for feature, chain in preprocessing_graph.chains.items():
                all_chain_candidate_fusion_plans.append(ChainCandidateFusionPlans(feature, chain))

            def enumerate_graph_fusion_plans(index=0, current_graph_fusion_plan:list = []):
                if index == len(all_chain_candidate_fusion_plans):
                    yield copy.deepcopy(current_graph_fusion_plan)
                    return
                for chain_fusion_plan in all_chain_candidate_fusion_plans[index].candidate_fusion_plans:
                    current_graph_fusion_plan.append(chain_fusion_plan)
                    yield from enumerate_graph_fusion_plans(index + 1, current_graph_fusion_plan)
                    current_graph_fusion_plan.pop()

            total_plan_num = 0

            if max_process_num == 1:
                for graph_implement_plan in enumerate_graph_implement_plans():
                    for graph_fusion_plan in enumerate_graph_fusion_plans():
                        preprocessing_graph_list = merge_sql_operator_by_graph_plan(preprocessing_graph, graph_implement_plan, graph_fusion_plan)
                        total_plan_num = total_plan_num + len(preprocessing_graph_list)
                        for graph in preprocessing_graph_list:
                            if cost_model == 'craftsman':
                                cost = get_craftsman_graph_cost(graph, data_rows)
                            elif cost_model == 'postgresql':
                                query_str = self.__compose_sql(graph, table_name, dbms, pre_sql, pipeline)
                                cost = get_pg_sql_cost(query_str)
                            if cost < min_cost:
                                min_cost_preprocessing_graph = graph
                                min_cost = cost

            else:
                all_graph_implements_plans = []
                for graph_implement_plan in enumerate_graph_implement_plans():
                    all_graph_implements_plans.append(graph_implement_plan)

                all_graph_fusion_plans = []
                for graph_fusion_plan in enumerate_graph_fusion_plans():
                    all_graph_fusion_plans.append(graph_fusion_plan)

                max_workers_num = max_process_num
                graph_fusion_plan_num = len(all_graph_fusion_plans)
                with ProcessPoolExecutor(max_workers=max_workers_num) as executor:
                    futures = []
                    if graph_fusion_plan_num >= max_workers_num:
                        per_worker_plan_num = graph_fusion_plan_num // max_workers_num
                        remain_plan_num = graph_fusion_plan_num % max_workers_num
                        begin_plan_idx = 0
                        for i in range(remain_plan_num):
                            futures.append(
                                executor.submit(
                                    task,
                                    all_graph_implements_plans,
                                    all_graph_fusion_plans[begin_plan_idx: begin_plan_idx + per_worker_plan_num + 1],
                                    preprocessing_graph.copy_graph(),
                                    cost_model,
                                    data_rows,
                                    table_name,
                                    dbms,
                                    pre_sql,
                                    pipeline,
                                )
                            )
                            begin_plan_idx = begin_plan_idx + per_worker_plan_num + 1
                        if per_worker_plan_num > 0:
                            for i in range(max_workers_num - remain_plan_num):
                                futures.append(
                                    executor.submit(
                                        task,
                                        all_graph_implements_plans,
                                        all_graph_fusion_plans[begin_plan_idx: begin_plan_idx + per_worker_plan_num],
                                        preprocessing_graph.copy_graph(),
                                        cost_model,
                                        data_rows,
                                        table_name,
                                        dbms,
                                        pre_sql,
                                        pipeline,
                                    )
                                )
                                begin_plan_idx = begin_plan_idx + per_worker_plan_num
                    else:
                        per_plan_worker_num = max_workers_num // graph_fusion_plan_num
                        remain_plan_num = max_workers_num % graph_fusion_plan_num
                        plan_worker_nums = [per_plan_worker_num] * (graph_fusion_plan_num - remain_plan_num) + [per_plan_worker_num + 1] * remain_plan_num
                        graph_implement_plan_num = len(all_graph_implements_plans)
                        for i, woker_num in enumerate(plan_worker_nums):
                            per_worker_implement_plan_num =  graph_implement_plan_num // woker_num
                            remain_implement_plan_num = graph_implement_plan_num % woker_num
                            begin_implement_plan_idx = 0
                            for j in range(remain_implement_plan_num):
                                futures.append(
                                    executor.submit(
                                        task,
                                        all_graph_implements_plans[begin_implement_plan_idx: begin_implement_plan_idx + per_worker_implement_plan_num + 1],
                                        all_graph_fusion_plans[i: i + 1],
                                        preprocessing_graph.copy_graph(),
                                        cost_model,
                                        data_rows,
                                        table_name,
                                        dbms,
                                        pre_sql,
                                        pipeline,
                                    )
                                )
                                begin_implement_plan_idx = begin_implement_plan_idx + per_worker_implement_plan_num + 1
                            if per_worker_implement_plan_num > 0:
                                for j in range(woker_num - remain_implement_plan_num):
                                    futures.append(
                                        executor.submit(
                                                task,
                                                all_graph_implements_plans[begin_implement_plan_idx: begin_implement_plan_idx + per_worker_implement_plan_num],
                                                all_graph_fusion_plans[i: i + 1],
                                                preprocessing_graph.copy_graph(),
                                                cost_model,
                                                data_rows,
                                                table_name,
                                                dbms,
                                                pre_sql,
                                                pipeline,
                                            )
                                    )
                                    begin_implement_plan_idx = begin_implement_plan_idx + per_worker_implement_plan_num    
                        
                    for future in futures:
                        try:
                            concurrent_min_cost_preprocessing_graph, concurrent_min_cost, process_plan_num = future.result()
                        except Exception as e:
                            print(f"Future raised an exception: {e}")
                        
                        total_plan_num += process_plan_num
                        if concurrent_min_cost < min_cost:
                            min_cost_preprocessing_graph = concurrent_min_cost_preprocessing_graph
                            min_cost = concurrent_min_cost  
            t1 = time.time()
            min_cost_query_str = self.__compose_sql(join_the_operators(min_cost_preprocessing_graph), table_name, dbms, pre_sql, pipeline)
            t2 = time.time()
            print(f'enum plan num: {total_plan_num}', flush=True)
            print(f'enum sql generate time: {(t2 - t1):.4f} s', flush=True)

        if group == 'prune':
            # enumerate the chain implement plans
            all_chain_candidate_implement_plans: list[ChainCandidateImplementPlans] = []
            for feature, chain in preprocessing_graph.chains.items():
                all_chain_candidate_implement_plans.append(ChainCandidateImplementPlans(feature, chain))

            def enumerate_graph_implement_plans(index=0, current_graph_implement_plan:list = []):
                if index == len(all_chain_candidate_implement_plans):
                    yield copy.deepcopy(current_graph_implement_plan) 
                    return
                for chain_implement_plan in all_chain_candidate_implement_plans[index].candidate_implement_plans:
                    current_graph_implement_plan.append(chain_implement_plan)
                    yield from enumerate_graph_implement_plans(index + 1, current_graph_implement_plan)
                    current_graph_implement_plan.pop()

            # enumerate the chain fusion plans
            all_chain_candidate_fusion_plans: list[ChainCandidateFusionPlans] = []
            for feature, chain in preprocessing_graph.chains.items():
                all_chain_candidate_fusion_plans.append(ChainCandidateFusionPlans(feature, chain))

            def enumerate_graph_fusion_plans(index=0, current_graph_fusion_plan:list = []):
                if index == len(all_chain_candidate_fusion_plans):
                    yield copy.deepcopy(current_graph_fusion_plan)
                    return
                for chain_fusion_plan in all_chain_candidate_fusion_plans[index].candidate_fusion_plans:
                    current_graph_fusion_plan.append(chain_fusion_plan)
                    yield from enumerate_graph_fusion_plans(index + 1, current_graph_fusion_plan)
                    current_graph_fusion_plan.pop()

            total_plan_num = 0        
            last_chain_min_cost_preprocessing_graph = preprocessing_graph.copy_graph()
            for chain_candidate_implement_plans, chain_candidate_fusion_plans, feature in zip(
                all_chain_candidate_implement_plans,
                all_chain_candidate_fusion_plans,
                preprocessing_graph.chains.keys(),
            ):
                for chain_implement_plan in chain_candidate_implement_plans.candidate_implement_plans:
                    for chain_fusion_plan in chain_candidate_fusion_plans.candidate_fusion_plans:
                        if feature == 'hours-per-week':
                            pass
                        preprocessing_graph_list = merge_sql_operator_by_chain_plan(last_chain_min_cost_preprocessing_graph, chain_implement_plan, chain_fusion_plan, feature, assigned_rule)
                        
                        total_plan_num += len(preprocessing_graph_list)
                        # print(f'current plan num: {total_plan_num}')
                        # t1 = time.time()
                        for graph in preprocessing_graph_list:
                            if defs.MASQ:
                                min_cost_preprocessing_graph = graph
                            else:
                                if cost_model == 'craftsman':
                                    cost = get_craftsman_graph_cost(graph, data_rows)
                                elif cost_model == 'postgresql':
                                    query_str = self.__compose_sql(graph, table_name, dbms, pre_sql, pipeline)
                                    cost = get_pg_sql_cost(query_str)
                                if cost < min_cost:
                                    min_cost_preprocessing_graph = graph
                                    min_cost = cost
                        # t2 = time.time()
                        # print(f'{len(preprocessing_graph_list)} plans\' cost evaluate time: {(t2 - t1):.4f} s', flush=True)
                last_chain_min_cost_preprocessing_graph = min_cost_preprocessing_graph.copy_graph()

            t1 = time.time()
            min_cost_query_str = self.__compose_sql(join_the_operators(min_cost_preprocessing_graph), table_name, dbms, pre_sql, pipeline)
            t2 = time.time()
            print(f'prune plan num: {total_plan_num}', flush=True)
            print(f'prune sql generate time: {(t2 - t1):.4f} s', flush=True)

        return min_cost_query_str
        # return min_cost_query_str, min_cost_graph_implement_plan, min_cost_graph_fusion_plan, min_cost_preprocessing_graph

        # merge operators by rules
        # if merge_flag and not masq:
        #     if defs.TIMER == 'ON':
        #         t1 = time.time()
        #         preprocessing_graph = merge_sql_operator_by_rules(preprocessing_graph)
        #         t2 = time.time()
        #         print(f'Apply Rules Time: {t2 - t1} s')
        #     else:
        #         preprocessing_graph = merge_sql_operator_by_rules(preprocessing_graph)

        # merge operators by cost model
        # if defs.TIMER == 'ON':
        #     t1 = time.time()
        #     preprocessing_graph = merge_by_cost_model(preprocessing_graph, train_data, merge_flag, cost_flag, masq)
        #     t2 = time.time()
        #     print(f'Cost Model Time: {t2 - t1} s')
        # else:
        #     preprocessing_graph = merge_by_cost_model(preprocessing_graph, train_data, merge_flag, cost_flag, masq)

        # generate sql through the merged graph
        # if defs.TIMER == 'ON':
        #     t1 = time.time()
        #     query_str = self.__compose_sql(preprocessing_graph, table_name, dbms, pre_sql, pipeline)
        #     t2 = time.time()
        #     print(f'Generate SQL Time: {t2 - t1} s')
        # else:
        #     query_str = self.__compose_sql(preprocessing_graph, table_name, dbms, pre_sql, pipeline)

        # return query_str

    def __compose_sql(self, graph: PrepGraph, table_name: str, dbms: str, pre_sql: str, pipeline) -> str:
        input_table = table_name
        expend_features = {}

        # compose join sqls
        join_feature_sqls = {}
        join_feature_list = {}
        first_join_op_feature = []
        left_join_ops = []
        for op in graph.join_operators:
            if op.features[0] not in first_join_op_feature:
                first_join_op_feature.append(op.features[0])
                input_table, feature_sql, join_features = op.get_join_sql(dbms, input_table, table_name, pipeline)
                join_feature_sqls[op.features[0]] = feature_sql
                join_feature_list[op.features[0]] = join_features
            else:
                left_join_ops.append(op)
        graph.join_operators = left_join_ops     

        # compose imputer sql if exists missing cols
        if pipeline['imputer']['missing_cols'] or join_feature_sqls:
            filled_values = pipeline['imputer']['filled_values']
            missing_cols = pipeline['imputer']['missing_cols']
            missing_col_indexs = pipeline['imputer']['missing_col_indexs']
            imputer_feature_sqls = []
            imputer_sql = '(SELECT '
            fill_sqls = {}
            for idx, feature in enumerate(missing_cols):
                fill_sqls[feature] = filled_values[missing_col_indexs[idx]]
            for feature, chain in graph.chains.items():
                delimited_feature = DBMSUtils.get_delimited_col(dbms, feature)
                if join_feature_sqls.get(feature):
                    imputer_feature_sqls.append(join_feature_sqls[feature])
                    join_feature_sqls.pop(feature)
                    if len(join_feature_list.get(feature)) > 1:
                        expend_features[feature] = join_feature_list.get(feature)
                elif feature in fill_sqls:
                    if type(fill_sqls[feature]) == str:
                        imputer_feature_sqls.append(f'COALESCE({delimited_feature}, \'{fill_sqls[feature]}\') AS {delimited_feature}')
                    else:
                        imputer_feature_sqls.append(f'COALESCE({delimited_feature}, {fill_sqls[feature]}) AS {delimited_feature}')
                else:
                    imputer_feature_sqls.append(f'{delimited_feature}')
            imputer_sql += ','.join(imputer_feature_sqls)
            imputer_sql += f' FROM {input_table}) AS data'
            input_table = imputer_sql

        # compose preprocessing sqls
        max_level = 0
        for _, chain in graph.chains.items():
            max_level = max(max_level, len(chain.prep_operators) - 1)

        prep_sqls = []

        prep_level = 0
        while prep_level <= max_level or len(join_feature_sqls) > 0:
            perp_level_sqls = []
            for feature, chain in graph.chains.items():
                if prep_level > 1 and len(graph.implements[feature]) > prep_level and graph.implements[feature][prep_level] == SQLPlanType.JOIN:
                    input_table, feature_sql, join_features = op.get_join_sql(dbms, input_table, table_name, pipeline)
                    join_feature_sqls[op.features[0]] = feature_sql
                    join_feature_list[op.features[0]] = join_features
                if len(chain.prep_operators) > prep_level:
                    op = chain.prep_operators[prep_level]
                    perp_level_sqls.append(op.get_sql(dbms))
                    if op.features != op.features_out:
                        expend_features[feature] = op.features_out
                else:
                    if expend_features.get(feature):
                        perp_level_sqls.append(','.join([DBMSUtils.get_delimited_col(dbms, f) for f in expend_features[feature]]))
                    elif join_feature_sqls.get(feature):
                        perp_level_sqls.append(join_feature_sqls[feature])
                        join_feature_sqls.pop(feature)
                        if len(join_feature_list.get(feature)) > 1:
                            expend_features[feature] = join_feature_list.get(feature)
                    else:
                        perp_level_sqls.append(DBMSUtils.get_delimited_col(dbms, feature))
            prep_sqls.append(','.join(perp_level_sqls))
            prep_level += 1

        for prep_sql in prep_sqls:
            # the input table for the possible next transformer is the output of the current transformer
            input_table = "({}) AS data".format('SELECT ' + prep_sql + f' FROM {input_table}')

        # compose model sql
        predication_query = graph.model.query(input_table, dbms)

        return pre_sql + predication_query
