from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from pandas import DataFrame, Series
from sympy import Eq, solve, lambdify
from craftsman.base.defs import *
import craftsman.base.defs as defs
from craftsman.utility.dbms_utils import DBMSUtils
from craftsman.utility.join_utils import insert_db, df_type2db_type
from craftsman.utility.base_utils import merge_intervals
from craftsman.cost_model.utils import calc_join_cost_by_train_data


class SQLOperator(ABC):

    def __init__(self, op_name: OperatorName):
        self.op_name: OperatorName = op_name
        self.input_data_type: DataType
        self.output_data_type: DataType
        self.calculation_type: CalculationType
        self.op_type: OperatorType
        self.features: list[str]
        self.features_out: list[str] = []
        self.costs: dict
        self.stats = []
        self.is_encoder: bool = False
        self.is_arithmetic_op: bool = False
        self.is_inequality_judgment_op: bool = False
        self.is_contain_ca_op: bool = False
        self.is_constant_output_op: bool = False
        self.is_contain_multi_ca_op: bool = False

    @abstractmethod
    def apply(self, first_op: Operator):
        """constant propagation
        Args:
            first_op (SQLOperator): _description_
        """

    @abstractmethod
    def simply(self, second_op: Operator):
        """algebraic simplification

        Args:
            second_op (SQLOperator): _description_
        """

    @abstractmethod
    def _extract(self, fitted_transform) -> None:
        """absract the infomation from the fitted_transorm

        Args:
            fitted_transform (object): sklearn-base model from sklearn or catgory_encoders
        """

    def _get_op_type(self) -> str:
        """return the father class type

        Returns:
            str: father class type
        """
        return self.__class__.__base__.__name__

    @staticmethod
    def trans_feature_names_in(input_data: DataFrame):
        """_summary_

        Args:
            input_data (DataFrame): _description_
        """

    @abstractmethod
    def get_sql(self, dbms: str):
        pass

    def fusion(self, graph):
        for feature in self.features_out:
            if hasattr(graph, "model"):
                graph.model.modify_model(feature, self)
            else:
                graph.modify_model(feature, self)

    @abstractmethod
    def modify_leaf(self, feature, op, thr, modified_feature):
        pass

    def push(self, graph):
        for feature in self.features_out:
            graph.model.modify_model_p(feature, self)

    @abstractmethod
    def modify_leaf_p(self, feature, op, thr):
        pass

    @abstractmethod
    def _get_op_cost(self, feature):
        pass

    def __get_case_cost(self, feature, graph):
        if graph.model.model_name in (
            ModelName.DECISIONTREECLASSIFIER,
            ModelName.RANDOMFORESTCLASSIFIER,
            ModelName.DECISIONTREEREGRESSOR,
            ModelName.RANDOMFORESTREGRESSOR,
        ):
            tree_costs = graph.model.get_tree_costs(feature, self)
            total_model_cost = sum(
                [tree_cost.calculate_no_fusion_cost() for tree_cost in tree_costs]
            )
        else:
            total_model_cost = 0
        op_cost = self._get_op_cost(feature)
        return total_model_cost + op_cost

    def __get_fusion_cost(self, feature, graph):
        tree_costs = graph.model.get_tree_costs(feature, self)
        total_fusion_cost = sum(
            [tree_cost.calculate_tree_cost() for tree_cost in tree_costs]
        )
        return total_fusion_cost

    def __get_push_cost(self, feature, graph):
        tree_costs = graph.model.get_tree_costs_p(feature, self)
        total_push_cost = sum(
            [tree_cost.calculate_tree_cost() for tree_cost in tree_costs]
        )
        return total_push_cost

    def get_best_plan(self, graph, data_rows) -> SQLPlanType:
        if graph.model.model_name in (
            ModelName.DECISIONTREECLASSIFIER,
            ModelName.RANDOMFORESTCLASSIFIER,
            ModelName.DECISIONTREEREGRESSOR,
            ModelName.RANDOMFORESTREGRESSOR,
        ):
            fusion_cost = 0
            push_cost = 0
            for feature in self.features_out:
                fusion_cost += self.__get_fusion_cost(feature, graph)
                push_cost += self.__get_push_cost(feature, graph)
        else:
            fusion_cost = float("inf")
            push_cost = float("inf")

        case_cost = 0
        for feature in self.features_out:
            case_cost += self.__get_case_cost(feature, graph)

        self.costs = {
            SQLPlanType.FUSION.value: fusion_cost,
            SQLPlanType.CASE.value: case_cost,
            SQLPlanType.PUSH.value: push_cost,
        }

        if self.is_encoder:
            join_cost = self._get_join_cost(self.features[0], graph, data_rows)
            self.costs[SQLPlanType.JOIN.value] = join_cost
        else:
            self.costs[SQLPlanType.JOIN.value] = float("inf")

        sorted_costs = sorted(self.costs.items(), key=lambda item: item[1])

        # --------------------!!!!!!!!!!!DEBUG CODE!!!!!!!!!!------------------------------
        # print(f'{self.op_name}, {self.features[0]}')
        # print(sorted_costs)
        # --------------------!!!!!!!!!!!DEBUG CODE!!!!!!!!!!------------------------------

        return SQLPlanType(sorted_costs[0][0]), [
            self.op_name.value,
            self.features[0],
            self.costs[SQLPlanType.CASE.value],
            self.costs[SQLPlanType.FUSION.value],
            self.costs[SQLPlanType.JOIN.value],
            self.costs[SQLPlanType.PUSH.value],
        ]

    @abstractmethod
    def get_fusion_primitive_type(self, feature, thr):
        pass

    @abstractmethod
    def get_fusion_primitive_length(self, feature, thr):
        pass

    @abstractmethod
    def get_push_primitive_type(self, feature, thr):
        pass

    @abstractmethod
    def get_push_primitive_length(self, feature, thr):
        pass


class EncoderOperator(SQLOperator):

    def __init__(self, op_name: OperatorName):
        super().__init__(op_name)
        self.is_encoder = True
        self.is_contain_ca_op = True
        self.is_constant_output_op = True

    def join(self, graph):
        graph.add_join_operator(self)

    @abstractmethod
    def get_join_sql(self, dbms: str, input_table: str, table_name: str, pipeline):
        pass

    @abstractmethod
    def _get_join_cost(self, feature, graph, data_rows):
        pass

    @abstractmethod
    def _get_join_cost_without_tree(self, feature, graph, data_rows):
        pass


Operator = Type[SQLOperator]


class CAT_C_CAT(EncoderOperator):

    def __init__(self, op_name: OperatorName):
        super().__init__(op_name)
        self.input_data_type = DataType.CAT
        self.output_data_type = DataType.CAT
        self.calculation_type = CalculationType.COMPARISON
        self.op_type = OperatorType[self._get_op_type()]

        self.mappings: list[Series] = []
        self.value_counts: dict

    def apply(self, first_op: Operator):
        if first_op.op_type == OperatorType.CON_C_CAT:
            merged_op = CON_C_CAT_Merged_OP(first_op)
            merged_op.bin_edges = first_op.bin_edges
            for idx, mapping in enumerate(self.mappings):
                merged_op.categories[idx] = (
                    Series(first_op.categories[idx])
                    .apply(lambda x: mapping[x] if x in mapping.index else 0)
                    .values
                )
            return merged_op

        elif first_op.op_type == OperatorType.CAT_C_CAT:
            merged_op = CAT_C_CAT_Merged_OP(first_op)
            for idx, mapping in enumerate(first_op.mappings):
                merged_op.mappings.append(
                    Series(
                        Series(mapping)
                        .apply(
                            lambda x: (
                                self.mappings[idx][x]
                                if x in self.mappings[idx].index
                                else 0
                            )
                        )
                        .values,
                        index=mapping.index,
                    )
                )
            return merged_op

        elif first_op.op_type == OperatorType.EXPAND:
            merged_op = EXPAND_Merged_OP(first_op)
            merged_op.mapping = first_op.mapping
            for idx, col in enumerate(first_op.mapping.columns):
                merged_op.mapping[col] = self.mappings[idx][merged_op.mapping[col]]
            return merged_op

        else:
            return None

    def simply(self, second_op: Operator):
        if second_op.op_type == OperatorType.EXPAND:
            merged_op = EXPAND_Merged_OP(second_op)
            merged_op.mapping = second_op.mapping
            index_mapping = self.mappings[0]
            reverse_index_mapping = Series(
                data=index_mapping.index, index=index_mapping
            )
            merged_op.mapping.index = reverse_index_mapping[merged_op.mapping.index]
            return merged_op

        else:
            return None

    def get_sql(self, dbms: str):
        sqls = []
        for idx in range(len(self.features)):
            feature_sql = "CASE "
            mapping = self.mappings[idx]
            if defs.ORDER_WHEN:
                value_counts = np.array(
                    [
                        self.value_counts[self.features[idx]][category]
                        for category in mapping.index
                    ]
                )
                pos_2_val = np.argsort(-value_counts)
            if mapping.index.inferred_type == 'mixed':
                for i in range(len(mapping.index)):
                    if defs.ORDER_WHEN:
                        val_idx = pos_2_val[i]
                        interval = mapping.index[val_idx]
                    else:
                        interval = mapping.index[i]
                    
                    feature_sql += f"WHEN {DBMSUtils.get_delimited_col(dbms, self.features[idx])} >= {interval[0]} AND {DBMSUtils.get_delimited_col(dbms, self.features[idx])} < {interval[1]} THEN {mapping[category]} "
            else:
                for i in range(len(mapping.index)):
                    if defs.ORDER_WHEN:
                        val_idx = pos_2_val[i]
                        category = mapping.index[val_idx]
                    else:
                        category = mapping.index[i]
                    if type(category) == str:
                        feature_sql += f"WHEN {DBMSUtils.get_delimited_col(dbms, self.features[idx])} = '{category}' THEN {mapping[category]} "
                    else:
                        feature_sql += f"WHEN {DBMSUtils.get_delimited_col(dbms, self.features[idx])} = {category} THEN {mapping[category]} "
            feature_sql += (
                f"END AS {DBMSUtils.get_delimited_col(dbms, self.features[idx])}"
            )
            sqls.append(feature_sql)
        return ",".join(sqls)

    def get_join_sql(self, dbms: str, input_table: str, table_name: str, pipeline):
        feature = self.features[0]
        mapping = self.mappings[0]
        join_table_name = feature + CAT_C_CAT_JOIN_POSTNAME
        join_table_name = join_table_name.lower()
        col_name = feature + CAT_C_CAT_JOIN_COL_POSTNAME
        col_name = col_name.lower()
        if self.op_name == OperatorName.CAT_C_CAT_Merged_OP:
            cols = {
                feature.lower(): (
                    DBDataType.VARCHAR.value
                    if dbms != "monetdb"
                    else DBDataType.VARCHAR512.value
                )
            }
            cols[col_name] = df_type2db_type(mapping.dtype, dbms)
            data = [
                (idx, mapping.tolist()[mapping.index.get_loc(idx)])
                for idx in mapping.index
            ]
            insert_db(dbms, join_table_name, cols, data)
        delimitied_feature = DBMSUtils.get_delimited_col(dbms, feature)
        missing_cols = pipeline["imputer"]["missing_cols"]
        if feature in missing_cols:
            missing_col_indexs = pipeline["imputer"]["missing_col_indexs"]
            filled_values = pipeline["imputer"]["filled_values"]
            fill_sqls = {}
            for idx, col in enumerate(missing_cols):
                fill_sqls[col] = filled_values[missing_col_indexs[idx]]
            if type(fill_sqls[feature]) != str:
                input_table = f"{input_table} left join {join_table_name} on COALESCE({table_name}.{delimitied_feature},{fill_sqls[feature]})={join_table_name}.{DBMSUtils.get_delimited_col(dbms, feature.lower())}"
            else:
                input_table = f"{input_table} left join {join_table_name} on COALESCE({table_name}.{delimitied_feature},'{fill_sqls[feature]}')={join_table_name}.{DBMSUtils.get_delimited_col(dbms, feature.lower())}"
        else:
            input_table = f"{input_table} left join {join_table_name} on {table_name}.{delimitied_feature}={join_table_name}.{DBMSUtils.get_delimited_col(dbms, feature.lower())}"
        featuer_sql = (
            f"{DBMSUtils.get_delimited_col(dbms, col_name)} AS {delimitied_feature}"
        )
        return input_table, featuer_sql, self.features_out

    def modify_leaf(self, feature, op, thr, modified_feature):
        mapping = self.mappings[self.features_out.index(feature)]
        # if feature == 'nom_6':
        #     pass
        if op == "<=":
            in_list = []
            # if defs.ORDER_WHEN:
            in_list_value_counts = []
            for idx, enc_value in mapping.items():
                if enc_value <= thr:
                    if defs.ORDER_WHEN:
                        in_list_value_counts.append(self.value_counts[feature][idx])
                    if type(idx) == str:
                        in_list.append(f"'{idx}'")
                    else:
                        in_list.append(f"{idx}")
            if defs.ORDER_WHEN:
                in_list = [
                    in_list[pos] for pos in np.argsort(-np.array(in_list_value_counts))
                ]
            if in_list:
                return (
                    DBMSUtils.get_delimited_col(defs.DBMS, feature),
                    "in",
                    f"({','.join(in_list)})",
                )
            else:
                return (
                    'False',
                    '',
                    '',
                )

        elif op == "in":
            in_list = []
            leaf_in_list = [float(x) for x in thr[1:-1].split(",")]
            for idx, enc_value in mapping.items():
                if enc_value in leaf_in_list:
                    if type(idx) == str:
                        in_list.append(f"'{idx}'")
                    else:
                        in_list.append(f"{idx}")

            if in_list:
                return (
                    DBMSUtils.get_delimited_col(defs.DBMS, feature),
                    "in",
                    f"({','.join(in_list)})",
                )
            else:
                return (
                    'False',
                    '',
                    '',
                )

        elif op == "":
            intervals = []
            for inequality in modified_feature.split("OR"):
                left_bin = float(inequality.split("AND")[0].split(">=")[1])
                right_bin = float(inequality.split("AND")[1].split("<")[1])
                intervals.append((left_bin, right_bin))

            def judge_in_intervals(value):
                for interval in intervals:
                    left_bin, right_bin = interval
                    if value >= left_bin and value < right_bin:
                        return True
                return False

            in_list = []
            for idx, enc_value in mapping.items():
                if judge_in_intervals(enc_value):
                    if type(idx) == str:
                        in_list.append(f"'{idx}'")
                    else:
                        in_list.append(f"{idx}")
            
            if in_list:
                return (
                    DBMSUtils.get_delimited_col(defs.DBMS, feature),
                    "in",
                    f"({','.join(in_list)})",
                )
            else:
                return (
                    'False',
                    '',
                    '',
                )

    def modify_leaf_p(self, feature, op, thr):
        mapping = self.mappings[self.features_out.index(feature)]
        feature_sql = "CASE "
        if defs.ORDER_WHEN:
            value_counts = np.array(
                [self.value_counts[feature][category] for category in mapping.index]
            )
            pos_2_val = np.argsort(-value_counts)
        for i in range(len(mapping.index)):
            if defs.ORDER_WHEN:
                val_idx = pos_2_val[i]
                category = mapping.index[val_idx]
            else:
                category = mapping.index[i]
            if type(category) == str:
                feature_sql += f"WHEN {DBMSUtils.get_delimited_col(defs.DBMS, feature)} = '{category}' THEN {mapping[category]} "
            else:
                feature_sql += f"WHEN {DBMSUtils.get_delimited_col(defs.DBMS, feature)} = {category} THEN {mapping[category]} "
        feature_sql += f"END"
        # for i, item in enumerate(mapping[:-1].items()):
        #     idx, val = item
        #     if type(idx) == str:
        #         feature_sub_sql += f"WHEN {DBMSUtils.get_delimited_col(defs.DBMS, feature)} = '{idx}' THEN {val} "
        #     else:
        #         feature_sub_sql += f"WHEN {DBMSUtils.get_delimited_col(defs.DBMS, feature)} = {idx} THEN {val} "
        # feature_sub_sql += "ELSE {} END ".format(mapping.iloc[-1])
        return feature_sql, op, thr

    def _get_join_cost(self, feature, graph, data_rows):
        if graph.model.model_name in (
            ModelName.DECISIONTREECLASSIFIER,
            ModelName.RANDOMFORESTCLASSIFIER,
            ModelName.DECISIONTREEREGRESSOR,
            ModelName.RANDOMFORESTREGRESSOR,
        ):
            tree_costs = graph.model.get_tree_costs(feature, self)
            total_tree_cost = sum(
                [tree_cost.calculate_no_fusion_cost() for tree_cost in tree_costs]
            )
        else:
            total_tree_cost = 0
        join_cost = calc_join_cost_by_train_data(
            data_rows, len(self.mappings[self.features_out.index(feature)]), 1
        )
        return total_tree_cost + join_cost

    def _get_join_cost_without_tree(self, feature, graph, data_rows):
        join_cost = calc_join_cost_by_train_data(
            data_rows, len(self.mappings[self.features_out.index(feature)]), 1
        )
        return join_cost

    def get_fusion_primitive_type(self, feature, thr):
        return PrimitiveType.IN

    def get_fusion_primitive_length(self, feature, thr):
        mapping = self.mappings[self.features.index(feature)]
        in_length = 0
        for enc_value in mapping:
            if enc_value <= thr:
                in_length += 1
        return in_length

    def _get_op_cost(self, feature):
        mapping = self.mappings[self.features.index(feature)]
        cached = False
        if defs.value_counts_sum_cache.get(feature):
            cached = True
        if not cached:
            value_counts = np.array(
                [self.value_counts[feature][category] for category in mapping.index]
            )
        
        if defs.ORDER_WHEN:
            if not cached:
                pos_2_val = np.argsort(-value_counts)
                val_2_pos = {val: pos for pos, val in enumerate(pos_2_val)}
                value_sum = sum([(val_2_pos[i] + 1) * num for i, num in enumerate(value_counts)])
                defs.value_counts_sum_cache[feature] = value_sum
            else:
                value_sum = defs.value_counts_sum_cache.get(feature)
            
            if mapping.index.inferred_type == 'mixed':
                return (
                    value_sum
                    * PrimitiveCost.OR.value
                )
            else:
                return (
                    value_sum
                    * PrimitiveCost.EQUAL.value
                )
        else:
            if not cached:
                value_sum = sum([(i + 1) * num for i, num in enumerate(value_counts)])
                defs.value_counts_sum_cache[feature] = value_sum
            else:
                value_sum = defs.value_counts_sum_cache.get(feature)
                
            if mapping.index.inferred_type == 'mixed':
                return (
                    value_sum
                    * PrimitiveCost.OR.value
                )
            else:
                return (
                    value_sum
                    * PrimitiveCost.EQUAL.value
                )

    def get_push_primitive_type(self, feature, thr):
        return PrimitiveType.EQUAL

    def get_push_primitive_length(self, feature, thr):
        mapping = self.mappings[self.features.index(feature)]
        # value_counts = np.array([self.value_counts[feature][category] for category in mapping.index])
        # if defs.ORDER_WHEN:
        #     pos_2_val = np.argsort(-value_counts)
        #     val_2_pos = {val:pos for pos, val in enumerate(pos_2_val)}
        if defs.PUSH_USE_AVERAGE:
            # if defs.ORDER_WHEN:

            # else:
            return len(mapping) / 2
        else:
            # if defs.ORDER_WHEN:

            # else:
            return len(mapping)
        # TODO: use more accurate length


class EXPAND(EncoderOperator):

    def __init__(self, op_name: OperatorName):
        super().__init__(op_name)
        self.input_data_type = DataType.CAT
        self.output_data_type = DataType.CAT
        self.calculation_type = CalculationType.COMPARISON
        self.op_type = OperatorType[self._get_op_type()]

        self.mapping: DataFrame | Series
        self.con_c_cat_mapping = None  # using for the merge of the con_c_cat and expand
        self.is_contain_multi_ca_op = True

    def apply(self, first_op: Operator):
        return None

    def simply(self, second_op: Operator):
        return None

    def get_sql(self, dbms: str):
        sqls = []
        feature = self.features[0]
        for col in self.mapping.columns:
            col_sql = "CASE "
            col_mapping = self.mapping[col]
            col_mapping = col_mapping[~col_mapping.index.isnull()]
            col_mapping = col_mapping[
                [idx for idx in col_mapping.index if idx != "NaN"]
            ]
            categories_list = col_mapping.groupby(col_mapping).apply(
                lambda x: x.index.tolist()
            )
            sorted_categories_list = categories_list.apply(
                lambda x: len(x)
            ).sort_values(ascending=True)
            categories_list = Series(
                categories_list[sorted_categories_list.index],
                index=sorted_categories_list.index,
            )
            if self.con_c_cat_mapping is None:
                for enc_value in categories_list.index.tolist()[:-1]:
                    if isinstance(categories_list[enc_value][0], tuple):
                        inquality_str = " OR ".join(
                                    [
                                        f"{DBMSUtils.get_delimited_col(feature)} >= {interval[0]} AND {DBMSUtils.get_delimited_col(feature)} < {interval[1]}" 
                                        for interval in categories_list[enc_value]
                                    ]
                                )
                        col_sql += f"WHEN {inquality_str} THEN {enc_value} "
                    else:
                        if len(categories_list[enc_value]) == 1:
                            if type(categories_list[enc_value][0]) == str:
                                col_sql += f"WHEN {DBMSUtils.get_delimited_col(dbms, feature)} = '{categories_list[enc_value][0]}' THEN {enc_value} "
                            else:
                                col_sql += f"WHEN {DBMSUtils.get_delimited_col(dbms, feature)} = {categories_list[enc_value][0]} THEN {enc_value} "
                        else:
                            if categories_list[enc_value]:
                                in_str = ",".join(
                                    [
                                        f"'{c}'" if type(c) == str else f"{c}"
                                        for c in categories_list[enc_value]
                                    ]
                                )
                                col_sql += f"WHEN {DBMSUtils.get_delimited_col(dbms, feature)} in ({in_str}) THEN {enc_value} "
                            else:
                                col_sql += f'WHEN False THEN {enc_value} '
            else:
                for enc_value in categories_list.index.tolist()[:-1]:
                    intervals = []
                    for c in categories_list[enc_value]:
                        intervals.extend(self.con_c_cat_mapping[c])
                    merged_intervals = merge_intervals(intervals)
                    condition_str = " OR ".join(
                        [
                            f"{DBMSUtils.get_delimited_col(dbms, feature)} >= {interval[0]}"
                            + " AND "
                            + f"{DBMSUtils.get_delimited_col(dbms, feature)} < {interval[1]}"
                            for interval in merged_intervals
                        ]
                    )
                    col_sql += f"WHEN {condition_str} THEN {enc_value} "

            if col_sql == "CASE ":
                col_sql = f"{categories_list.index.tolist()[-1]} AS {DBMSUtils.get_delimited_col(dbms, col)}"
            else:
                col_sql += f"ELSE {categories_list.index.tolist()[-1]} END AS {DBMSUtils.get_delimited_col(dbms, col)}"
            sqls.append(col_sql)

        return ",".join(sqls)

    def get_join_sql(self, dbms: str, input_table: str, table_name: str, pipeline):
        feature = self.features[0]
        join_table_name = feature + EXPAND_JOIN_POSTNAME
        join_table_name = join_table_name.lower()
        if self.op_name == OperatorName.EXPAND_Merged_OP:
            cols = {
                feature.lower(): (
                    DBDataType.VARCHAR.value
                    if dbms != "monetdb"
                    else DBDataType.VARCHAR512.value
                )
            }
            for col in self.mapping.columns:
                cols[col.lower()] = df_type2db_type(self.mapping[col].dtype, dbms)
            data = []
            for idx in self.mapping.index:
                data.append((idx,) + tuple(self.mapping.loc[idx]))
            insert_db(dbms, join_table_name, cols, data)
        delimitied_feature = DBMSUtils.get_delimited_col(dbms, feature)
        missing_cols = pipeline["imputer"]["missing_cols"]
        if feature in missing_cols:
            missing_col_indexs = pipeline["imputer"]["missing_col_indexs"]
            filled_values = pipeline["imputer"]["filled_values"]
            fill_sqls = {}
            for idx, col in enumerate(missing_cols):
                fill_sqls[col] = filled_values[missing_col_indexs[idx]]
            if type(fill_sqls[feature]) != str:
                input_table = f"{input_table} left join {join_table_name} on COALESCE({table_name}.{delimitied_feature},{fill_sqls[feature]})={join_table_name}.{DBMSUtils.get_delimited_col(dbms, feature.lower())}"
            else:
                input_table = f"{input_table} left join {join_table_name} on COALESCE({table_name}.{delimitied_feature},'{fill_sqls[feature]}')={join_table_name}.{DBMSUtils.get_delimited_col(dbms, feature.lower())}"
        else:
            input_table = f"{input_table} left join {join_table_name} on {table_name}.{delimitied_feature}={join_table_name}.{DBMSUtils.get_delimited_col(dbms, feature.lower())}"
        feature_sql = ",".join(
            [
                f"{DBMSUtils.get_delimited_col(dbms, c.lower())} AS {DBMSUtils.get_delimited_col(dbms, c)}"
                for c in self.mapping.columns.tolist()
            ]
        )
        return input_table, feature_sql, self.features_out

    def modify_leaf(self, feature, op, thr, modified_feature):
        mapping = self.mapping[feature]
        mapping = mapping[~mapping.index.isnull()]
        mapping = mapping[[idx for idx in mapping.index if idx != "NaN"]]
        feature = self.features[0]
        if self.con_c_cat_mapping is None:
            in_list = []
            not_in_list = []
            for idx, enc_value in mapping.items():
                if enc_value <= thr:
                    if type(idx) == str:
                        in_list.append(f"'{idx}'")
                    else:
                        in_list.append(f"{idx}")
                else:
                    if type(idx) == str:
                        not_in_list.append(f"'{idx}'")
                    else:
                        not_in_list.append(f"{idx}")

            if len(not_in_list) == 1:
                return (
                    DBMSUtils.get_delimited_col(defs.DBMS, feature),
                    "<>",
                    f"{not_in_list[0]}",
                )
            elif len(in_list) == 1:
                return (
                    DBMSUtils.get_delimited_col(defs.DBMS, feature),
                    "=",
                    f"{in_list[0]}",
                )
            return (
                DBMSUtils.get_delimited_col(defs.DBMS, feature),
                "in",
                f"({','.join(in_list)})",
            )
        else:
            all_intervals = []
            for enc_value, intervals in self.con_c_cat_mapping.items():
                if enc_value <= thr:
                    all_intervals.extend(intervals)
            merged_intervals = merge_intervals(all_intervals)
            condition_str = " OR ".join(
                [
                    f"{DBMSUtils.get_delimited_col(defs.DBMS, feature)} >= {interval[0]}"
                    + " AND "
                    + f"{DBMSUtils.get_delimited_col(defs.DBMS, feature)} < {interval[1]}"
                    for interval in merged_intervals
                ]
            )
            return condition_str, "", ""

    def modify_leaf_p(self, feature, op, thr):
        mapping = self.mapping[feature]
        mapping = mapping[~mapping.index.isnull()]
        mapping = mapping[[idx for idx in mapping.index if idx != "NaN"]]
        feature = self.features[0]
        categories_list = mapping.groupby(mapping).apply(lambda x: x.index.tolist())
        sorted_categories_list = categories_list.apply(lambda x: len(x)).sort_values(
            ascending=True
        )
        categories_list = Series(
            categories_list[sorted_categories_list.index],
            index=sorted_categories_list.index,
        )
        if self.con_c_cat_mapping is None:
            col_sql = "CASE "
            for enc_value in categories_list.index.tolist()[:-1]:
                if len(categories_list[enc_value]) == 1:
                    if type(categories_list[enc_value][0]) == str:
                        col_sql += f"WHEN {DBMSUtils.get_delimited_col(defs.DBMS, feature)} = '{categories_list[enc_value][0]}' THEN {enc_value} "
                    else:
                        col_sql += f"WHEN {DBMSUtils.get_delimited_col(defs.DBMS, feature)} = {categories_list[enc_value][0]} THEN {enc_value} "
                else:
                    in_str = ",".join(
                        [
                            f"'{c}'" if type(c) == str else f"{c}"
                            for c in categories_list[enc_value]
                        ]
                    )
                    col_sql += f"WHEN {DBMSUtils.get_delimited_col(defs.DBMS, feature)} in ({in_str}) THEN {enc_value} "
            col_sql += f"ELSE {categories_list.index.tolist()[-1]} END "
            return col_sql, op, thr

        else:
            col_sql = "CASE "
            for enc_value in categories_list.index.tolist()[:-1]:
                intervals = []
                for c in categories_list[enc_value]:
                    intervals.extend(self.con_c_cat_mapping[c])
                merged_intervals = merge_intervals(intervals)
                condition_str = " OR ".join(
                    [
                        f"{DBMSUtils.get_delimited_col(defs.DBMS, feature)} >= {interval[0]}"
                        + " AND "
                        + f"{DBMSUtils.get_delimited_col(defs.DBMS, feature)} < {interval[1]}"
                        for interval in merged_intervals
                    ]
                )
                col_sql += f"WHEN {condition_str} THEN {enc_value} "
            col_sql += f"ELSE {categories_list.index.tolist()[-1]} END "
            return col_sql, op, thr

    def get_fusion_primitive_type(self, feature, thr):
        if self.con_c_cat_mapping is not None:
            PrimitiveType.OR
        else:
            mapping = self.mapping[feature]
            in_length = 0
            not_in_length = 0
            for enc_value in mapping:
                if enc_value <= thr:
                    in_length += 1
                else:
                    not_in_length += 1
            if in_length == 1:
                return PrimitiveType.EQUAL
            elif not_in_length == 1:
                return PrimitiveType.INEQUAL
            return PrimitiveType.IN

    def get_fusion_primitive_length(self, feature, thr):
        mapping = self.mapping[feature]

        if self.con_c_cat_mapping:
            intervals = []
            for enc_value in mapping:
                if enc_value <= thr:
                    intervals.extend(self.con_c_cat_mapping[enc_value])
            merged_intervals = merge_intervals(intervals)
            return len(merged_intervals)
        else:
            in_length = 0
            not_in_length = 0
            for enc_value in mapping:
                if enc_value <= thr:
                    in_length += 1
                else:
                    not_in_length += 1
            if in_length == 1:
                return 1
            elif not_in_length == 1:
                return 1
            return in_length

    def _get_op_cost(self, feature):
        mapping = self.mapping[feature]
        mapping = mapping[~mapping.index.isnull()]
        mapping = mapping[[idx for idx in mapping.index if idx != "NaN"]]
        categories_list = mapping.groupby(mapping).apply(lambda x: x.index.tolist())
        sorted_categories_list = categories_list.apply(lambda x: len(x)).sort_values(
            ascending=True
        )
        categories_list = Series(
            categories_list[sorted_categories_list.index],
            index=sorted_categories_list.index,
        )
        value_counts = self.value_counts[feature]

        if self.con_c_cat_mapping is None:

            # def calc_cost_e(x):
            #     cost = 0
            #     for in_list in categories_list:
            #         cost += PrimitiveCost.IN(len(in_list))
            #         if x in in_list:
            #             break
            #     return cost

            # data_primitive_costs = sample_data["_".join(feature.split("_")[:-1])].apply(
            #     lambda x: calc_cost_e(x)
            # )
            # return sum(data_primitive_costs)
            cost = 0
            before_len = 0
            n = 0
            for enc_value in categories_list.index.tolist()[:-1]:
                list_len = len(categories_list[enc_value])
                if list_len == 1:
                    if isinstance(categories_list[enc_value][0], tuple):
                        cost += PrimitiveCost.OR.value * value_counts[enc_value]
                    else:
                        cost += PrimitiveCost.EQUAL.value * value_counts[enc_value]
                else:
                    if isinstance(categories_list[enc_value][0], tuple):
                        cost += PrimitiveCost.OR.value * (before_len + list_len) * value_counts[enc_value]
                    else:
                        cost += (
                            (PrimitiveCost.IN(before_len + list_len) + PrimitiveCost.IN(0) * n) \
                            * value_counts[enc_value]
                        )
                    n = n + 1
                    before_len += list_len
            
            if len(categories_list) > 1:
                last_enc_value = categories_list.index.tolist()[-1]
                if list_len == 1:
                    if isinstance(categories_list[enc_value][0], tuple):
                        cost += PrimitiveCost.OR.value * value_counts[last_enc_value]
                    else:
                        cost += PrimitiveCost.EQUAL.value * value_counts[last_enc_value]
                else:
                    if isinstance(categories_list[enc_value][0], tuple):
                        cost += PrimitiveCost.OR.value * (before_len) * value_counts[last_enc_value]
                    else:
                        cost += (
                            (PrimitiveCost.IN(before_len) + PrimitiveCost.IN(0) * (n-1)) \
                            * value_counts[last_enc_value]
                        )
                
            return cost
        else:

            # def calc_cost_c(x):
            #     cost = 0
            #     contain = False
            #     for _, in_list in categories_list:
            #         intervals = []
            #         for in_value in in_list:
            #             intervals.extend(self.con_c_cat_mapping[in_value])
            #         merged_intervals = merge_intervals(intervals)
            #         for interval in merged_intervals:
            #             cost += PrimitiveCost.OR.value
            #             if x >= interval[0] and x < interval[1]:
            #                 contain = True
            #                 break
            #         if contain:
            #             break

            # data_primitive_costs = sample_data[feature].apply(lambda x: calc_cost_c(x))
            # return sum(data_primitive_costs)
            cost = 0
            before_len = 0
            for enc_value in categories_list.index.tolist()[:-1]:
                in_list = categories_list[enc_value]
                intervals = []
                for in_value in in_list:
                    intervals.extend(self.con_c_cat_mapping[in_value])
                merged_intervals = merge_intervals(intervals)
                inequal_len = len(merged_intervals)
                cost += (
                    PrimitiveCost.OR.value
                    * (before_len + inequal_len)
                    * value_counts[enc_value]
                )
                before_len += inequal_len
            if len(categories_list) > 1:
                last_enc_value = categories_list.index.tolist()[-1]
                cost += (
                    PrimitiveCost.OR.value
                    * (before_len)
                    * value_counts[last_enc_value]
                )
            return cost

    def get_push_primitive_type(self, feature, thr):
        if self.con_c_cat_mapping:
            return PrimitiveType.OR
        else:
            return PrimitiveType.IN

    def get_push_primitive_length(self, feature, thr):
        mapping = self.mapping[feature]
        mapping = mapping[~mapping.index.isnull()]
        mapping = mapping[[idx for idx in mapping.index if idx != "NaN"]]
        categories_list = mapping.groupby(mapping).apply(lambda x: x.index.tolist())
        sorted_categories_list = categories_list.apply(lambda x: len(x)).sort_values(
            ascending=True
        )
        categories_list = Series(
            categories_list[sorted_categories_list.index],
            index=sorted_categories_list.index,
        )
        if self.con_c_cat_mapping:
            or_length = 0
            for enc_value in categories_list.index.tolist()[:-1]:
                intervals = []
                for c in categories_list[enc_value]:
                    intervals.extend(self.con_c_cat_mapping[c])
                merged_intervals = merge_intervals(intervals)
                or_length += len(merged_intervals)
            if defs.PUSH_USE_AVERAGE:
                return or_length / 2
            else:
                return or_length
        else:
            in_length = 0
            for enc_value in categories_list.index.tolist()[:-1]:
                in_length += len(categories_list[enc_value])
            return in_length
        # TODO: need use more accurate length

    def _get_join_cost(self, feature, graph, data_rows):
        if graph.model.model_name in (
            ModelName.DECISIONTREECLASSIFIER,
            ModelName.RANDOMFORESTCLASSIFIER,
            ModelName.DECISIONTREEREGRESSOR,
            ModelName.RANDOMFORESTREGRESSOR,
        ):
            tree_costs = graph.model.get_tree_costs(feature, self)
            total_tree_cost = sum(
                [tree_cost.calculate_no_fusion_cost() for tree_cost in tree_costs]
            )
        else:
            total_tree_cost = 0
        join_cost = calc_join_cost_by_train_data(
            data_rows, len(self.mapping), len(self.mapping.columns)
        )
        return total_tree_cost + join_cost

    def _get_join_cost_without_tree(self, feature, graph, data_rows):
        join_cost = calc_join_cost_by_train_data(
            data_rows, len(self.mapping), len(self.mapping.columns)
        )
        return join_cost


class CON_A_CON(SQLOperator):

    def __init__(self, op_name: OperatorName):
        super().__init__(op_name)
        self.input_data_type = DataType.CON
        self.output_data_type = DataType.CON
        self.calculation_type = CalculationType.ARITHMETIC
        self.op_type = OperatorType[self._get_op_type()]

        self.equation: Eq
        self.symbols: dict = {}
        self.parameter_values: list = []

        self.is_arithmetic_op = True

    def apply(self, first_op: Operator):
        if first_op.op_type == OperatorType.CON_C_CAT:
            merged_op = CON_C_CAT_Merged_OP(first_op)
            merged_op.bin_edges = first_op.bin_edges
            for idx, category_list in enumerate(first_op.categories):
                sub_equation = self.equation.rhs.subs(
                    {
                        self.symbols[sym_name]: self.parameter_values[idx][sym_name]
                        for sym_name in self.parameter_values[idx]
                    }
                )
                f = lambdify(self.symbols["x"], sub_equation, "numpy")
                merged_op.categories[idx] = f(category_list)
            return merged_op

        elif first_op.op_type == OperatorType.CAT_C_CAT:
            merged_op = CAT_C_CAT_Merged_OP(first_op)
            for idx, mapping in enumerate(first_op.mappings):
                sub_equation = self.equation.rhs.subs(
                    {
                        self.symbols[sym_name]: self.parameter_values[idx][sym_name]
                        for sym_name in self.parameter_values[idx]
                    }
                )
                f = lambdify(self.symbols["x"], sub_equation, "numpy")
                merged_op.mappings.append(Series(f(mapping), index=mapping.index))
            return merged_op

        elif first_op.op_type == OperatorType.EXPAND:
            merged_op = EXPAND_Merged_OP(first_op)
            merged_op.mapping = first_op.mapping
            for idx, col in enumerate(first_op.mapping.columns):
                sub_equation = self.equation.rhs.subs(
                    {
                        self.symbols[sym_name]: self.parameter_values[idx][sym_name]
                        for sym_name in self.parameter_values[idx]
                    }
                )
                f = lambdify(self.symbols["x"], sub_equation, "numpy")
                merged_op.mapping[col] = f(merged_op.mapping[col])
            return merged_op

        else:
            return None

    def simply(self, second_op: Operator):
        if second_op.op_type == OperatorType.CON_A_CON:
            merged_op = CON_A_CON_Merged_OP(second_op)
            merged_op.equation = Eq(
                second_op.equation.lhs,
                second_op.equation.rhs.subs(
                    {second_op.symbols["x"]: self.equation.rhs}
                ),
            )
            merged_op.symbols = {**second_op.symbols, **self.symbols}
            merged_op.parameter_values = [
                {**second_op.parameter_values[idx], **self.parameter_values[idx]}
                for idx in range(len(self.parameter_values))
            ]
            return merged_op

        elif second_op.op_type == OperatorType.CON_C_CAT:
            merged_op = CON_C_CAT_Merged_OP(second_op)
            merged_op.categories = second_op.categories
            reversed_equation = solve(self.equation, self.symbols["x"])[0]
            for idx, bin_edge in enumerate(second_op.bin_edges):
                sub_equation = reversed_equation.subs(
                    {
                        self.symbols[sym_name]: self.parameter_values[idx][sym_name]
                        for sym_name in self.parameter_values[idx]
                    }
                )
                f = lambdify(self.symbols["y"], sub_equation, "numpy")
                merged_op.bin_edges.append(f(bin_edge))
            return merged_op

        else:
            return None

    def get_sql(self, dbms: str):
        sqls = []
        for idx in range(len(self.features)):
            sub_equation = self.equation.rhs.subs(
                {
                    self.symbols[sym_name]: self.parameter_values[idx][sym_name]
                    for sym_name in self.parameter_values[idx]
                }
            )
            feature_sql = (
                str(sub_equation).replace(
                    "x", DBMSUtils.get_delimited_col(dbms, self.features[idx])
                )
                + f" AS {DBMSUtils.get_delimited_col(dbms, self.features[idx])}"
            )
            sqls.append(feature_sql)

        return ",".join(sqls)

    def modify_leaf(self, feature, op, thr, modified_feature):
        reversed_equation = solve(self.equation, self.symbols["x"])[0]
        idx = self.features_out.index(feature)
        sub_equation = reversed_equation.subs(
            {
                self.symbols[sym_name]: self.parameter_values[idx][sym_name]
                for sym_name in self.parameter_values[idx]
            }
        )
        thr = sub_equation.subs(self.symbols["y"], thr)
        return DBMSUtils.get_delimited_col(defs.DBMS, feature), op, thr

    def modify_leaf_p(self, feature, op, thr):
        idx = self.features_out.index(feature)
        equation = self.equation.rhs.subs(
            {
                self.symbols[sym_name]: self.parameter_values[idx][sym_name]
                for sym_name in self.parameter_values[idx]
            }
        )
        feature_sql = (
            str(equation).replace(
                "x", f"{DBMSUtils.get_delimited_col(defs.DBMS, feature)}"
            )
            + " "
        )
        return feature_sql, op, thr

    def get_fusion_primitive_type(self, feature, feature_value):
        return PrimitiveType.LE_EQ

    def get_fusion_primitive_length(self, feature, feature_value):
        return 1

    def _get_op_cost(self, feature):
        return 1

    def get_push_primitive_type(self, feature, thr):
        return PrimitiveType.LE_EQ

    def get_push_primitive_length(self, feature, thr):
        return 1


class CON_S_CON(SQLOperator):

    def __init__(self, op_name: OperatorName):
        super().__init__(op_name)
        self.input_data_type = DataType.CON
        self.output_data_type = DataType.CON
        self.calculation_type = CalculationType.ARITHMETIC
        self.op_type = OperatorType[self._get_op_type()]

        self.mappings = []
        self.is_contain_ca_op = True
        self.is_arithmetic_op = True

    def get_sql(self, dbms: str):
        # TODO: 分段函数
        sqls = []
        for idx in range(len(self.features)):
            mapping = self.mappings[idx]
            if len(mapping) == 1 and mapping.index[0] == (-float("inf"), float("inf")):
                sub_equation = mapping.iloc[0].rhs.subs(
                    {
                        self.symbols[sym_name]: self.parameter_values[idx][sym_name][0]
                        for sym_name in self.parameter_values[idx]
                    }
                )
                feature_sql = (
                    str(sub_equation).replace(
                        "x", DBMSUtils.get_delimited_col(dbms, self.features[idx])
                    )
                    + f" AS {DBMSUtils.get_delimited_col(dbms, self.features[idx])}"
                )
                sqls.append(feature_sql)
            else:
                feature_sql = "CASE "
                for eq_idx, (interval, equation) in enumerate(mapping.items()):
                    sub_equation = equation.subs(
                        {
                            self.symbols[sym_name]: self.parameter_values[idx][sym_name][eq_idx]
                            for sym_name in self.parameter_values[idx]
                        }
                    )
                    if interval[0] == -float("inf"):
                        feature_sql += f"WHEN {DBMSUtils.get_delimited_col(dbms, self.features[idx])} <= {interval[1]}"
                        + " THEN "
                        + str(sub_equation).replace("x", DBMSUtils.get_delimited_col(dbms, self.features[idx]))
                    elif interval[1] == float("inf"):
                        feature_sql += f"WHEN {DBMSUtils.get_delimited_col(dbms, self.features[idx])} > {interval[0]}"
                        + " THEN "
                        + str(sub_equation).replace("x", DBMSUtils.get_delimited_col(dbms, self.features[idx]))
                    else:
                        feature_sql += f"WHEN {DBMSUtils.get_delimited_col(dbms, self.features[idx])} > {interval[0]}"
                        + " AND "
                        + f"{DBMSUtils.get_delimited_col(dbms, self.features[idx])} <= {interval[1]}"
                        + " THEN "
                        + str(sub_equation).replace("x", DBMSUtils.get_delimited_col(dbms, self.features[idx]))
                    feature_sql += " END AS " + DBMSUtils.get_delimited_col(
                        dbms, self.features[idx]
                    )
            sqls.append(feature_sql)

        return ",".join(sqls)

    def _get_op_cost(self, feature):
        feature_idx = self.features_out.index(feature)
        return len(self.mappings[feature_idx]) * (PrimitiveType.OR + 1)

    def get_fusion_primitive_type(self, feature, feature_value):
        pass

    def get_fusion_primitive_length(self, feature, feature_value):
        pass

    def get_push_primitive_type(self, feature, thr):
        pass

    def get_push_primitive_length(self, feature, thr):
        pass

    def apply(self, first_op: Operator):
        pass

    def simply(self, second_op: Operator):
        pass

    def modify_leaf(self, feature, op, thr, modified_feature):
        pass

    def modify_leaf_p(self, feature, op, thr):
        pass


class CON_C_CAT(SQLOperator):

    def __init__(self, op_name: OperatorName):
        super().__init__(op_name)
        self.input_data_type = DataType.CON
        self.output_data_type = DataType.CAT
        self.calculation_type = CalculationType.COMPARISON
        self.op_type = OperatorType[self._get_op_type()]

        self.bin_edges: list = []
        self.n_bins: list = []
        self.categories: list = []
        self.bin_distribution: dict = {}
        self.mappings: list[Series] = []
        self.is_contain_ca_op = True
        self.is_constant_output_op = True

        self.inequations = {}
        self.inequations_mappings = {}
        self.is_inequality_judgment_op = True

    def update_intervals_by_inequalities(self):
        for feature_idx, feature in enumerate(self.features):
            new_intervals = []
            for eq_idx, equation in enumerate(self.inequations[feature]):
                new_intervals.append(
                    (equation.args[0].args[1], equation.args[1].args[1])
                )
                if eq_idx == 0:
                    self.bin_edges[feature_idx][eq_idx] = equation.args[0].args[1]
                self.bin_edges[feature_idx][eq_idx + 1] = equation.args[1].args[1]
                self.categories[feature_idx][eq_idx] = self.mappings[feature_idx].iloc[
                    eq_idx
                ]
            self.mappings[feature_idx].index = new_intervals

    def __judge_feature_value(self, xs, feature_idx):
        res_xs = []
        for x in xs:
            for i in range(self.n_bins[feature_idx]):
                if (
                    x >= self.bin_edges[feature_idx][i]
                    and x <= self.bin_edges[feature_idx][i + 1]
                ):
                    res_xs.append(self.categories[feature_idx][i])
                    break

        return np.array(res_xs)

    def apply(self, first_op: Operator):
        if first_op.op_type == OperatorType.EXPAND:
            merged_op = EXPAND_Merged_OP(first_op)
            merged_op.mapping = first_op.mapping
            for idx, column in enumerate(merged_op.mapping.columns):
                merged_op.mapping[column] = self.__judge_feature_value(
                    merged_op.mapping[column], idx
                )

            return merged_op

        elif first_op.op_type == OperatorType.CON_C_CAT:
            merged_op = CON_C_CAT_Merged_OP(first_op)
            merged_op.bin_edges = first_op.bin_edges
            for idx, category_list in enumerate(first_op.categories):
                merged_op.categories[idx] = self.__judge_feature_value(
                    category_list, idx
                )

            return merged_op

        elif first_op.op_type == OperatorType.CAT_C_CAT:
            merged_op = CAT_C_CAT_Merged_OP(first_op)
            for idx, mapping in enumerate(first_op.mappings):
                merged_op.mappings.append(
                    Series(
                        self.__judge_feature_value(mapping.values, idx),
                        index=mapping.index,
                    )
                )
            return merged_op
        else:
            return None

    def simply(self, second_op: Operator):
        if second_op.op_type == OperatorType.EXPAND:
            merged_op = EXPAND_Merged_OP(second_op)
            merged_op.mapping = second_op.mapping
            intervals = [
                (self.bin_edges[0][i], self.bin_edges[0][i + 1])
                for i in range(self.n_bins[0])
            ]
            interval_cat_map = Series(self.categories[0], index=intervals)
            cat_interval_map = interval_cat_map.groupby(interval_cat_map).apply(
                lambda x: x.index.tolist()
            )
            merged_op.con_c_cat_mapping = cat_interval_map
            merged_op.value_counts = second_op.value_counts

            return merged_op

        else:
            return None

    def get_sql(self, dbms: str):
        sqls = []

        for idx in range(len(self.features)):
            bin_distribution = self.bin_distribution[self.features[idx]]
            pos_2_bin = np.argsort(-bin_distribution)
            feature_sql = "CASE "
            if self.n_bins[idx] > 1:
                for i in range(self.n_bins[idx] - 1):
                    if defs.ORDER_WHEN:
                        feature_sql += (
                            f"WHEN {DBMSUtils.get_delimited_col(dbms, self.features[idx])} >= {self.bin_edges[idx][pos_2_bin[i]]}"
                            + " AND "
                            + f"{DBMSUtils.get_delimited_col(dbms, self.features[idx])} < {self.bin_edges[idx][pos_2_bin[i]+1]} THEN {self.categories[idx][pos_2_bin[i]]} "
                        )
                    else:
                        feature_sql += (
                            f"WHEN {DBMSUtils.get_delimited_col(dbms, self.features[idx])} >= {self.bin_edges[idx][i]}"
                            + " AND "
                            + f"{DBMSUtils.get_delimited_col(dbms, self.features[idx])} < {self.bin_edges[idx][i+1]} THEN {self.categories[idx][i]} "
                        )
                if defs.ORDER_WHEN:
                    feature_sql += f"ELSE {self.categories[idx][pos_2_bin[self.n_bins[idx] - 1]]} END AS {DBMSUtils.get_delimited_col(dbms, self.features[idx])} "
                else:
                    feature_sql += f"ELSE {self.categories[idx][-1]} END AS {DBMSUtils.get_delimited_col(dbms, self.features[idx])} "

            elif self.n_bins[idx] == 1:
                feature_sql += (
                    f"WHEN {DBMSUtils.get_delimited_col(dbms, self.features[idx])} >= {self.bin_edges[idx][0]}"
                    + " AND "
                    + f"{DBMSUtils.get_delimited_col(dbms, self.features[idx])} < {self.bin_edges[idx][1]} THEN {self.categories[idx][0]} "
                )
                feature_sql += (
                    f"END AS {DBMSUtils.get_delimited_col(dbms, self.features[idx])} "
                )
            sqls.append(feature_sql)

        return ",".join(sqls)

    def modify_leaf(self, feature, op, thr, modified_feature):
        bin_edge = self.bin_edges[self.features_out.index(feature)]
        categoiry_list = self.categories[self.features_out.index(feature)]
        bin_distribution = self.bin_distribution[feature]
        if self.op_name != OperatorName.KBINSDISCRETIZER:
            intervals = []
            interval_distributions = []
            for i, category in enumerate(categoiry_list):
                if category <= thr:
                    intervals.append((bin_edge[i], bin_edge[i + 1]))
                    interval_distributions.append(bin_distribution[i])

            if defs.ORDER_WHEN:
                merged_intervals, merged_distributions = merge_intervals(
                    intervals, interval_distributions
                )
                pos_2_bin = np.argsort(-np.array(merged_distributions))
            else:
                merged_intervals = merge_intervals(intervals)

            condition_sqls = []
            for i in range(len(merged_intervals)):
                if defs.ORDER_WHEN:
                    condition_sql = (
                        f"{DBMSUtils.get_delimited_col(defs.DBMS, feature)} >= {merged_intervals[pos_2_bin[i]][0]}"
                        + " AND "
                        + f"{DBMSUtils.get_delimited_col(defs.DBMS, feature)} < {merged_intervals[pos_2_bin[i]][1]}"
                    )
                else:
                    condition_sql = (
                        f"{DBMSUtils.get_delimited_col(defs.DBMS, feature)} >= {merged_intervals[i][0]}"
                        + " AND "
                        + f"{DBMSUtils.get_delimited_col(defs.DBMS, feature)} < {merged_intervals[i][1]}"
                    )
                condition_sqls.append(condition_sql)

            return " OR ".join(condition_sqls), "", ""

        else:
            if op == "in":
                intervals = []
                for in_value in [float(x) for x in thr[1:-1].split(",")]:
                    if in_value in categoiry_list:
                        i = categoiry_list.tolist().index(in_value)
                        intervals.append((bin_edge[i], bin_edge[i + 1]))
                merged_intervals = merge_intervals(intervals)
                condition_sqls = []
                for i in range(len(merged_intervals)):
                    condition_sql = (
                        f"{DBMSUtils.get_delimited_col(defs.DBMS, feature)} >= {merged_intervals[i][0]}"
                        + " AND "
                        + f"{DBMSUtils.get_delimited_col(defs.DBMS, feature)} < {merged_intervals[i][1]}"
                    )
                    condition_sqls.append(condition_sql)

                return " OR ".join(condition_sqls), "", ""

            elif op == "<=":
                for i, category in enumerate(categoiry_list):
                    if category > thr:
                        return (
                            DBMSUtils.get_delimited_col(defs.DBMS, feature),
                            op,
                            bin_edge[i],
                        )
            
            elif op == "<>":
                intervals = []
                for i, category in enumerate(categoiry_list):
                    if category != thr:
                        intervals.append((bin_edge[i], bin_edge[i+1]))
                merged_intervals = merge_intervals(intervals)
                condition_sqls = []
                for i in range(len(merged_intervals)):
                    condition_sql = (
                        f"{DBMSUtils.get_delimited_col(defs.DBMS, feature)} >= {merged_intervals[i][0]}"
                        + " AND "
                        + f"{DBMSUtils.get_delimited_col(defs.DBMS, feature)} < {merged_intervals[i][1]}"
                    )
                    condition_sqls.append(condition_sql)

                return " OR ".join(condition_sqls), "", ""

            else:
                pass

    def modify_leaf_p(self, feature, op, thr):
        idx = self.features_out.index(feature)
        # feature_sql = "CASE "
        # for i in range(self.n_bins[idx] - 1):
        #     feature_sql += (
        #         f"WHEN {DBMSUtils.get_delimited_col(defs.DBMS, self.features[idx])} >= {self.bin_edges[idx][i]}"
        #         + " AND "
        #         + f"{DBMSUtils.get_delimited_col(defs.DBMS, self.features[idx])} < {self.bin_edges[idx][i+1]} THEN {self.categories[idx][i]} "
        #     )
        # feature_sql += f"ELSE {self.categories[idx][-1]} END "
        if defs.ORDER_WHEN:
            bin_distribution = self.bin_distribution[self.features[idx]]
            pos_2_bin = np.argsort(-bin_distribution)
        feature_sql = "CASE "
        for i in range(self.n_bins[idx] - 1):
            if defs.ORDER_WHEN:
                feature_sql += (
                    f"WHEN {DBMSUtils.get_delimited_col(defs.DBMS, self.features[idx])} >= {self.bin_edges[idx][pos_2_bin[i]]}"
                    + " AND "
                    + f"{DBMSUtils.get_delimited_col(defs.DBMS, self.features[idx])} < {self.bin_edges[idx][pos_2_bin[i]+1]} THEN {self.categories[idx][pos_2_bin[i]]} "
                )
            else:
                feature_sql += (
                    f"WHEN {DBMSUtils.get_delimited_col(defs.DBMS, self.features[idx])} >= {self.bin_edges[idx][i]}"
                    + " AND "
                    + f"{DBMSUtils.get_delimited_col(defs.DBMS, self.features[idx])} < {self.bin_edges[idx][i+1]} THEN {self.categories[idx][i]} "
                )
        if defs.ORDER_WHEN:
            feature_sql += (
                f"ELSE {self.categories[idx][pos_2_bin[self.n_bins[idx] - 1]]} END "
            )
        else:
            feature_sql += f"ELSE {self.categories[idx][-1]} END "

        return feature_sql, op, thr

    def get_fusion_primitive_type(self, feature, thr):
        if self.op_name == OperatorName.KBINSDISCRETIZER:
            return PrimitiveType.LE
        else:
            return PrimitiveType.OR

    def get_fusion_primitive_length(self, feature, thr):
        if self.op_name == OperatorName.KBINSDISCRETIZER:
            return 1
        else:
            bin_edge = self.bin_edges[self.features_out.index(feature)]
            categoiry_list = self.categories[self.features_out.index(feature)]
            intervals = []
            for i, category in enumerate(categoiry_list):
                if category <= thr:
                    intervals.append((bin_edge[i], bin_edge[i + 1]))
            merged_intervals = merge_intervals(intervals)
            return len(merged_intervals) / 2

    def _get_op_cost(self, feature):
        idx = self.features_out.index(feature)
        bin_distribution = self.bin_distribution[self.features[idx]]
        if defs.ORDER_WHEN:
            pos_2_bin = np.argsort(-bin_distribution)
            bin_2_pos = {bin: pos for pos, bin in enumerate(pos_2_bin)}
            return PrimitiveCost.OR.value * sum(
                [(bin_2_pos[i] + 1) * num for i, num in enumerate(bin_distribution)]
            )
        else:
            return PrimitiveCost.OR.value * sum(
                [(i + 1) * num for i, num in enumerate(bin_distribution)]
            )

        # def calc_cost(x):
        #     cost = 0
        #     for i in range(len(self.categories[idx])):
        #         cost += PrimitiveCost.OR.value
        #         if x >= self.bin_edges[idx][i] and x < self.bin_edges[idx][i + 1]:
        #             break
        #     return cost

        # data_primitive_costs = sample_data[feature].apply(lambda x: calc_cost(x))
        # return sum(data_primitive_costs)

    def get_push_primitive_type(self, feature, thr):
        return PrimitiveType.OR

    def get_push_primitive_length(self, feature, thr):
        if defs.PUSH_USE_AVERAGE:
            return len(self.categories) / 2
        else:
            return len(self.categories)


class CON_C_CAT_Merged_OP(CON_C_CAT):

    def __init__(self, op: Type[CON_C_CAT]):
        super().__init__(OperatorName.CON_C_CAT_Merged_OP)
        self.n_bins = op.n_bins.copy()
        self.categories = op.categories.copy()
        self.features = op.features
        self.features_out = op.features_out
        self.bin_distribution = op.bin_distribution.copy()

    def _extract(self, fitted_transform) -> None:
        pass


class EXPAND_Merged_OP(EXPAND):

    def __init__(self, op: Type[EXPAND]):
        super().__init__(OperatorName.EXPAND_Merged_OP)
        self.features = op.features
        self.features_out = op.features_out

    def _extract(self, fitted_transform) -> None:
        pass


class CON_A_CON_Merged_OP(CON_A_CON):

    def __init__(self, op: Type[CON_A_CON]):
        super().__init__(OperatorName.CON_A_CON_Merged_OP)
        self.features = op.features
        self.features_out = op.features_out

    def _extract(self, fitted_transform) -> None:
        pass


class CAT_C_CAT_Merged_OP(CAT_C_CAT):

    def __init__(self, op: Type[CAT_C_CAT]):
        super().__init__(OperatorName.CAT_C_CAT_Merged_OP)
        self.features = op.features
        self.features_out = op.features_out
        self.value_counts = op.value_counts

    def _extract(self, fitted_transform) -> None:
        pass
