import numpy as np
from sympy import sympify, LessThan, Eq, And, Symbol, Ne
from sklearn.tree import DecisionTreeRegressor  
from craftsman.utility.dbms_utils import DBMSUtils
from craftsman.model.base_model import TreeModel
from craftsman.base.operator import Operator
from craftsman.base.defs import ModelName
import craftsman.base.defs as defs
from craftsman.cost_model.cost import TreeCost


class DecisionTreeRegressorSQLModel(TreeModel):
    """
    This class implements the SQL wrapper for a Sklearn's Decision Tree Model (DTM).
    """

    def __init__(self, trained_model: DecisionTreeRegressor):
        super().__init__()
        self.model_name = ModelName.DECISIONTREEREGRESSOR
        self.trained_model = trained_model
        # get for each node, left, right child nodes, thresholds and features
        self.left = self.trained_model.tree_.children_left  # left child for each node
        self.right = self.trained_model.tree_.children_right  # right child for each node
        self.thresholds = self.trained_model.tree_.threshold.tolist()  # test threshold for each node
        self.ops = ['<='] * len(self.trained_model.tree_.feature)
        if hasattr(self.trained_model, 'feature_names_in_'):
            self.input_features = self.trained_model.feature_names_in_
            self.features = [DBMSUtils.get_delimited_col(defs.DBMS, self.input_features[i]) for i in self.trained_model.tree_.feature]
            self.features_origin = [self.input_features[i] for i in self.trained_model.tree_.feature]
            
            if defs.AUTO_RULE_GEN:
                # abstract the tree model to a operator consisted of inequations
                for feature in self.input_features:
                    self.inequations[feature] = []
                    self.tree_node_mappings[feature] = []
                for idx, thr in enumerate(self.thresholds):
                    if self.left[idx] == -1 and self.right[idx] == -1:
                        continue
                    self.inequations[self.features_origin[idx]].append(sympify(f'x {self.ops[idx]} {self.thresholds[idx]}'))
                    self.tree_node_mappings[self.features_origin[idx]].append(idx)

    def set_features(self, feature_names_in):
        self.trained_model.feature_names_in_ = feature_names_in
        self.input_features = self.trained_model.feature_names_in_
        self.features = [DBMSUtils.get_delimited_col(defs.DBMS, self.input_features[i]) for i in self.trained_model.tree_.feature]
        self.features_origin = [self.input_features[i] for i in self.trained_model.tree_.feature]
        
        if defs.AUTO_RULE_GEN:
            # abstract the tree model to a operator consisted of inequations
            for feature in self.input_features:
                self.inequations[feature] = []
                self.tree_node_mappings[feature] = []
            for idx, thr in enumerate(self.thresholds):
                if self.left[idx] == -1 and self.right[idx] == -1:
                    continue
                self.inequations[self.features_origin[idx]].append(sympify(f'x {self.ops[idx]} {self.thresholds[idx]}'))
                self.tree_node_mappings[self.features_origin[idx]].append(idx)

    def get_case_sql(self, dbms: str) -> str:

        def visit_tree(node):
            # leaf node
            if self.left[node] == -1 and self.right[node] == -1:
                return " {} ".format(round(self.trained_model.tree_.value[node][0][0], 5))

            # internal node
            op = self.ops[node]
            thr = self.thresholds[node]

            sql_dtm_rule = f" CASE WHEN {self.features[node]} {op} {thr} THEN"

            # check if current node has a left child
            if self.left[node] != -1:
                sql_dtm_rule += visit_tree(self.left[node])

            sql_dtm_rule += "ELSE"

            # check if current node has a right child
            if self.right[node] != -1:
                sql_dtm_rule += visit_tree(self.right[node])

            sql_dtm_rule += "END "

            return sql_dtm_rule

        # start tree visit from the root node
        root = 0
        sql_dtm_rules = visit_tree(root)

        return sql_dtm_rules
    
    def get_tree_costs(self, feature, operator):
        tree_cost = TreeCost(feature, operator, self)
        tree_cost.analyze_path_fusion_cost(feature, operator, self)
        return [tree_cost]
    
    def get_tree_costs_p(self, feature, operator):
        tree_cost = TreeCost(feature, operator, self)
        tree_cost.analyze_path_push_cost(feature, operator, self)
        return [tree_cost]
    
    def get_tree_costs_static(self):
        tree_cost = TreeCost(model=self)
        tree_cost.analyze_path_cost(self)
        return [tree_cost]

    def modify_model(self, feature: str, sql_operator: Operator):
        for idx, node in enumerate(self.trained_model.tree_.feature):
            # if self.features[idx] == feature:
            if self.input_features[node] == feature or \
                (feature in self.input_features[node] and sql_operator.op_type != defs.OperatorType.EXPAND):
                self.features[idx], self.ops[idx], self.thresholds[idx] = sql_operator.modify_leaf(
                    feature, self.ops[idx], self.thresholds[idx], self.features[idx]
                    # self.features_origin[idx], self.ops[idx], self.thresholds[idx], self.features[idx]
                )
                
    def modify_model_p(self, feature: str, sql_operator: Operator):
        for idx, node in enumerate(self.trained_model.tree_.feature):
            if self.input_features[node] == feature:
                self.features[idx], self.ops[idx], self.thresholds[idx] = sql_operator.modify_leaf_p(
                    self.features_origin[idx], self.ops[idx], self.thresholds[idx]
                )

    def query(self, input_table: str, dbms: str) -> str:
        # self.update_tree_by_inequalities()
        query = "SELECT {} AS Score".format(self.get_case_sql(dbms))
        query += " FROM {}".format(input_table)

        return query
    
    def update_tree_by_inequalities(self):
        for feature in self.input_features:
            for tree_node_idx, inequality in zip(self.tree_node_mappings[feature], self.inequations[feature]):
                if isinstance(inequality, LessThan):
                    self.features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature) 
                    self.ops[tree_node_idx] = '<='
                    self.thresholds[tree_node_idx] = float(inequality.rhs)
                elif isinstance(inequality, list) and len(inequality) > 1:
                    if isinstance(inequality[0], Eq): 
                        equal_values = []
                        for equality in inequality:
                            if isinstance(equality.args[1], Symbol):
                                equal_values.append(f"'{equality.args[1].name}'")
                            else:
                                equal_values.append(f"{equality.args[1]}")
                        if feature.split('_')[-1].isnumeric():
                            self.features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature.rsplit('_', 1)[0])
                        else:
                            self.features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature) 
                        self.ops[tree_node_idx] = 'in'
                        self.thresholds[tree_node_idx] = f"({','.join(equal_values)})"  
                    elif isinstance(inequality[0], And):
                        interval_strs = []
                        for and_expr in inequality:
                            lower_bound = and_expr.args[0].args[1]
                            upper_bound = and_expr.args[1].args[1]
                            interval_strs.append(f"{DBMSUtils.get_delimited_col(defs.DBMS, feature)} >= {lower_bound}" 
                                                 + " AND " + 
                                                 f"{DBMSUtils.get_delimited_col(defs.DBMS, feature)} < {upper_bound}")
                        self.features[tree_node_idx] = " OR ".join(interval_strs)
                        self.ops[tree_node_idx] = ''
                        self.thresholds[tree_node_idx] = ''
                elif isinstance(inequality, list) and len(inequality) == 1:
                    if isinstance(inequality[0], Eq):
                        if feature not in defs.PIPELINE_FEATURES_IN and feature.split('_')[-1].isnumeric():
                            self.features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature.rsplit('_', 1)[0])
                        else:
                            self.features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature) 
                        self.ops[tree_node_idx] = '='
                        if isinstance(inequality[0].args[1], Symbol):
                            self.thresholds[tree_node_idx] = f"'{inequality[0].args[1].name}'"
                        else:
                            self.thresholds[tree_node_idx] = inequality[0].args[1]
                    elif isinstance(inequality[0], Ne):
                        if feature not in defs.PIPELINE_FEATURES_IN and feature.split('_')[-1].isnumeric():
                            self.features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature.rsplit('_', 1)[0])
                        else:
                            self.features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature) 
                        self.ops[tree_node_idx] = '<>'
                        if isinstance(inequality[0].args[1], Symbol):
                            self.thresholds[tree_node_idx] = f"'{inequality[0].args[1].name}'"
                        else:
                            self.thresholds[tree_node_idx] = inequality[0].args[1]
                    elif isinstance(inequality[0], And):
                        self.features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature)
                        upper_bound = inequality[0].args[1].args[1]
                        self.thresholds[tree_node_idx] = float(upper_bound)
