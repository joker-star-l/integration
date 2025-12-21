from sklearn.ensemble import RandomForestRegressor
from sympy import sympify, LessThan, Eq, And, Symbol, Ne
from craftsman.utility.dbms_utils import DBMSUtils
from craftsman.model.decision_tree_regressor import DecisionTreeRegressorSQLModel
from craftsman.model.base_model import TreeModel
from craftsman.base.operator import Operator
from craftsman.base.defs import ModelName
import craftsman.base.defs as defs
from craftsman.cost_model.cost import TreeCost

class RandomForestRegressorSQLModel(TreeModel):
    """
    This class implements the SQL wrapper for a Sklearn's Random Forest Model (RFM).

    The current random forest classifier implementation applies a majority voting technique across the forest labels,
    while Sklearn computes the the predicted class by selecting the one with highest mean probability estimate across
    the trees.
    """

    def __init__(self, trained_model: RandomForestRegressor):
        super().__init__()
        self.model_name = ModelName.RANDOMFORESTREGRESSOR
        self.trained_model = trained_model
        self.input_features = trained_model.feature_names_in_
        self.estimators = trained_model.estimators_
        self.decision_tree_regressors: list[DecisionTreeRegressorSQLModel] = []
        for estimator in self.estimators:
            decision_tree_regressor = DecisionTreeRegressorSQLModel(estimator)
            decision_tree_regressor.set_features(trained_model.feature_names_in_)
            self.decision_tree_regressors.append(decision_tree_regressor)
        self.tree_weights = [1/len(self.decision_tree_regressors)] * len(self.decision_tree_regressors)
        if defs.AUTO_RULE_GEN:
            # abstract the tree model to a operator consisted of inequations
            for feature in self.input_features:
                self.inequations[feature] = []
                self.tree_node_mappings[feature] = []
            for tree_idx, dtc in enumerate(self.decision_tree_regressors):
                for feature in dtc.input_features:
                    self.inequations[feature].extend(dtc.inequations[feature])
                    self.tree_node_mappings[feature].extend([(tree_idx, node_idx) for node_idx in dtc.tree_node_mappings[feature]])
        
    def modify_model(self, feature: str, sql_operator: Operator):
        for decision_tree_regressor in self.decision_tree_regressors:
            decision_tree_regressor.modify_model(feature, sql_operator)
            
    def modify_model_p(self, feature: str, sql_operator: Operator):
        for decision_tree_regressor in self.decision_tree_regressors:
            decision_tree_regressor.modify_model_p(feature, sql_operator)
            
    def get_tree_costs(self, feature, operator):
        tree_costs = []
        for dtc in self.decision_tree_regressors:
            tree_cost = TreeCost(feature, operator, dtc)
            tree_cost.analyze_path_fusion_cost(feature, operator, dtc)
            tree_costs.append(tree_cost)
        return tree_costs
    
    def get_tree_costs_p(self, feature, operator):
        tree_costs = []
        for dtc in self.decision_tree_regressors:
            tree_cost = TreeCost(feature, operator, dtc)
            tree_cost.analyze_path_push_cost(feature, operator, dtc)
            tree_costs.append(tree_cost)
        return tree_costs
    
    def get_tree_costs_static(self):
        tree_costs = []
        for dtc in self.decision_tree_regressors:
            tree_cost = TreeCost(model=dtc)
            tree_cost.analyze_path_cost(dtc)
            tree_costs.append(tree_cost)
        return tree_costs

    def query(self, input_table: str, dbms: str) -> str:
        query = "SELECT "
        # loop over the trees and create a CASE statement for each tree
        for i in range(len(self.decision_tree_regressors)):
            decision_tree_regressor = self.decision_tree_regressors[i]
            sql_case = decision_tree_regressor.get_case_sql(dbms)
            sql_case += " AS tree_{},".format(i)
            query += sql_case
        query = query[:-1]  # remove the last ","
        query += " FROM {}".format(input_table)

        # combine the tree scores with a mean
        external_query = " SELECT ("
        for i in range(len(self.decision_tree_regressors)):
            tree_weight = self.tree_weights[i]
            external_query += "{} * tree_{} + ".format(tree_weight, i)

        external_query = external_query[:-3]  # remove the last ' + '
        external_query += ") AS Score"

        # combine internal and external queries in a single query
        final_query = "{} FROM ({}) AS F".format(external_query, query)

        return final_query
    
    def update_tree_by_inequalities(self):
        for feature in self.input_features:
            for (tree_idx, tree_node_idx), inequality in zip(self.tree_node_mappings[feature], self.inequations[feature]):
                if isinstance(inequality, LessThan):
                    self.decision_tree_regressors[tree_idx].features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature) 
                    self.decision_tree_regressors[tree_idx].ops[tree_node_idx] = '<='
                    self.decision_tree_regressors[tree_idx].thresholds[tree_node_idx] = float(inequality.rhs)
                elif isinstance(inequality, list) and len(inequality) > 1:
                    if isinstance(inequality[0], Eq): 
                        equal_values = []
                        for equality in inequality:
                            if isinstance(equality.args[1], Symbol):
                                equal_values.append(f"'{equality.args[1].name}'")
                            else:
                                equal_values.append(f"{equality.args[1]}")
                        if feature.split('_')[-1].isnumeric():
                            self.decision_tree_regressors[tree_idx].features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature.rsplit('_', 1)[0])
                        else:
                            self.decision_tree_regressors[tree_idx].features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature) 
                        self.decision_tree_regressors[tree_idx].ops[tree_node_idx] = 'in'
                        self.decision_tree_regressors[tree_idx].thresholds[tree_node_idx] = f"({','.join(equal_values)})"  
                    elif isinstance(inequality[0], And):
                        interval_strs = []
                        for and_expr in inequality:
                            lower_bound = and_expr.args[0].args[1]
                            upper_bound = and_expr.args[1].args[1]
                            interval_strs.append(f"{DBMSUtils.get_delimited_col(defs.DBMS, feature)} >= {lower_bound}" 
                                                 + " AND " + 
                                                 f"{DBMSUtils.get_delimited_col(defs.DBMS, feature)} < {upper_bound}")
                        self.decision_tree_regressors[tree_idx].features[tree_node_idx] = " OR ".join(interval_strs)
                        self.decision_tree_regressors[tree_idx].ops[tree_node_idx] = ''
                        self.decision_tree_regressors[tree_idx].thresholds[tree_node_idx] = ''
                elif isinstance(inequality, list) and len(inequality) == 1:
                    if isinstance(inequality[0], Eq):
                        if feature not in defs.PIPELINE_FEATURES_IN and feature.split('_')[-1].isnumeric():
                            self.decision_tree_regressors[tree_idx].features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature.rsplit('_', 1)[0])
                        else:
                            self.decision_tree_regressors[tree_idx].features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature) 
                        self.decision_tree_regressors[tree_idx].ops[tree_node_idx] = '='
                        if isinstance(inequality[0].args[1], Symbol):
                            self.decision_tree_regressors[tree_idx].thresholds[tree_node_idx] = f"'{inequality[0].args[1].name}'"
                        else:
                            self.decision_tree_regressors[tree_idx].thresholds[tree_node_idx] = inequality[0].args[1]
                    elif isinstance(inequality[0], Ne):
                        if feature not in defs.PIPELINE_FEATURES_IN and feature.split('_')[-1].isnumeric():
                            self.decision_tree_regressors[tree_idx].features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature.rsplit('_', 1)[0])
                        else:
                            self.decision_tree_regressors[tree_idx].features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature) 
                        self.decision_tree_regressors[tree_idx].ops[tree_node_idx] = '<>'
                        if isinstance(inequality[0].args[1], Symbol):
                            self.decision_tree_regressors[tree_idx].thresholds[tree_node_idx] = f"'{inequality[0].args[1].name}'"
                        else:
                            self.decision_tree_regressors[tree_idx].thresholds[tree_node_idx] = inequality[0].args[1]
                    elif isinstance(inequality[0], And):
                        self.decision_tree_regressors[tree_idx].features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature)
                        self.decision_tree_regressors[tree_idx].ops[tree_node_idx] = '<='
                        upper_bound = inequality[0].args[1].args[1]
                        self.decision_tree_regressors[tree_idx].thresholds[tree_node_idx] = float(upper_bound)