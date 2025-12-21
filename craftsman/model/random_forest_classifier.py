from sklearn.ensemble import RandomForestClassifier
from sympy import sympify, LessThan, Eq, And, Symbol, Ne, Tuple
from craftsman.utility.dbms_utils import DBMSUtils
from craftsman.model.decision_tree_classifier import DecisionTreeClassifierSQLModel
from craftsman.model.base_model import TreeModel
from craftsman.base.operator import Operator
from craftsman.base.defs import ModelName
import craftsman.base.defs as defs
from craftsman.cost_model.cost import TreeCost

class RandomForestClassifierSQLModel(TreeModel):
    """
    This class implements the SQL wrapper for a Sklearn's Random Forest Model (RFM).

    The current random forest classifier implementation applies a majority voting technique across the forest labels,
    while Sklearn computes the the predicted class by selecting the one with highest mean probability estimate across
    the trees.
    """

    def __init__(self, trained_model: RandomForestClassifier):
        super().__init__()
        self.model_name = ModelName.RANDOMFORESTCLASSIFIER
        self.trained_model = trained_model
        self.input_features = trained_model.feature_names_in_
        self.estimators = trained_model.estimators_
        self.classes = trained_model.classes_
        self.decision_tree_classifiers: list[DecisionTreeClassifierSQLModel] = []
        for estimator in self.estimators:
            decision_tree_classifier = DecisionTreeClassifierSQLModel(estimator)
            decision_tree_classifier.set_features(trained_model.feature_names_in_)
            self.decision_tree_classifiers.append(decision_tree_classifier)
        
        if defs.AUTO_RULE_GEN:
            # abstract the tree model to a operator consisted of inequations
            for feature in self.input_features:
                self.inequations[feature] = []
                self.tree_node_mappings[feature] = []
            for tree_idx, dtc in enumerate(self.decision_tree_classifiers):
                for feature in dtc.input_features:
                    self.inequations[feature].extend(dtc.inequations[feature])
                    self.tree_node_mappings[feature].extend([(tree_idx, node_idx) for node_idx in dtc.tree_node_mappings[feature]])
        
    def modify_model(self, feature: str, sql_operator: Operator):
        for decision_tree_classifier in self.decision_tree_classifiers:
            decision_tree_classifier.modify_model(feature, sql_operator)
            
    def modify_model_p(self, feature: str, sql_operator: Operator):
        for decision_tree_classifier in self.decision_tree_classifiers:
            decision_tree_classifier.modify_model_p(feature, sql_operator)
            
    def get_tree_costs(self, feature, operator):
        tree_costs = []
        for dtc in self.decision_tree_classifiers:
            tree_cost = TreeCost(feature, operator, dtc)
            tree_cost.analyze_path_fusion_cost(feature, operator, dtc)
            tree_costs.append(tree_cost)
        return tree_costs
    
    def get_tree_costs_p(self, feature, operator):
        tree_costs = []
        for dtc in self.decision_tree_classifiers:
            tree_cost = TreeCost(feature, operator, dtc)
            tree_cost.analyze_path_push_cost(feature, operator, dtc)
            tree_costs.append(tree_cost)
        return tree_costs
    
    def get_tree_costs_static(self):
        tree_costs = []
        for dtc in self.decision_tree_classifiers:
            tree_cost = TreeCost(model=dtc)
            tree_cost.analyze_path_cost(dtc)
            tree_costs.append(tree_cost)
        return tree_costs

    def query(self, input_table: str, dbms: str) -> str:

        query = "SELECT "
        # loop over the trees and create a CASE statement for each tree
        for i in range(len(self.decision_tree_classifiers)):
            decision_tree_classifier = self.decision_tree_classifiers[i]
            sql_case = decision_tree_classifier.get_case_sql(dbms)
            sql_case += " AS tree_{},".format(i)
            query += sql_case

        query = query[:-1]  # remove the last ","

        query += " FROM {}".format(input_table)

        # count the number of trees that have predicted the same class label
        majority_class_query = "SELECT "
        for class_ix, class_label in enumerate(self.classes):
            majority_class_query += "("

            for i in range(len(self.decision_tree_classifiers)):
                majority_class_query += "CASE WHEN tree_{} = {} THEN 1 ELSE 0 END + ".format(i, class_label)
            majority_class_query = majority_class_query[:-3]  # remove the last ' + '

            majority_class_query += ") AS class_{}, ".format(class_ix)
        majority_class_query = majority_class_query[:-2]  # remove the last ', '
        majority_class_query += " FROM ({}) AS F".format(query)

        # find the majority class label
        final_query = "SELECT "
        case_stm = "CASE"
        for i in range(len(self.classes)):
            case_stm += " WHEN "
            for j in range(len(self.classes)):
                if j == i:
                    continue
                case_stm += "class_{} >= class_{} AND ".format(i, j)
            case_stm = case_stm[:-5]  # remove the last ' AND '
            case_stm += " THEN {}\n".format(self.classes[i])
        case_stm += "END AS Score"

        final_query += "{} FROM ({}) AS F".format(case_stm, majority_class_query)

        return final_query
    
    def update_tree_by_inequalities(self):
        for feature in self.input_features:
            for (tree_idx, tree_node_idx), inequality in zip(self.tree_node_mappings[feature], self.inequations[feature]):
                if isinstance(inequality, LessThan):
                    self.decision_tree_classifiers[tree_idx].features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature) 
                    self.decision_tree_classifiers[tree_idx].ops[tree_node_idx] = '<='
                    self.decision_tree_classifiers[tree_idx].thresholds[tree_node_idx] = float(inequality.rhs)
                elif isinstance(inequality, list) and len(inequality) > 1:
                    if isinstance(inequality[0], Eq): 
                        equal_values = []
                        for equality in inequality:
                            if isinstance(equality.args[1], Symbol):
                                equal_values.append(f"'{equality.args[1].name}'")
                            else:
                                equal_values.append(f"{equality.args[1]}")
                        if feature.split('_')[-1].isnumeric():
                            self.decision_tree_classifiers[tree_idx].features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature.rsplit('_', 1)[0])
                        else:
                            self.decision_tree_classifiers[tree_idx].features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature) 
                        self.decision_tree_classifiers[tree_idx].ops[tree_node_idx] = 'in'
                        self.decision_tree_classifiers[tree_idx].thresholds[tree_node_idx] = f"({','.join(equal_values)})"  
                    elif isinstance(inequality[0], And):
                        interval_strs = []
                        for and_expr in inequality:
                            lower_bound = and_expr.args[0].args[1]
                            upper_bound = and_expr.args[1].args[1]
                            interval_strs.append(f"{DBMSUtils.get_delimited_col(defs.DBMS, feature)} >= {lower_bound}" 
                                                 + " AND " + 
                                                 f"{DBMSUtils.get_delimited_col(defs.DBMS, feature)} < {upper_bound}")
                        self.decision_tree_classifiers[tree_idx].features[tree_node_idx] = " OR ".join(interval_strs)
                        self.decision_tree_classifiers[tree_idx].ops[tree_node_idx] = ''
                        self.decision_tree_classifiers[tree_idx].thresholds[tree_node_idx] = ''
                elif isinstance(inequality, list) and len(inequality) == 1:
                    if isinstance(inequality[0], Eq):
                        if feature not in defs.PIPELINE_FEATURES_IN and feature.split('_')[-1].isnumeric():
                            self.decision_tree_classifiers[tree_idx].features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature.rsplit('_', 1)[0])
                        else:
                            self.decision_tree_classifiers[tree_idx].features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature) 
                        self.decision_tree_classifiers[tree_idx].ops[tree_node_idx] = '='
                        if isinstance(inequality[0].args[1], Symbol):
                            self.decision_tree_classifiers[tree_idx].thresholds[tree_node_idx] = f"'{inequality[0].args[1].name}'"
                        else:
                            self.decision_tree_classifiers[tree_idx].thresholds[tree_node_idx] = inequality[0].args[1]
                    elif isinstance(inequality[0], Ne):
                        # if feature == 'var_191_9':
                        #     pass
                        if isinstance(inequality[0].args[1], Tuple):
                            if feature not in defs.PIPELINE_FEATURES_IN and feature.split('_')[-1].isnumeric():
                                self.decision_tree_classifiers[tree_idx].features[tree_node_idx] = f"{DBMSUtils.get_delimited_col(defs.DBMS, feature.rsplit('_', 1)[0])} < {inequality[0].args[1][0]}" + \
                                    " OR " + \
                                        f"{DBMSUtils.get_delimited_col(defs.DBMS, feature.rsplit('_', 1)[0])} >= {inequality[0].args[1][1]}"
                            else:
                                self.decision_tree_classifiers[tree_idx].features[tree_node_idx] = f"{DBMSUtils.get_delimited_col(defs.DBMS, feature)} < {inequality[0].args[1][0]}" +\
                                    " OR " + \
                                        f"{DBMSUtils.get_delimited_col(defs.DBMS, feature)} >= {inequality[0].args[1][1]}"
                        else: 
                            if feature not in defs.PIPELINE_FEATURES_IN and feature.split('_')[-1].isnumeric():
                                self.decision_tree_classifiers[tree_idx].features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature.rsplit('_', 1)[0])
                            else:
                                self.decision_tree_classifiers[tree_idx].features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature) 
                            self.decision_tree_classifiers[tree_idx].ops[tree_node_idx] = '<>'
                            if isinstance(inequality[0].args[1], Symbol):
                                self.decision_tree_classifiers[tree_idx].thresholds[tree_node_idx] = f"'{inequality[0].args[1].name}'"
                            else:
                                self.decision_tree_classifiers[tree_idx].thresholds[tree_node_idx] = inequality[0].args[1]
                    elif isinstance(inequality[0], And):
                        self.decision_tree_classifiers[tree_idx].features[tree_node_idx] = DBMSUtils.get_delimited_col(defs.DBMS, feature)
                        self.decision_tree_classifiers[tree_idx].ops[tree_node_idx] = '<='
                        upper_bound = inequality[0].args[1].args[1]
                        self.decision_tree_classifiers[tree_idx].thresholds[tree_node_idx] = float(upper_bound)