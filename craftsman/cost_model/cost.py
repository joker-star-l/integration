from craftsman.cost_model.utils import tree_paths
from craftsman.base.defs import PrimitiveType, PrimitiveCost
import craftsman.base.defs as defs


class PathCost(object):
    """
    Path cost of specific field
    """

    def __init__(self, path, feature=None):
        self.path = path
        self.feature = feature
        self.occurs_time = 0
        self.infos = {primitive_type.value: {} for primitive_type in PrimitiveType}

    def set_occurs(self, count):
        self.occurs_time = count  # TODO: extend "<=" to others:

    def set_cost(self, primitive=PrimitiveType.LE_EQ, length=1):
        if self.infos[primitive.value].get(length):
            self.infos[primitive.value][length] += 1
        else:
            self.infos[primitive.value][length] = 1


class TreeCost(object):
    """
    Tree cost for a special field
    """

    def __init__(self, feature=None, operator=None, model=None) -> None:
        self.feature = feature
        self.path_costs: list[PathCost] = []
        self.n_node_samples = model.trained_model.tree_.n_node_samples

    def analyze_path_cost(self, model):
        tree = model.trained_model.tree_
        for path in tree_paths(tree):
            path_cost = PathCost(path)
            occurs_time = 0
            for node in path:
                
                # use for no fusion
                occurs_time += 1
                # use for fusion
                op = model.ops[node]
                if op == '<=':
                    primitive_type = PrimitiveType.LE_EQ
                    primitive_length = 1
                elif op == '=':
                    primitive_type = PrimitiveType.EQUAL
                    primitive_length = 1
                elif op == '<>':
                    primitive_type = PrimitiveType.INEQUAL
                    primitive_length = 1
                elif op == '<':
                    primitive_type = PrimitiveType.LE
                    primitive_length = 1
                elif op == '>':
                    primitive_type = PrimitiveType.GE
                    primitive_length = 1
                elif op == '>=':
                    primitive_type = PrimitiveType.GE_EQ
                    primitive_length = 1
                elif op == 'in':
                    primitive_type = PrimitiveType.IN
                    primitive_length = defs.tree_node_in_length.get(node)
                    if not primitive_length:
                        primitive_length = len(model.thresholds[node].split(','))
                        defs.tree_node_in_length[node] = primitive_length
                elif op == '':
                    primitive_type = PrimitiveType.OR
                    primitive_length = defs.tree_node_or_length.get(node)
                    if not primitive_length:
                        primitive_length = len(model.features[node].split('OR'))
                        defs.tree_node_or_length[node] = primitive_length
                     
                path_cost.set_cost(primitive_type, primitive_length)

            path_cost.set_occurs(occurs_time)
            self.path_costs.append(path_cost)

    def analyze_path_fusion_cost(self, feature, operator, model):
        tree = model.trained_model.tree_
        for path in tree_paths(tree):
            path_cost = PathCost(path, feature)
            occurs_time = 0
            for node in path:
                if model.input_features[tree.feature[node]] == feature:
                    # use for no fusion
                    occurs_time += 1
                    # use for fusion
                    primitive_type = operator.get_fusion_primitive_type(
                        feature, tree.threshold[node]
                    )
                    primitive_length = operator.get_fusion_primitive_length(
                        feature, tree.threshold[node]
                    )
                    path_cost.set_cost(primitive_type, primitive_length)

            path_cost.set_occurs(occurs_time)
            self.path_costs.append(path_cost)
            
    def analyze_path_push_cost(self, feature, operator, model):
        tree = model.trained_model.tree_
        for path in tree_paths(tree):
            path_cost = PathCost(path, feature)
            occurs_time = 0
            for node in path:
                if model.input_features[tree.feature[node]] == feature:
                    # use for no fusion
                    occurs_time += 1
                    # use for fusion
                    primitive_type = operator.get_push_primitive_type(
                        feature, tree.threshold[node]
                    )
                    primitive_length = operator.get_push_primitive_length(
                        feature, tree.threshold[node]
                    )
                    path_cost.set_cost(primitive_type, primitive_length)

            path_cost.set_occurs(occurs_time)
            self.path_costs.append(path_cost)

    def calculate_tree_cost(self):
        fusion_cost = 0
        total_infos = {}
        for path_cost in self.path_costs:
            path_leaf_node = path_cost.path[-1]
            path_quantity = self.n_node_samples[path_leaf_node]
            
            infos_with_quantity = {}
            for primitive, statistics in path_cost.infos.items():
                infos_with_quantity[primitive] = {
                    key: value * path_quantity
                    for key, value in statistics.items()
                }
                
            merged_dict = {}
            for key in set(total_infos) | set(infos_with_quantity):
                # If the key exists in both dictionaries, merge their value dictionaries
                if key in total_infos and key in infos_with_quantity:
                    merged_value = {
                        k: total_infos[key].get(k, 0) + infos_with_quantity[key].get(k, 0)
                        for k in set(total_infos[key]) | set(infos_with_quantity[key])
                    }
                else:
                    # If the key is in only one of the dictionaries, the value from that dictionary is used directly
                    merged_value = total_infos.get(key, infos_with_quantity.get(key))
                merged_dict[key] = merged_value
            total_infos = merged_dict
            
        for primitive, statistics in total_infos.items():
            fusion_cost += self.__cal_primitive_cost(primitive, statistics)
        
        return fusion_cost
    
    def __cal_primitive_cost(self, primitive, statistics):
        cost = 0
        match PrimitiveType(primitive):
            case PrimitiveType.IN:
                for length, quantity in statistics.items():
                    cost += PrimitiveCost.IN(length) * quantity
                return cost
            case PrimitiveType.OR:
                for length, quantity in statistics.items():
                    cost += (PrimitiveCost.GE_EQ.value + PrimitiveCost.LE.value) * quantity * length
                return cost
            case PrimitiveType.INEQUAL:
                for length, quantity in statistics.items():
                    cost += PrimitiveCost.INEQUAL.value * quantity * length
                return cost
            case PrimitiveType.EQUAL:
                for length, quantity in statistics.items():
                    cost += PrimitiveCost.EQUAL.value * quantity * length
                return cost
            case PrimitiveType.LE:
                for length, quantity in statistics.items():
                    cost += PrimitiveCost.LE.value * quantity * length
                return cost
            case PrimitiveType.LE_EQ:
                for length, quantity in statistics.items():
                    cost += PrimitiveCost.LE_EQ.value * quantity * length
                return cost
            case PrimitiveType.GE:
                for length, quantity in statistics.items():
                    cost += PrimitiveCost.GE.value * quantity * length
                return cost
            case PrimitiveType.GE_EQ:
                for length, quantity in statistics.items():
                    cost += PrimitiveCost.GE_EQ.value * quantity * length
                return cost
            
    def calculate_no_fusion_cost(self):
        no_fusion_cost = 0
        for path_cost in self.path_costs:
            path_leaf_node = path_cost.path[-1]
            path_quantity = self.n_node_samples[path_leaf_node]
            no_fusion_cost += path_quantity * path_cost.occurs_time * PrimitiveCost.LE_EQ.value
        return no_fusion_cost
