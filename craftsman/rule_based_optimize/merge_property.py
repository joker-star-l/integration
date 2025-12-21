from copy import deepcopy
from sympy import Eq, simplify, LessThan, Or, Symbol, And, Symbol, Ne
from pandas import Series, DataFrame
from craftsman.model.base_model import TreeModel
from craftsman.utility.base_utils import merge_intervals
from craftsman.base.operator import CON_C_CAT

class PropertyManager(object):

    def __init__(self):
        pass
    
    def property1(self, op1, op2):
        if op1.is_arithmetic_op:
            if op2.is_arithmetic_op:
                if not op1.is_contain_ca_op and not op2.is_contain_ca_op:
                    merged_op = deepcopy(op2)
                    merged_op.equation = Eq(
                        op2.equation.lhs,
                        op2.equation.rhs.subs(
                            {op2.symbols["x"]: op1.equation.rhs}
                        )
                    )
                    merged_op.symbols = {**op2.symbols, **op1.symbols}
                    merged_op.parameter_values = [
                        {**op2.parameter_values[idx], **op1.parameter_values[idx]}
                        for idx in range(len(op1.parameter_values))
                    ]
                elif op1.is_contain_ca_op and not op2.is_contain_ca_op:
                    merged_op = deepcopy(op1)
                    for feature_idx, feature in enumerate(op1.features_out):
                        new_equations = []
                        for interval, equation in op1.mappings[feature_idx].items():
                            new_equation = Eq(
                                op2.equation.lhs,
                                op2.equation.rhs.subs(
                                    {op2.symbols["x"]: equation.rhs}
                                )
                            )
                            new_equations.append(new_equation)
                        merged_op.mappings[feature_idx] = Series(new_equations, index=op1.mappings[feature_idx].index)
                    merged_op.symbols = {**op2.symbols, **op1.symbols}
                    merged_op.parameter_values = [
                        {**op2.parameter_values[idx], **op1.parameter_values[idx]}
                        for idx in range(len(op1.parameter_values))
                    ]
                elif not op1.is_contain_ca_op and op2.is_contain_ca_op:
                    merged_op = deepcopy(op2)
                    for feature_idx, feature in enumerate(op1.features_out):
                        new_equations = []
                        for interval, equation in op1.mappings[feature_idx].items():
                            new_equation = Eq(
                                equation.lhs,
                                equation.rhs.subs(
                                    {op2.symbols["x"]: op1.equation.rhs}
                                )
                            )
                            new_equations.append(new_equation)
                        merged_op.mappings[feature_idx] = Series(new_equations, index=op2.mappings[feature_idx].index)
                    merged_op.symbols = {**op2.symbols, **op1.symbols}
                    merged_op.parameter_values = [
                        {**op2.parameter_values[idx], **op1.parameter_values[idx]}
                        for idx in range(len(op1.parameter_values))
                    ]
                elif op1.is_contain_ca_op and op2.is_contain_ca_op:
                    merged_op = deepcopy(op1)
                    # TODO: merge mappings

            elif op2.is_inequality_judgment_op and not op1.is_contain_ca_op:
                merged_op = deepcopy(op2)
                for idx, feature in enumerate(op1.features_out):
                    sub_equation = op1.equation.rhs.subs(
                        {
                            op1.symbols[sym_name]: op1.parameter_values[idx][sym_name]
                            for sym_name in op1.parameter_values[idx]
                        }
                    )
                    merged_inequations = []
                    for inequation in op2.inequations[feature]:
                        if isinstance(inequation, LessThan):
                            substituted_inequality = inequation.subs(op1.symbols['x'], sub_equation)
                            simplified_inequality = simplify(substituted_inequality)
                            merged_inequations.append(simplified_inequality)
                        elif isinstance(inequation, And):
                            substituted_inequality_left = inequation.args[0].subs(op1.symbols['x'], sub_equation)
                            simplified_inequality_left = simplify(substituted_inequality_left)
                            substituted_inequality_right= inequation.args[1].subs(op1.symbols['x'], sub_equation)
                            simplified_inequality_right = simplify(substituted_inequality_right)
                            merged_inequations.append(And(simplified_inequality_left, simplified_inequality_right))
                            
                    merged_op.inequations[feature] = merged_inequations
                    
                if isinstance(merged_op, TreeModel):
                    merged_op.update_tree_by_inequalities()
                if isinstance(merged_op, CON_C_CAT):
                    merged_op.update_intervals_by_inequalities()
                    
            else:
                return None
            
            return merged_op
                
        else:
            return None
        
    def property2(self, op1, op2):
        if op1.is_contain_ca_op and not op2.is_arithmetic_op:
            # if op2 contains ca(eg, con-s-cat, cat-s-cat, cat-N-vec)
            # op1's mapping -> op2's mapping
            if op2.is_contain_ca_op:
                merged_op = deepcopy(op2)
                for feature_idx, feature in enumerate(op1.features_out):
                    if hasattr(op1, 'mappings'):
                        if hasattr(op2, 'mappings'):
                            if op2.mappings[feature_idx].index.inferred_type == 'string':
                                new_values = op2.mappings[feature_idx][op1.mappings[feature_idx].values]
                            elif op2.mappings[feature_idx].index.inferred_type == 'mixed':
                                new_values = []
                                for op1_val in op1.mappings[feature_idx].values:
                                    for interval in op2.mappings[feature_idx].index:
                                        if op1_val >= interval[0] and op1_val <= interval[1]:
                                            new_values.append(op2.mappings[feature_idx][interval])
                                            break
                            merged_op.mappings[feature_idx] = Series(
                                new_values,
                                index=op1.mappings[feature_idx].index
                            )
                                
                        elif hasattr(op2, 'mapping'):
                            if isinstance(op2.mapping, DataFrame):
                                merged_op.mapping = op2.mapping.loc[op1.mappings[feature_idx].values.tolist()]
                                merged_op.mapping.index = op1.mappings[feature_idx].index
                            elif isinstance(op2.mappings[feature_idx], Series):
                                merged_op.mapping = op2.mapping[op1.mappings[feature_idx].values.tolist()]
                                merged_op.mapping.index = op1.mappings[feature_idx].index
                    
                    elif hasattr(op1, 'mapping'):
                        if hasattr(op2, 'mappings'):
                            if op2.mappings[feature_idx].index.inferred_type == 'string':
                                new_values = op2.mappings[feature_idx][op1.mapping[feature].values]
                            elif op2.mappings[feature_idx].index.inferred_type == 'mixed':
                                new_values = []
                                for op1_val in op1.mapping[feature].values:
                                    for interval in op2.mappings[feature_idx].index:
                                        if op1_val >= interval[0] and op1_val <= interval[1]:
                                            new_values.append(op2.mappings[feature_idx][interval])
                                            break
                            merged_op.mappings[feature_idx] = Series(
                                new_values,
                                index=op1.mapping.index
                            )
                            
                if isinstance(merged_op, CON_C_CAT):
                    merged_op.update_intervals_by_inequalities()
 
            # if op2 contains inequality(eg. tree model)
            # op1's mapping -> op2's inequations
            elif op2.is_inequality_judgment_op:
                merged_op = deepcopy(op2)
                for feature_idx, feature in enumerate(op1.features_out):
                    merged_inequations = []
                    valid_features = [f for f in op2.inequations if f in feature]
                    for feature in valid_features:
                        for inequation in op2.inequations[feature]:
                            if isinstance(inequation, LessThan):
                                if hasattr(op1, 'mappings'):
                                    if op1.mappings[feature_idx].index.inferred_type == 'string':
                                        thr = float(inequation.rhs)
                                        filtered_indexs = [idx for idx in op1.mappings[feature_idx].index if op1.mappings[feature_idx][idx] <= thr]
                                        if len(filtered_indexs) == 0:
                                            merged_inequation = []
                                        else:
                                            if isinstance(filtered_indexs[0], str): 
                                                merged_inequation = [Eq(Symbol('x'), Symbol(val)) for val in filtered_indexs]
                                            else:
                                                merged_inequation = [Eq(Symbol('x'), val) for val in filtered_indexs]
                                    elif op1.mappings[feature_idx].index.inferred_type == 'mixed':
                                        if isinstance(op1.mappings[feature_idx].iloc[0], Eq):
                                            intervals = []
                                            for eq_idx, (interval, equation) in enumerate(op1.mappings[feature_idx].items()):
                                                sub_equation = equation.rhs.subs(
                                                    {
                                                        op1.symbols[sym_name]: op1.parameter_values[idx][sym_name][eq_idx]
                                                        for sym_name in op1.parameter_values[idx]
                                                    }
                                                )
                                                substituted_inequality = inequation.subs(op1.symbols['x'], sub_equation)
                                                simplified_inequality = simplify(substituted_inequality)
                                                thr = float(simplified_inequality.rhs)
                                                if thr > interval[0] and thr <= interval[1]:
                                                    intervals.append((interval[0], thr))
                                                elif thr > interval[1]:
                                                    intervals.append(interval)
                                            merged_intervals = merge_intervals(filtered_intervals)
                                            x = Symbol('x')
                                            merged_inequation = [And(x > interval[0], x < interval[1]) for interval in merged_intervals]
                                        else:
                                            thr = float(inequation.rhs)
                                            filtered_intervals = [idx for idx in op1.mappings[feature_idx].index if op1.mappings[feature_idx][idx] <= thr]
                                            merged_intervals = merge_intervals(filtered_intervals)
                                            x = Symbol('x')
                                            merged_inequation = [And(x > interval[0], x < interval[1]) for interval in merged_intervals]
                                elif hasattr(op1, 'mapping'):
                                    thr = float(inequation.rhs)
                                    filtered_indexs = [idx for idx in op1.mapping.index if op1.mapping[feature][idx] <= thr]
                                    not_filtered_indexs = [idx for idx in op1.mapping.index if op1.mapping[feature][idx] > thr]
                                    if len(not_filtered_indexs) == 1:
                                        if isinstance(not_filtered_indexs[0], str): 
                                            merged_inequation = [Ne(Symbol('x'), Symbol(val)) for val in not_filtered_indexs]
                                        else:
                                            merged_inequation = [Ne(Symbol('x'), val) for val in not_filtered_indexs]
                                    else:
                                        if isinstance(filtered_indexs[0], str): 
                                            merged_inequation = [Eq(Symbol('x'), Symbol(val)) for val in filtered_indexs]
                                        else:
                                            merged_inequation = [Eq(Symbol('x'), val) for val in filtered_indexs]
                                
                            elif isinstance(inequation, list) and len(inequation) > 1:
                                if isinstance(inequation[0], Eq): 
                                    if hasattr(op1, 'mappings'):
                                        if op1.mappings[feature_idx].index.inferred_type == 'string':
                                            filtered_indexs = []
                                            inversed_mapping = Series(op1.mappings[feature_idx].index, index=op1.mappings[feature_idx].values)
                                            inversed_mapping = inversed_mapping.groupby(inversed_mapping.index).apply(list)
                                            for equality in inequation:
                                                if isinstance(equality.args[1], Symbol): 
                                                    val = equality.args[1].name
                                                else:
                                                    val = equality.args[1]
                                                filtered_indexs.extend(inversed_mapping[val])
                                                
                                            if isinstance(filtered_indexs[0], str): 
                                                merged_inequation = [Eq(Symbol('x'), Symbol(val)) for val in filtered_indexs]
                                            else:
                                                merged_inequation = [Eq(Symbol('x'), val) for val in filtered_indexs]
                                        elif op1.mappings[feature_idx].index.inferred_type == 'mixed':
                                            filtered_intervals = []
                                            inversed_mapping = Series(op1.mappings[feature_idx].index, index=op1.mappings[feature_idx].values)
                                            inversed_mapping = inversed_mapping.groupby(inversed_mapping.index).apply(list)
                                            for equality in inequation:
                                                if isinstance(equality.args[1], Symbol): 
                                                    val = equality.args[1].name
                                                else:
                                                    val = equality.args[1]
                                                filtered_intervals.extend(inversed_mapping[val])
                                            merged_intervals = merge_intervals(filtered_intervals)
                                            x = Symbol('x')
                                            merged_inequation = Or(*[And(x > interval[0], x < interval[1]) for interval in merged_intervals])
                                    elif hasattr(op1, 'mapping'):
                                        filtered_indexs = []
                                        inversed_mapping = Series(op1.mapping.index, index=op1.mapping[feature].values)
                                        inversed_mapping = inversed_mapping.groupby(inversed_mapping.index).apply(list)
                                        for equality in inequation:
                                            if isinstance(equality.args[1], Symbol): 
                                                val = equality.args[1].name
                                            else:
                                                val = equality.args[1]
                                            filtered_indexs.extend(inversed_mapping[val])
                                        if isinstance(filtered_indexs[0], str): 
                                            merged_inequation = [Eq(Symbol('x'), Symbol(val)) for val in filtered_indexs]
                                        else:
                                            merged_inequation = [Eq(Symbol('x'), val) for val in filtered_indexs]
                                
                                elif isinstance(inequation[0], And): 
                                    filtered_intervals = []
                                    inversed_mapping = Series(op1.mappings[feature_idx].index, index=op1.mappings[feature_idx].values)
                                    inversed_mapping = inversed_mapping.groupby(inversed_mapping.index).apply(list)
                                    for in_eq in inequation:
                                        for val in inversed_mapping.index:
                                            if val >= float(in_eq.args[0]) and val < float(in_eq.args[1]):
                                                filtered_intervals.extend(inversed_mapping[val])
                                    merged_intervals = merge_intervals(filtered_intervals)
                                    x = Symbol('x')
                                    merged_inequation = [And(x > interval[0], x < interval[1]) for interval in merged_intervals]
                            
                            elif isinstance(inequation, list) and len(inequation) == 1:
                                if isinstance(inequation[0], Eq) or isinstance(inequation[0], Ne):
                                    if isinstance(inequation[0], Eq):
                                        op = Eq
                                    elif isinstance(inequation[0], Ne):
                                        op = Ne
                                    if hasattr(op1, 'mappings'):
                                        if op1.mappings[feature_idx].index.inferred_type == 'string':
                                            filtered_indexs = []
                                            inversed_mapping = Series(op1.mappings[feature_idx].index, index=op1.mappings[feature_idx].values)
                                            inversed_mapping = inversed_mapping.groupby(inversed_mapping.index).apply(list)
                                            if isinstance(inequation[0].args[1], Symbol): 
                                                val = inequation[0].args[1].name
                                            else:
                                                val = inequation[0].args[1]
                                            filtered_indexs.extend(inversed_mapping[val])
                                            if isinstance(filtered_indexs[0], str): 
                                                merged_inequation = [op(Symbol('x'), Symbol(val)) for val in filtered_indexs]
                                            else:
                                                merged_inequation = [op(Symbol('x'), val) for val in filtered_indexs]
                                        elif op1.mappings[feature_idx].index.inferred_type == 'mixed':
                                            filtered_intervals = []
                                            inversed_mapping = Series(op1.mappings[feature_idx].index, index=op1.mappings[feature_idx].values)
                                            inversed_mapping = inversed_mapping.groupby(inversed_mapping.index).apply(list)
                                            if isinstance(inequation[0].args[1], Symbol): 
                                                val = inequation[0].args[1].name
                                            else:
                                                val = inequation[0].args[1]
                                            filtered_intervals.extend(inversed_mapping[val])
                                            merged_intervals = merge_intervals(filtered_intervals)
                                            x = Symbol('x')
                                            merged_inequation = [And(x > interval[0], x < interval[1]) for interval in merged_intervals]
                                    elif hasattr(op1, 'mapping'):
                                        filtered_indexs = []
                                        inversed_mapping = Series(op1.mapping.index, index=op1.mapping[feature].values)
                                        inversed_mapping = inversed_mapping.groupby(inversed_mapping.index).apply(list)
                                        if isinstance(inequation[0].args[1], Symbol): 
                                            val = inequation[0].args[1].name
                                        else:
                                            val = inequation[0].args[1]
                                        filtered_indexs.extend(inversed_mapping[val])
                                        if isinstance(filtered_indexs[0], str): 
                                            merged_inequation = [op(Symbol('x'), Symbol(val)) for val in filtered_indexs]
                                        else:
                                            merged_inequation = [op(Symbol('x'), val) for val in filtered_indexs]
                            
                                elif isinstance(inequation[0], And):
                                    if hasattr(op1, 'mappings'):
                                        if op1.mappings[feature_idx].index.inferred_type == 'mixed':
                                            filtered_intervals = []
                                            inversed_mapping = Series(op1.mappings[feature_idx].index, index=op1.mappings[feature_idx].values)
                                            inversed_mapping = inversed_mapping.groupby(inversed_mapping.index).apply(list)
                                            for val in inversed_mapping.index:
                                                if val >= float(inequation[0].args[0].args[1]) and val < float(inequation[0].args[1].args[1]):
                                                    filtered_intervals.extend(inversed_mapping[val])
                                            merged_intervals = merge_intervals(filtered_intervals)
                                            x = Symbol('x')
                                            merged_inequation = [And(x > interval[0], x < interval[1]) for interval in merged_intervals]
                                        elif op1.mappings[feature_idx].index.inferred_type == 'string':
                                            filtered_indexs = []
                                            inversed_mapping = Series(op1.mappings[feature_idx].index, index=op1.mappings[feature_idx].values)
                                            inversed_mapping = inversed_mapping.groupby(inversed_mapping.index).apply(list)
                                            for val in inversed_mapping.index:
                                                if val >= float(inequation[0].args[0].args[1]) and val < float(inequation[0].args[1].args[1]):
                                                    filtered_indexs.extend(inversed_mapping[val])
                                            x = Symbol('x')
                                            if isinstance(filtered_indexs[0], str): 
                                                merged_inequation = [Eq(Symbol('x'), Symbol(val)) for val in filtered_indexs]
                                            else:
                                                merged_inequation = [Eq(Symbol('x'), val) for val in filtered_indexs]
                                    elif hasattr(op1, 'mapping'):
                                        filtered_indexs = []
                                        not_filtered_indexs = []
                                        inversed_mapping = Series(op1.mapping.index, index=op1.mapping[feature].values)
                                        inversed_mapping = inversed_mapping.groupby(inversed_mapping.index).apply(list)
                                        for val in inversed_mapping.index:
                                            if val >= float(inequation[0].args[0].args[1]) and val < float(inequation[0].args[1].args[1]):
                                                filtered_indexs.extend(inversed_mapping[val])
                                            else:
                                                not_filtered_indexs.extend(inversed_mapping[val])
                                        x = Symbol('x')
                                        if len(not_filtered_indexs) == 1:
                                            if isinstance(not_filtered_indexs[0], str): 
                                                merged_inequation = [Ne(Symbol('x'), Symbol(val)) for val in not_filtered_indexs]
                                            else:
                                                merged_inequation = [Ne(Symbol('x'), val) for val in not_filtered_indexs]
                                        else:
                                            if isinstance(filtered_indexs[0], str): 
                                                merged_inequation = [Eq(Symbol('x'), Symbol(val)) for val in filtered_indexs]
                                            else:
                                                merged_inequation = [Eq(Symbol('x'), val) for val in filtered_indexs]
                                        
                                
                            merged_inequations.append(merged_inequation)
                    merged_op.inequations[feature] = merged_inequations
                    
                if isinstance(merged_op, TreeModel):
                    merged_op.update_tree_by_inequalities()
                if isinstance(merged_op, CON_C_CAT):
                    merged_op.update_intervals_by_inequalities()
                    
            else:
                return None
            
            merged_op.features = op1.features_out  
            return merged_op
        else:
            return None
    
    def property3(self, op1, op2):
        if op1.is_constant_output_op and not op2.is_contain_multi_ca_op:
            if op2.is_arithmetic_op:
                merged_op = deepcopy(op1)
                for feature_idx, feature in enumerate(op1.features_out):
                    sub_equation = op2.equation.rhs.subs(
                        {
                            op2.symbols[sym_name]: op2.parameter_values[feature_idx][sym_name]
                            for sym_name in op2.parameter_values[feature_idx]
                        }
                    )
                    if hasattr(op1, 'mappings'):
                        for idx in op1.mappings[feature_idx].index:
                            merged_op.mappings[feature_idx][idx] = sub_equation.subs(op2.symbols['x'], op1.mappings[feature_idx][idx])
                    elif hasattr(op1, 'mapping'):
                        for idx in op1.mapping.index:
                             merged_op.mapping[feature][idx] = sub_equation.subs(op2.symbols['x'], op1.mapping[feature][idx])
                            
            elif op2.is_contain_ca_op:
                merged_op = deepcopy(op1)
                if hasattr(op1, 'mappings'):
                    for feature_idx, _ in enumerate(op1.features_out):
                        mapping = op2.mappings[feature_idx]
                        if mapping.index.inferred_type == 'string':
                            for idx in op1.mappings[feature_idx].index:
                                merged_op.mappings[feature_idx][idx] = mapping(op1.mappings[feature_idx][idx])
                        elif mapping.index.inferred_type == 'mixed':
                            for idx in op1.mappings[feature_idx].index:
                                for interval in mapping.index:
                                    if op1.mappings[feature_idx][idx] >= interval[0] and op1.mappings[feature_idx][idx] < interval[1]:
                                        merged_op.mappings[feature_idx][idx] = mapping[interval]
                                        break
                
                elif hasattr(op1, 'mapping'):
                    for feature_idx, feature in enumerate(op1.features_out):
                        mapping = op2.mappings[feature_idx]
                        if mapping.index.inferred_type == 'string':
                            for idx in op1.mapping.index:
                                merged_op.mapping[feature][idx] = mapping(op1.mapping[feature][idx])
                        elif mapping.index.inferred_type == 'mixed':
                            for idx in op1.mapping.index:
                                for interval in mapping.index:
                                    if op1.mapping[feature][idx] >= interval[0] and op1.mapping[feature][idx] < interval[1]:
                                        merged_op.mapping[feature][idx] = mapping[interval]
                                        break
            else:
                return None
            if isinstance(merged_op, CON_C_CAT):
                merged_op.update_intervals_by_inequalities()
            return merged_op
        else:
            return None