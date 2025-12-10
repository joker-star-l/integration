from typing import Union, Tuple, List, Callable, Any
from loguru import logger
import sqlparse

class MLPredicate:
    def __init__(self, token: sqlparse.sql.Comparison):
        self.token: sqlparse.sql.Comparison = token
        self.model_path: sqlparse.sql.Token = MLPredicate.get_model_path(token)
        self.comparison_operator: sqlparse.sql.Token = MLPredicate.get_comparison_operator(token)
        self.comparison_value: sqlparse.sql.Token = MLPredicate.get_comparison_value(token)

    @staticmethod
    def get_model_path(ml_predicate: sqlparse.sql.Comparison) -> sqlparse.sql.Token:
        ml_func: sqlparse.sql.Function = ml_predicate.left
        args: sqlparse.sql.IdentifierList = ml_func.token_next(0)[1].token_next(0)[1]
        return args[0]

    @staticmethod
    def get_comparison_operator(ml_predicate: sqlparse.sql.Comparison) -> sqlparse.sql.Token:
        return ml_predicate.token_next(0)[1]

    @staticmethod
    def get_comparison_value(ml_predicate: sqlparse.sql.Comparison) -> sqlparse.sql.Token:
        return ml_predicate.right

    def get_model_path_str(self) -> str:
        return self.model_path.value[1:-1]

    def get_comparison_operator_str(self) -> str:
        return self.comparison_operator.value

    def get_comparison_value_number(self) -> Union[int, float]:
        if self.comparison_value.ttype == sqlparse.tokens.Number.Integer:
            return int(self.comparison_value.value)
        if self.comparison_value.ttype == sqlparse.tokens.Number.Float:
            return float(self.comparison_value.value)
        raise Exception(f'illegal comparison value type: {self.comparison_value.ttype}')

    def get_func(self) -> Callable[[Any], bool]:
        o = self.get_comparison_operator_str()
        v = self.get_comparison_value_number()
        if o == '>':
            return lambda x : x > v
        if o == '>=':
            return lambda x : x >= v
        if o == '<':
            return lambda x : x < v
        if o == '<=':
            return lambda x : x <= v
        if o == '=':
            return lambda x : x == v

    def rewrite(self) -> str:
        self.model_path.value = f"\'out_{self.get_model_path_str()}\'"
        self.comparison_operator.value = '>'
        self.comparison_value.value = '0.5'
        self.comparison_value.ttype = sqlparse.tokens.Number.Float


def parse(sql: str, func_name: str) -> Tuple[sqlparse.sql.Statement, List[MLPredicate]]:
    parsed = sqlparse.parse(sql)[0]
    where = None
    for token in parsed.tokens:
        if isinstance(token, sqlparse.sql.Where):
            where = token
            break
    if where is None:
        return parsed, []
    ml_precidates = []
    for token in where.tokens:
        if isinstance(token, sqlparse.sql.Comparison) and \
            isinstance(token.left, sqlparse.sql.Function) and \
            token.left.tokens[0].value == func_name and \
            token.right.ttype in sqlparse.tokens.Number:
            ml_precidates.append(MLPredicate(token))
            logger.info(f'find an ml predicate: [{token}]')
    return parsed, ml_precidates
