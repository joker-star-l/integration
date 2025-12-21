from __future__ import annotations
from typing import Type
import re
import importlib
import copy

from craftsman.base.chains import PrepChain
from craftsman.model.base_model import SQLModel
from craftsman.base.defs import MODEL_PACKAGE_PATH, SQLPlanType
from craftsman.base.operator import Operator

class PrepGraph(object):

    def __init__(self, input_features: list[str] = None, pipeline: dict = None) -> None:
        self.model: Type[SQLModel]
        self.chains: dict[str, PrepChain] = {}
        self.implements: dict[str, list] = {}
        self.join_operators: list[Operator] = []
        if pipeline is not None:
            self.__build_graph(input_features, pipeline)

    def __camel_to_snake(self, name: str):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def __build_graph(self, input_features: list[str], pipeline: dict) -> None:
        model_name = pipeline['model']['model_name']
        module_name = self.__camel_to_snake(model_name)
        transform_module = importlib.import_module(MODEL_PACKAGE_PATH + module_name)
        model_class = getattr(transform_module, model_name + 'SQLModel')
        self.model = model_class(pipeline['model']['trained_model']) 
        
        for feature in input_features:
            self.chains[feature] = PrepChain(feature, pipeline)
            self.implements[feature] = [SQLPlanType.CASE] * len(self.chains[feature].prep_operators)
        
    def get_empty_chains_graph(self) -> PrepGraph:
        """construct a new graph with empty featrues preprocessing chains

        Returns:
            PrepGraph: new graph
        """
        
        new_graph = PrepGraph()
        new_graph.model = copy.deepcopy(self.model)
        new_graph.join_operators = copy.deepcopy(self.join_operators)
        for feature, _ in self.chains.items():
            new_graph.chains[feature] = PrepChain(feature)
            new_graph.implements[feature] = []

        return new_graph
    
    def copy_graph(self) -> PrepGraph:
        """construct a new graph by clone

        Returns:
            PrepGraph: new graph
        """
        
        new_graph = PrepGraph()
        new_graph.model = copy.deepcopy(self.model)
        new_graph.join_operators = copy.deepcopy(self.join_operators)
        for feature, _ in self.chains.items():
            new_graph.chains[feature] = self.chains[feature].copy()
            new_graph.implements[feature] = copy.deepcopy(self.implements[feature])

        return new_graph
    
    def copy_prune_graph(self, cur_feature) -> PrepGraph:
        """construct a new graph by clone

        Returns:
            PrepGraph: new graph
        """
        
        new_graph = PrepGraph()
        
        # new_graph.model = copy.deepcopy(self.model)
        new_graph.model = self.model
        
        
        # new_graph.join_operators = copy.deepcopy(self.join_operators)
        new_graph.join_operators = self.join_operators
        
        for feature, _ in self.chains.items():
            if feature == cur_feature:
                new_graph.chains[feature] = self.chains[feature].copy_prune()
                new_graph.implements[feature] = copy.deepcopy(self.implements[feature])
            else:
                new_graph.chains[feature] = self.chains[feature]
                new_graph.implements[feature] = self.implements[feature]
        

        return new_graph
    
    def add_join_operator(self, op):
        self.join_operators.append(op)