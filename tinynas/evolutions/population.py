# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import os
import sys
from abc import ABCMeta, abstractmethod

import numpy as np


class Population(metaclass=ABCMeta):

    def __init__(self, popu_size, budgets, logger):
        self.logger = logger
        self.budgets = budgets
        self.popu_size = popu_size
        self.init_population()
        self.logger.info('****** Successfully build the Population ******')

    def init_population(self, ):
        self.num_evaluated_nets_count = 0
        self.popu_structure_list = []
        self.popu_acc_list = []
        self.popu_score_list = []
        for key in self.budgets:
            temp_popu_list = list()
            setattr(self, f'popu_{key}_list', temp_popu_list)

    def update_population(self, model_info):
        if 'score' not in model_info.keys():
            raise NameError(
                'To update population, score must in the model_info')

        acc_temp = model_info['score']

        if len(self.popu_acc_list) > 0:
            insert_idx = len(
                self.popu_acc_list) if self.popu_acc_list[-1] < acc_temp else 0
            for idx, pupu_acc in enumerate(self.popu_acc_list):
                if pupu_acc >= acc_temp:
                    insert_idx = idx
        else:
            insert_idx = 0

        self.popu_structure_list.insert(insert_idx,
                                        model_info['structure_info'])
        self.popu_acc_list.insert(insert_idx, acc_temp)
        self.popu_score_list.insert(insert_idx, model_info['score'])
        for key in self.budgets:
            _popu_list = getattr(self, f'popu_{key}_list')
            _popu_list.insert(insert_idx, model_info[key])


    def rank_population(self, maintain_popu=False):
        # filter out the duplicate structure
        unique_structure_set = set()
        unique_idx_list = []
        for the_idx, the_strucure in enumerate(self.popu_structure_list):
            if str(the_strucure) in unique_structure_set:
                continue
            unique_structure_set.add(str(the_strucure))
            unique_idx_list.append(the_idx)

        # sort population list, pop the duplicate structure, and maintain the population
        # sort_idx = list(np.argsort(self.popu_acc_list))
        objectives = [self.popu_acc_list] + [getattr(self, f'popu_{key}_list') for key in self.budgets]
        sort_idx = nsga2_non_dominated_sort(objectives)

        sort_idx = sort_idx[::-1]
        temp_sort_idx = sort_idx[:]
        for idx in temp_sort_idx:
            if idx not in unique_idx_list:
                sort_idx.remove(idx)
        if maintain_popu:
            sort_idx = sort_idx[0:self.popu_size]

        self.popu_structure_list = [
            self.popu_structure_list[idx] for idx in sort_idx
        ]
        self.popu_acc_list = [self.popu_acc_list[idx] for idx in sort_idx]
        self.popu_score_list = [self.popu_score_list[idx] for idx in sort_idx]

        for key in self.budgets:
            popu_list_name = f'popu_{key}_list'
            temp_popu_list = getattr(self, popu_list_name)
            setattr(self, popu_list_name,
                    [temp_popu_list[idx] for idx in sort_idx])

    def gen_random_structure_net(self, ):
        pass

    def merge_shared_data(self, popu_nas_info, update_num=True):

        if isinstance(popu_nas_info, Population):
            self.popu_structure_list += popu_nas_info.popu_structure_list
            self.popu_acc_list += popu_nas_info.popu_acc_list
            self.popu_score_list += popu_nas_info.popu_score_list
            for key in self.budgets:
                popu_list_name = f'popu_{key}_list'
                righthand = getattr(popu_nas_info, popu_list_name)
                lefthand = getattr(self, popu_list_name)
                lefthand += righthand
                setattr(self, popu_list_name, lefthand)

        if isinstance(popu_nas_info, dict):
            if update_num:
                self.num_evaluated_nets_count = popu_nas_info[
                    'num_evaluated_nets_count']
            self.popu_structure_list += popu_nas_info['popu_structure_list']
            self.popu_acc_list += popu_nas_info['popu_acc_list']
            self.popu_score_list += popu_nas_info['popu_score_list']
            for key in self.budgets:
                popu_list_name = f'popu_{key}_list'
                righthand = popu_nas_info[f'popu_{key}_list']
                lefthand = getattr(self, popu_list_name)
                lefthand += righthand
                setattr(self, popu_list_name, lefthand)

        self.rank_population(maintain_popu=True)

    def export_dict(self, ):
        popu_nas_info = {}
        self.rank_population(maintain_popu=True)

        popu_nas_info[
            'num_evaluated_nets_count'] = self.num_evaluated_nets_count
        popu_nas_info['popu_structure_list'] = self.popu_structure_list
        popu_nas_info['popu_acc_list'] = self.popu_acc_list
        popu_nas_info['popu_score_list'] = self.popu_score_list
        for key in self.budgets:
            popu_nas_info[f'popu_{key}_list'] = getattr(
                self, f'popu_{key}_list')

        return popu_nas_info

    def get_individual_info(self, idx=0, is_struct=False):
        individual_info = {}
        self.rank_population(maintain_popu=True)

        if is_struct:
            individual_info['structure'] = self.popu_structure_list[idx]
        individual_info['acc'] = self.popu_acc_list[idx]
        individual_info['score'] = self.popu_score_list[idx]
        for key in self.budgets:
            individual_info[key] = getattr(self, f'popu_{key}_list')[idx]

        return individual_info


def non_dominated_sort(objectives):
    """
    Perform non-dominated sorting on a population based on a list of objectives.
    
    Parameters:
    objectives (list of list): A list where each sublist contains the values of a single objective for all individuals.
    
    Returns:
    list of lists: A list of non-dominated fronts, each represented as a list of indices.
    """
    # Transpose the list to have individuals with their objective values
    population_size = len(objectives[0])  # Number of individuals
    num_objectives = len(objectives)  # Number of objectives
    fronts = []
    dominated_count = np.zeros(population_size)
    dominated_set = [set() for _ in range(population_size)]
    rank = np.zeros(population_size)

    # Compare each individual to all others to determine dominance
    for p in range(population_size):
        for q in range(population_size):
            if p != q:
                # Check if individual p dominates individual q
                dominates_pq = all(objectives[i][p] <= objectives[i][q] for i in range(num_objectives)) and any(objectives[i][p] < objectives[i][q] for i in range(num_objectives))
                dominates_qp = all(objectives[i][q] <= objectives[i][p] for i in range(num_objectives)) and any(objectives[i][q] < objectives[i][p] for i in range(num_objectives))
                
                if dominates_pq:
                    dominated_set[p].add(q)
                elif dominates_qp:
                    dominated_count[p] += 1

        # If the individual is not dominated by anyone, it's part of the first front
        if dominated_count[p] == 0:
            rank[p] = 0
    
    # First front initialization
    front = [i for i in range(population_size) if rank[i] == 0]
    fronts.append(front)
    
    # Process subsequent fronts
    current_front = 0
    while len(fronts[current_front]) > 0:
        next_front = []
        for p in fronts[current_front]:
            for q in dominated_set[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    rank[q] = current_front + 1
                    next_front.append(q)
        current_front += 1
        fronts.append(next_front)

    return fronts


def crowding_distance(objectives, front):
    """
    Compute the crowding distance of each individual in the given front.
    
    Parameters:
    objectives (list of list): A list where each sublist contains the values of a single objective for all individuals.
    front (list): A list of indices corresponding to the individuals in the front.
    
    Returns:
    list: A list of crowding distances for each individual in the front.
    """
    num_objectives = len(objectives)
    front_size = len(front)
    crowding_dist = np.zeros(front_size)

    # Iterate over each objective and calculate the crowding distance
    for i in range(num_objectives):
        # Sort the front based on the i-th objective
        front_sorted = sorted(front, key=lambda x: objectives[i][x])
        # Assign extreme values to the boundaries
        crowding_dist[0] = crowding_dist[front_size - 1] = float('inf')
        
        for j in range(1, front_size - 1):
            # Calculate the crowding distance
            if objectives[i][front_sorted[front_size - 1]] == objectives[i][front_sorted[0]]:
                crowding_dist[j] = float('inf')
            else:
                crowding_dist[j] += (objectives[i][front_sorted[j + 1]] - objectives[i][front_sorted[j - 1]]) / (objectives[i][front_sorted[front_size - 1]] - objectives[i][front_sorted[0]])

    return crowding_dist 


def nsga2_non_dominated_sort(objectives):
    """
    Perform NSGA-II non-dominated sorting and return the indices sorted by non-domination levels.
    This includes diversity within the same rank using crowding distance.
    
    Parameters:
    objectives (list of list): A list where each sublist contains the values of a single objective for all individuals.
    
    Returns:
    list: A list of indices sorted by non-domination level and crowding distance.
    """
    fronts = non_dominated_sort(objectives)
    sorted_indices = []

    # Create a sorted list of indices based on non-domination rank
    for front in fronts:
        if len(front) > 1:
            # Calculate crowding distance for the front if it has more than one individual
            crowding_dist = crowding_distance(objectives, front)
            # Sort the front by crowding distance in descending order
            front_sorted = [x for _, x in sorted(zip(crowding_dist, front), reverse=True)]
            sorted_indices.extend(front_sorted)
        else:
            sorted_indices.extend(front)
    
    return sorted_indices



# def non_dominated_sort(objectives):
#     """
#     Perform non-dominated sorting on a population based on a list of objectives.
    
#     Parameters:
#     objectives (list of list): A list of lists where each sublist contains the objective values of an individual.
    
#     Returns:
#     list of lists: A list of non-dominated fronts, each represented as a list of indices.
#     """
#     population_size = len(objectives)
#     fronts = []
#     dominated_count = np.zeros(population_size)
#     dominated_set = [set() for _ in range(population_size)]
#     rank = np.zeros(population_size)

#     # Compare each individual to all others to determine dominance
#     for p in range(population_size):
#         for q in range(population_size):
#             if p != q:
#                 # Check if p dominates q or vice versa
#                 dominates_pq = all(objectives[p][i] <= objectives[q][i] for i in range(len(objectives[0]))) and any(objectives[p][i] < objectives[q][i] for i in range(len(objectives[0])))
#                 dominates_qp = all(objectives[q][i] <= objectives[p][i] for i in range(len(objectives[0]))) and any(objectives[q][i] < objectives[p][i] for i in range(len(objectives[0])))
                
#                 if dominates_pq:
#                     dominated_set[p].add(q)
#                 elif dominates_qp:
#                     dominated_count[p] += 1

#         # If the individual is not dominated, it's part of the first front
#         if dominated_count[p] == 0:
#             rank[p] = 0
    
#     # First front initialization
#     front = [i for i in range(population_size) if rank[i] == 0]
#     fronts.append(front)
    
#     # Process subsequent fronts
#     current_front = 0
#     while len(fronts[current_front]) > 0:
#         next_front = []
#         for p in fronts[current_front]:
#             for q in dominated_set[p]:
#                 dominated_count[q] -= 1
#                 if dominated_count[q] == 0:
#                     rank[q] = current_front + 1
#                     next_front.append(q)
#         current_front += 1
#         fronts.append(next_front)

#     return fronts


# def crowding_distance(objectives, front):
#     """
#     Compute the crowding distance of each individual in the given front.
    
#     Parameters:
#     objectives (list of list): A list of lists where each sublist contains the objective values of an individual.
#     front (list): A list of indices corresponding to the individuals in the front.
    
#     Returns:
#     list: A list of crowding distances for each individual in the front.
#     """
#     num_objectives = len(objectives[0])
#     front_size = len(front)
#     crowding_dist = np.zeros(front_size)

#     # Iterate over each objective and calculate the crowding distance
#     for i in range(num_objectives):
#         # Sort the front based on the i-th objective
#         front_sorted = sorted(front, key=lambda x: objectives[x][i])
#         # Assign extreme values to the boundaries
#         crowding_dist[0] = crowding_dist[front_size - 1] = float('inf')
        
#         for j in range(1, front_size - 1):
#             # Calculate the crowding distance
#             crowding_dist[j] += (objectives[front_sorted[j + 1]][i] - objectives[front_sorted[j - 1]][i]) / (objectives[front_sorted[front_size - 1]][i] - objectives[front_sorted[0]][i])

#     return crowding_dist


# def nsga2_non_dominated_sort(objectives):
#     """
#     Perform NSGA-II non-dominated sorting and return the indices sorted by non-domination levels.
#     This includes diversity within the same rank using crowding distance.
    
#     Parameters:
#     objectives (list of list): A list of lists where each sublist contains the objective values of an individual.
    
#     Returns:
#     list: A list of indices sorted by non-domination level and crowding distance.
#     """
#     fronts = non_dominated_sort(objectives)
#     sorted_indices = []

#     # Create a sorted list of indices based on non-domination rank
#     for front in fronts:
#         if len(front) > 1:
#             # Calculate crowding distance for the front if it has more than one individual
#             crowding_dist = crowding_distance(objectives, front)
#             # Sort the front by crowding distance in descending order
#             front_sorted = [x for _, x in sorted(zip(crowding_dist, front), reverse=True)]
#             sorted_indices.extend(front_sorted)
#         else:
#             sorted_indices.extend(front)
    
#     return sorted_indices
