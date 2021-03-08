"""Demo application solving the symmetric traveling salesman problem.

Given n cities and a symmetric distance matrix for all city pairs, find a shortest round trip through all cities.
"""

import random
import math
from typing import Tuple, Any
import numpy as np
import itertools

from pymhlib.permutation_solution import PermutationSolution
from pymhlib.solution import TObj
from pymhlib.ts_helper import TabuList

class TSPInstance:
    """An instance of the traveling salesman problem.

    This instance contains the distances between all city pairs.
    Starting from a solution in which the cities are visited in the order they are defined in the instance file,
    a local search in a 2-opt neighborhood using edge exchange is performed.

    Attributes
        - n: number of cities, i.e., size of incidence vector
        - distances: square matrix of integers representing the distances between two cities;
            zero means there is not connection between the two cities
    """

    def __init__(self, file_name: str):
        """Read an instance from the specified file."""
        self.coordinates = {}
        dimension = None

        with open(file_name, "r") as f:
            for line in f:
                if line.startswith("NAME") or line.startswith("COMMENT") or line.startswith("NODE_COORD_SECTION"):
                    pass
                elif line.startswith("EOF"):
                    break
                elif line.startswith("TYPE"):
                    assert line.split()[-1] == "TSP"
                elif line.startswith("EDGE_WEIGHT_TYPE"):
                    assert line.split()[-1] == "EUC_2D"
                elif line.startswith("DIMENSION"):
                    dimension = int(line.split()[-1])
                else:
                    split_line = line.split()
                    num = int(split_line[0]) - 1  # starts at 1
                    x = int(split_line[1])
                    y = int(split_line[2])

                    self.coordinates[num] = (x, y)

        assert len(self.coordinates) == dimension

        # building adjacency matrix
        distances = np.zeros((dimension, dimension))

        for i in range(0, dimension):
            for j in range(i + 1, dimension):
                x1, y1 = self.coordinates[i]
                x2, y2 = self.coordinates[j]
                dist = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
                distances[i][j] = distances[j][i] = int(dist)

        self.distances = distances
        self.n = dimension

        # make basic check if instance is meaningful
        if not 1 <= self.n <= 1000000:
            raise ValueError(f"Invalid n: {self.n}")

    def __repr__(self):
        """Write out the instance data."""
        return f"n={self.n},\ndistances={self.distances!r}\n"


class TSPSolution(PermutationSolution):
    """Solution to a TSP instance.

    Attributes
        - inst: associated TSPInstance
        - x: order in which cities are visited, i.e., a permutation of 0,...,n-1
    """

    to_maximize = False

    def __init__(self, inst: TSPInstance):
        super().__init__(inst.n, inst=inst)
        self.obj_val_valid = False
        self.k_opt_segment_starts_dict = {}

    def copy(self):
        sol = TSPSolution(self.inst)
        sol.copy_from(self)
        return sol

    def calc_objective(self):
        distance = 0
        for i in range(self.inst.n - 1):
            distance += self.inst.distances[self.x[i]][self.x[i + 1]]
        distance += self.inst.distances[self.x[-1]][self.x[0]]
        return distance

    def check(self):
        """Check if valid solution.

        :raises ValueError: if problem detected.
        """
        if len(self.x) != self.inst.n:
            raise ValueError("Invalid length of solution")
        super().check()

    def construct(self, par, _result): # todo add greedy initialization
        """Scheduler method that constructs a new solution.

        Here we just call initialize.
        """
        self.initialize(par)

    def shaking(self, par, result):
        """Scheduler method that performs shaking by 'par'-times swapping a pair of randomly chosen cities."""
        for _ in range(par):
            a = random.randint(0, self.inst.n - 1)
            b = random.randint(0, self.inst.n - 1)
            self.x[a], self.x[b] = self.x[b], self.x[a]
        self.invalidate()
        result.changed = True

    def local_improve(self, par, _result):
        """2-opt local search."""
        assert(par>1)

        if par < 3:
            self.two_opt_neighborhood_search(True)
        else:
            self.k_opt(par)


    def k_opt(self, k):
        """Best improvement k-opt local search."""
        permutations = list(itertools.permutations(list(range(1, k)))) 
        possible_flips = int(math.pow(2, k - 1))

        if not k in self.k_opt_segment_starts_dict:
            self.k_opt_segment_starts_dict[k] = self.all_k_opt_segment_starts(k, 0, self.inst.n)
    
        best = (None, None, None, 0) # (segment_starts, permutation, flips, delta)

        for segment_starts in self.k_opt_segment_starts_dict[k]:           
            #for i in range(math.factorial(k - 1)): # iterate over possible segment permutations with fixed initial segment
            for permutation in permutations: # iterate over possible segment permutations with fixed initial segment
                for j in range(possible_flips): # iterate over all possible segment flips with fixed initial segment
                    # i.e. for flips = [1, 0, 0, 1] reverse the second and fifth segment
                    flips = [(j >> l) & 1 for l in range(possible_flips.bit_length() - 1)]
                    delta = self.k_opt_move_delta_eval(segment_starts, permutation, flips)
                    if delta < best[3]:
                        best = (segment_starts, permutation, flips, delta)
                    #todo best improvement/first improvement handling
        if best[3] < 0:
            self.apply_k_opt_move(self.k_opt_segments(k, best[0]), best[1], best[2], best[3])


    def k_opt_segments(self, k, segment_starts):
        segments = []
        for i in range(k - 1):
            segments.append(self.x[segment_starts[i]:segment_starts[i+1]])
        segments.append(np.concatenate((self.x[segment_starts[k-1]:], self.x[:segment_starts[0]])))
        return segments

    def all_k_opt_segment_starts(self, k: int, j: int, n: int):
        """todo"""

        if k == 0:
            return [[]]

        segment_starts = []
        for i in range(j, n):
            if i > 0 or j > 0:
                combinations = [([i] + rest) for rest in self.all_k_opt_segment_starts(k-1, i + 2, n)]
            else:
                combinations = [([i] + rest) for rest in self.all_k_opt_segment_starts(k-1, i + 2, n - 1)]
            
            for comb in combinations:
                segment_starts.append(comb)
        return segment_starts


    def k_opt_move_delta_eval(self, segment_starts, permutation, flips):
        k = len(segment_starts) # (1, 3, 2)
        segment_ends = []
        for i in range(k):
            segment_ends.append((segment_starts[(i+1)%k] - 1) % self.inst.n)
        
        old_distance = 0 # will hold the sum of the length of the edges that will be deleted
        for i in range(k):
            old_distance += self.inst.distances[ self.x[segment_ends[i]] ][ self.x[segment_starts[(i+1) % k]] ]
        
        permutation = [0] + list(permutation) # fixed initial segment
        new_segment_starts = [segment_starts[i] for i in permutation]
        new_segment_ends = [segment_ends[i] for i in permutation]

        flips = [0] + flips # fixed initial segment
        for i in range(k): # flip
            if flips[i]:
                new_segment_starts[i], new_segment_ends[i] = new_segment_ends[i], new_segment_starts[i]
        
        new_distance = 0
        for i in range(k):
            new_distance += self.inst.distances[self.x[new_segment_ends[i]]][self.x[new_segment_starts[(i+1) % k]]]

        return new_distance - old_distance


    def apply_k_opt_move(self, segments, permutation, flips, delta=None):
        k = len(segments)
        permutation = [0] + list(permutation)
        flips = [0] + flips
        self.x = np.array([],dtype=int)

        segments = [segments[i] for i in permutation]
        for i in range(k):
            if flips[i]:
                segments[i] = list(reversed(segments[i]))
            self.x = np.concatenate((self.x, segments[i]))

        if delta is None:
            self.invalidate()
        else:
            self.obj_val += delta

    def two_opt_move_delta_eval(self, p1: int, p2: int) -> int:
        """ This method performs the delta evaluation for inverting self.x from position p1 to position p2.

        The function returns the difference in the objective function if the move would be performed,
        the solution, however, is not changed.
        """
        assert p1 < p2
        n = len(self.x)
        if p1 == 0 and p2 == n - 1:
            # reversing the whole solution has no effect
            return 0
        prev = (p1 - 1) % n
        nxt = (p2 + 1) % n
        x_p1 = self.x[p1]
        x_p2 = self.x[p2]
        x_prev = self.x[prev]
        x_next = self.x[nxt]
        d = self.inst.distances
        delta = d[x_prev][x_p2] + d[x_p1][x_next] - d[x_prev][x_p1] - d[x_p2][x_next]
        return delta

    def random_move_delta_eval(self) -> Tuple[Any, TObj]:
        """Choose a random move and perform delta evaluation for it, return (move, delta_obj)."""
        return self.random_two_opt_move_delta_eval()

    def apply_neighborhood_move(self, move):
        """This method applies a given neighborhood move accepted by SA,
            without updating the obj_val or invalidating, since obj_val is updated incrementally by the SA scheduler."""
        self.apply_two_opt_move(*move)

    def crossover(self, other: 'TSPSolution') -> 'TSPSolution':
        """Perform edge recombination."""
        return self.edge_recombination(other)


    # not implemented GRASP and Tabu methods
    def greedy_randomized_construction(self, par, _result):
        raise NotImplementedError

    def copy_empty(self) -> 'Solution':
        raise NotImplementedError

    def is_complete_solution(self) -> bool:
        raise NotImplementedError

    def candidate_list(self) -> dict:
        raise NotImplementedError

    def restricted_candidate_list(self, cl: dict) -> np.array:
        """Selects best candidates from candidate list according to parameter k or alpha.

        k and alpha are passed in settings.mh_grc_k and settings.mh_grc_alpha.
        The decision if k or alpha should be use is passed in dsettings.mh_grc_par  (default=k)"""
        if settings.mh_grc_par:
            return self.restricted_candidate_list_k(cl, settings.mh_grc_k)
        else:
            return self.restricted_candidate_list_alpha(cl, settings.mh_grc_alpha)

    def restricted_candidate_list_k(self, cl: dict, par) -> np.array:
        raise NotImplementedError

    def restricted_candidate_list_alpha(self, cl: dict, par) -> np.array:
        raise NotImplementedError

    def update_solution(self, sel):
        raise NotImplementedError

    def construct_greedy(self, par, _result):
        raise NotImplementedError
        # greedy_sol = self.copy_empty()

        # while not greedy_sol.is_complete_solution():
            
        #     cl = greedy_sol.candidate_list()
        #     rcl = greedy_sol.restricted_candidate_list_k(cl, 1)
        #     sel = random.choice(rcl)
        #     greedy_sol.update_solution(sel)

        # self.copy_from(greedy_sol)
        # self.obj()


    def is_tabu(self, tabu_list: TabuList) -> bool:
        raise NotImplementedError


    def get_tabu_attribute(self, sol_old: 'Solution'):
        raise NotImplementedError



if __name__ == '__main__':
    from pymhlib.demos.common import run_optimization, data_dir
    from pymhlib.settings import get_settings_parser
    parser = get_settings_parser()
    run_optimization('TSP', TSPInstance, TSPSolution, data_dir + "xqf131.tsp")
