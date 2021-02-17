"""A generic class for solutions that are represented by fixed-length binary vectors."""

from abc import ABC
import random
from typing import Tuple
import numpy as np

from pymhlib.solution import VectorSolution, TObj
from pymhlib.ts_helper import TabuList


class BinaryVectorSolution(VectorSolution, ABC):
    """Abstract solution class with fixed-length 0/1 vector as solution representation.

    Attributes
        - x: 0/1 vector representing a solution
    """

    def __init__(self, length, **kwargs):
        """Initializes the solution vector with zeros."""
        super().__init__(length, dtype=bool, **kwargs)

    def dist(self, other: 'BinaryVectorSolution'):
        """Return Hamming distance of current solution to other solution."""
        return sum(np.logical_xor(self.x, other.x))

    def initialize(self, k):
        """Random initialization."""
        self.x = np.random.randint(0, 2, len(self.x), dtype=bool)
        self.invalidate()

    def k_random_flips(self, k):
        """Perform k random flips and call invalidate()."""
        for _ in range(k):
            p = random.randrange(self.inst.n)
            self.x[p] = not self.x[p]
        self.invalidate()

    def check(self):
        """Check if valid solution.

        Raises ValueError if problem detected.
        """
        super().check()
        for v in self.x:
            if not 0 <= v <= 1:
                raise ValueError("Invalid value in BinaryVectorSolution: {self.x}")

    def k_flip_neighborhood_search(self, k: int, best_improvement: bool, tabu_list: TabuList=None, incumbent: VectorSolution=None) -> bool:
        """Perform one major iteration of a k-flip local search, i.e., search one neighborhood.

        If best_improvement is set, the neighborhood is completely searched and a best neighbor is kept;
        otherwise the search terminates in a first-improvement manner, i.e., keeping a first encountered
        better solution.

        :returns: True if an improved solution has been found.
        """
        x = self.x
        assert 0 < k <= len(x)
        better_found = False
        best_sol = self.copy()
        next_best_sol = False
        perm = np.random.permutation(len(x))  # permutation for randomization of enumeration order
        p = np.full(k, -1)  # flipped positions
        # initialize
        i = 0  # current index in p to consider
        while i >= 0:
            # evaluate solution
            if i == k:

                sol_is_tabu = self.is_tabu(tabu_list)
                if self.is_better(best_sol):
                    if not best_improvement:
                        return True
                    if (sol_is_tabu and self.is_better(incumbent)) or not sol_is_tabu:
                        #tabu_list.delete(tabu_attr) #deletion of attribute in ts
                        best_sol.copy_from(self)
                        better_found = True
                elif tabu_list != None and not sol_is_tabu:
                    # the solution found is not better:
                    # when ts is used and the found solution is not tabu
                    # TODO tie breaking?
                    if not next_best_sol or self.is_better(next_best_sol):
                        #first round: copy in any case
                        next_best_sol = self.copy()


                i -= 1  # backtrack
            else:
                if p[i] == -1:
                    # this index has not yet been placed
                    p[i] = (p[i-1] if i > 0 else -1) + 1
                    self.flip_variable(perm[p[i]])
                    i += 1  # continue with next position (if any)
                elif p[i] < len(x) - (k - i):
                    # further positions to explore with this index
                    self.flip_variable(perm[p[i]])
                    p[i] += 1
                    self.flip_variable(perm[p[i]])
                    i += 1
                else:
                    # we are at the last position with the i-th index, backtrack
                    self.flip_variable(perm[p[i]])
                    p[i] = -1  # unset position
                    i -= 1
        if better_found:
            self.copy_from(best_sol)
            self.invalidate()
        elif tabu_list != None and next_best_sol:
            self.copy_from(next_best_sol)
            self.invalidate()
        return better_found

    def flip_variable(self, pos: int):
        """Flip the variable at position pos and possibly incrementally update objective value or invalidate.

        This generic implementation just calls invalidate() after flipping the variable.
        """
        self.x[pos] = not self.x[pos]
        self.invalidate()

    def flip_move_delta_eval(self, pos: int) -> TObj:
        """Determine delta in objective value when flipping position p.

        Here the solution is evaluated from scratch. If possible, it should be overloaded by a more
        efficient delta evaluation.
        """
        obj = self.obj()
        self.x[pos] = not self.x[pos]
        self.invalidate()
        delta = self.obj() - obj
        self.x[pos] = not self.x[pos]
        self.obj_val = obj
        return delta

    def random_flip_move_delta_eval(self) -> Tuple[int, TObj]:
        """Choose random move in the flip neighborhood and perform delta evaluation, returning (move, delta_obj).

        The solution is not changed here yet.
        Primarily used in simulated annealing.
        """
        p = random.randrange(len(self.x))
        delta_obj = self.flip_move_delta_eval(p)
        return p, delta_obj


    def is_tabu(self, tabu_list: TabuList):
        if tabu_list == None:
            return False
        solution = {i*-1 if v ==0 else i for i,v in enumerate(self.x,start=1)}
        for ta in tabu_list.tabu_list:
            if ta.attribute.issubset(solution):
                return True
        return False

    def get_tabu_attribute(self, sol_old: 'BinaryVectorSolution'):
        # tabu attribute is stored as set of positive/negative variables
        new_sol = {i*-1 if v ==0 else i for i,v in enumerate(self.x,start=1)}
        old_sol = {i*-1 if v ==0 else i for i,v in enumerate(sol_old.x,start=1)}
        diff = old_sol.difference(new_sol)
        if len(diff) == 0:
            return False
        return diff

