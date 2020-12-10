import sys
sys.path.append('..\\HeuOptDemos_Eva')


from pymhlib.solution import Solution
from pymhlib.demos.misp import MISPInstance, MISPSolution
import random
import numpy as np
import logging
from copy import deepcopy


logger_step = logging.getLogger("step-by-step")


### methods specifically for misp solutions

def update_solution(sol: Solution, sel_elem):

    sol.x[sol.sel] = sel_elem
    sol.sel += 1
    sol.element_added_delta_eval()
    sol.sort_sel()


def copy_empty(sol: Solution):

    greedy_sol = MISPSolution(deepcopy(sol.inst))
    greedy_sol.obj_val_valid = False
    return greedy_sol


def is_complete_solution(sol: Solution):

    return not sol.may_be_extendible()


def candidate_list(sol: Solution):
    # get all selected nodes and their neighbors
    covered = set(sol.x[:sol.sel])
    for n in sol.x[:sol.sel]:
        covered.update(sol.inst.graph.neighbors(n))

    # for each uncovered node get the number of uncovered neighbors (Restknotengrad)
    candidates = {i:len(set(sol.inst.graph.neighbors(i)) - covered) for i,n in  enumerate(sol.covered) if n == 0}
    return candidates


def restricted_candidate_list(sol: Solution, cl: dict(), par):

    rcl = list()

    if type(par) == int:
        candidates = {k:v for k,v in cl.items()}
        k = min(len(candidates),par)
        for i in range(k):
            key = min(candidates, key=candidates.get)
            rcl.append(key)
            candidates.pop(key)

    if type(par) == float:
        minimum = min(cl.values())
        rcl = [k for k,v in cl.items() if v <= minimum * par]
 
    return np.array(rcl)


### this method can be used by any solution class, provided they implement the 
### methods 'copy_empty', 'candidate_list', 'restricted_candidate_list' and 'update_solution'
def greedy_randomized_construction(sol: Solution, par, _result):

    greedy_sol = copy_empty(sol)

    while not is_complete_solution(greedy_sol):

        log = f'SOL: {greedy_sol}\n'

        cl = candidate_list(greedy_sol)
        rcl = restricted_candidate_list(greedy_sol, cl, par)
        sel = random.choice(rcl)

        update_solution(greedy_sol,sel)

        log += f'CL: {cl}\nPAR: {par}\nRCL: {rcl}\nSEL: {sel}'
        logger_step.info(log)

    sol.copy_from(greedy_sol)
    sol.obj()



# only used for debugging
if __name__ == '__main__':
    handler = logging.StreamHandler(sys.stdout)
    logger_step.handlers = []
    logger_step.addHandler(handler)
    logger_step.setLevel(logging.INFO)
    initial = MISPSolution(MISPInstance('gnm-30-60-1'))
    logger_step.info(initial.covered)
    greedy_randomized_construction(initial,3,None)



    

