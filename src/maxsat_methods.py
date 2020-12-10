import sys
sys.path.append('..\\HeuOptDemos_Eva')


from pymhlib.solution import Solution
from pymhlib.demos.maxsat import MAXSATSolution, MAXSATInstance
import random
import numpy as np
import logging
from copy import deepcopy


logger_step = logging.getLogger("step-by-step")


### methods specifically for maxsat solutions

def update_solution(sol: MAXSATSolution, sel):
    sol.x[abs(sel)-1] = False if sel < 0 else True

    #remove fulfilled clauses from greedy solution (solution needs to be a deep copy from original sol object!)
    clauses = []
    for c in sol.inst.clauses:
        if sel not in c:
            clauses.append(c)
    sol.inst.clauses = clauses

def copy_empty(sol: MAXSATSolution):
    #greedy_sol = MAXSATSolution(sol.inst)
    greedy_sol = deepcopy(sol)
    greedy_sol.x = np.full([sol.inst.n], -1, dtype=int)
    greedy_sol.obj_val_valid = False
    return greedy_sol

def is_complete_solution(sol: MAXSATSolution):
    return not (-1 in sol.x)

def candidate_list(sol: MAXSATSolution):
    candidates = dict()
    # find unfulfilled clauses
    #clauses = []
    #for c in [np.array(c, copy=True) for c in sol.inst.clauses]:
    #    for v in c:
    #        if sol.x[abs(v)-1] == (1 if v > 0 else 0):
    #            break
    #        else:
    #            clauses.append(c)

    for v in [i+1 for i,v in enumerate(sol.x) if v == -1]:
        pos = sum([1 for i in sol.inst.clauses if v in i])
        neg = sum([1 for i in sol.inst.clauses if -v in i])
        candidates[v], candidates[-v] = pos, neg

    return candidates


def restricted_candidate_list(sol: Solution, cl: dict(), par):

    rcl = list()

    if type(par) == int:
        candidates = {k:v for k,v in cl.items()}
        k = min(len(candidates),par)
        for i in range(k):
            key = max(candidates, key=candidates.get)
            rcl.append(key)
            candidates.pop(key)

    if type(par) == float:
        maximum = max(cl.values())
        rcl = [k for k,v in cl.items() if v >= maximum * par]
        
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
    initial = MAXSATSolution(MAXSATInstance('instances\maxsat\my_maxsat.cnf'))
    greedy_randomized_construction(initial,4, {})
    logger_step.info(initial.obj_val)
    logger_step.info(initial)
    logger_step.info(initial.inst.clauses)
    

