from pymhlib.solution import Solution
import random
import numpy as np
import logging


logger_step = logging.getLogger("step-by-step")



def greedy_randomized_construction(sol: Solution, par, _result):


    c_sol = sol.copy()
    x = np.full([c_sol.inst.n], -1, dtype=int)
    variables = [i for i in range(1, c_sol.inst.n+1)] # contains unselected variables
    clauses = [[x for x in c] for c in c_sol.inst.clauses] # contains unfulfilled clauses

    for i in range(c_sol.inst.n):
        candidates = {}
        for v in variables: # maybe use variable_usage instead
            pos = sum([1 for i in clauses if v in i])
            neg = sum([1 for i in clauses if -v in i])
            candidates[v], candidates[-v] = pos, neg

        logger_step.info(f'CL: {candidates}')
        logger_step.info(f'X_START: {x}')


        rcl = restricted_candidate_list(candidates, par)
        logger_step.info(f'RCL: {rcl}')

        e = random.choice(rcl)      # choose random element from rcl
        logger_step.info(f'SEL: {e}')
        

        x[abs(e)-1] = 0 if e < 0 else 1   # add e to solution
        clauses = [c for c in clauses if e not in c] # remove fulfilled clauses

        variables.remove(abs(e))        # remove e from unselected variables

    sol.x = x
    sol.obj_val_valid = False


def restricted_candidate_list(candidates: dict(), par):

    rcl = np.array([], dtype=int)
    cand = {k:v for k,v in candidates.items()}
    if type(par) == int:

        k = min(len(cand),par)
        for i in range(k):
            key = max(cand, key=cand.get)
            rcl = np.append(rcl, key)
            cand.pop(key)
        return rcl

    if type(par) == float:
        maximum = max(candidates.values())
        rcl = np.array([k for k,v in candidates.items() if v >= maximum * par],dtype=int)
        return rcl
