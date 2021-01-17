import enum
from numpy.lib.function_base import gradient
#needed for overwriting gnvs
from pymhlib.solution import Solution
from pymhlib.scheduler import Result
from typing import List
import time

from pymhlib.demos.maxsat import MAXSATInstance, MAXSATSolution
from pymhlib.demos.misp import MISPInstance, MISPSolution
from pymhlib.gvns import GVNS, Method
import pymhlib.demos as demos
import logging
import os
from abc import ABC, abstractmethod
from pymhlib.solution import Solution
import random
import numpy as np
import logging
from copy import deepcopy

logger_step = logging.getLogger("step-by-step")
logger_step.setLevel(logging.INFO)
demo_data_path = os.path.dirname(demos.__file__) + os.path.sep + 'data'

# extend enums as needed, they hold the string values which are used for representation in widgets
class Problem(enum.Enum):
    MAXSAT = 'MAX-SAT'
    MISP = 'MAX-Independent Set'

class Algorithm(enum.Enum):
    GVNS = 'GVNS'
    GRASP = 'GRASP'

class Option(enum.Enum):
    CH = 'Initial Solution'
    LI = 'Local Improvement'
    SH = 'Shaking'
    RGC = 'Randomized Greedy Construction'

class InitSolution(enum.Enum):
    random = 0
    greedy = 1


class ProblemDefinition(ABC):
    """A base class for problem definition to store and retrieve available algorithms and options and problem specific solution instances.

    Attributes
        - name: name of the problem, has to correspond to name of directory where logs/instances are stored
        - options: dict of dicts of available algorithms and their corresponding available options/methodes, which have to be stored as tuples of the form: 
                    (<name for widget> , <function callback with signature (solution: Solution, par: Any, result: Result) or None>  , <type of additional parameter, fixed default or None>)
    """
    def __init__(self, name: str, options: dict):
        self.name = name
        self.options = options

    def get_algorithms(self):
        return [k.value for k,_ in self.options.items()]

    def get_options(self, algo: Algorithm):
        options = {}
        for o,m in self.options[algo].items():
            options[o] = [(t[0],t[2]) for t in m]
        return options

    @abstractmethod
    def get_solution(self, instance_path: str):
        pass

    def get_instances(self,visualisation):
        instance_path = "instances" + os.path.sep
        if os.path.isdir(instance_path + self.name):
            return os.listdir(instance_path + self.name)
        return []
    

    def get_method(self, algo:Algorithm, opt: Option, name: str, par):
        m = [t for t in self.options[algo][opt] if t[0] == name][0]
        method = Method(f'{opt.name.lower()}{par if type(m[2]) == type else m[2]}', m[1], par if type(m[2]) == type else m[2])
        return method


    def greedy_randomized_construction(self, sol: Solution, par, _result):

        greedy_sol = self.copy_empty(sol)

        while not self.is_complete_solution(greedy_sol):
            #sol_str = '[ ' + ' '.join([str(s) for s in greedy_sol.x[:greedy_sol.sel]])+ ' ]' #TODO has to hold for all kind of solutions
            sol_str = str(greedy_sol).replace('\n', ' ')
            logger_step.info(f'SOL: {sol_str}')

            cl = self.candidate_list(greedy_sol)
            rcl = self.restricted_candidate_list(greedy_sol, cl, par)
            sel = random.choice(rcl)
            self.update_solution(greedy_sol,sel)

            logger_step.info(f'CL: {cl}\nPAR: {par}')
            rcl_str = '[ ' + ' '.join([str(r) for r in rcl]) + ' ]'
            logger_step.info(f'RCL: {rcl_str}')
            logger_step.info(f'SEL: {sel}')

        sol.copy_from(greedy_sol)
        sol.obj()

    @abstractmethod
    def copy_empty(self,sol: Solution):
        pass

    @abstractmethod
    def is_complete_solution(self, sol: Solution):
        pass

    @abstractmethod
    def candidate_list(self, sol: Solution):
        pass

    @abstractmethod
    def restricted_candidate_list(self, sol: Solution, cl: dict, par):
        pass

    @abstractmethod
    def update_solution(self, sol: Solution, sel):
        pass

    def construct_greedy(self, sol: Solution, par, _result):
        self.greedy_randomized_construction(sol, 1, _result)



class MAXSAT(ProblemDefinition):

    def __init__(self):

        options = {Algorithm.GVNS: {
                                Option.CH: [(InitSolution.random.name, MAXSATSolution.construct, InitSolution.random.value)
                                            ,(InitSolution.greedy.name, self.construct_greedy, InitSolution.greedy.value)
                                            ],
                                Option.LI: [('k-flip neighborhood search', MAXSATSolution.local_improve, int)],
                                Option.SH: [('k random flip', MAXSATSolution.shaking, int)]
                                },
                    Algorithm.GRASP: {
                                Option.CH: [('random', MAXSATSolution.construct, 0)],   #not needed for grasp
                                Option.LI: [('k-flip neighborhood search', MAXSATSolution.local_improve, int)],
                                Option.RGC: [('k-best', self.greedy_randomized_construction, int),('alpha', self.greedy_randomized_construction, float)]
                                }
                    }

        super().__init__(Problem.MAXSAT.name.lower(), options)

    def get_solution(self, instance_path: str):
        instance = MAXSATInstance(instance_path)
        return MAXSATSolution(instance)

    def get_instances(self,visualisation):
        if visualisation:
            return super().get_instances(True)
        else: 
            instances = os.listdir(demo_data_path)
            return [i for i in instances if i[-3:] == 'cnf']


    def update_solution(self, sol: MAXSATSolution, sel):
        sol.x[abs(sel)-1] = False if sel < 0 else True

        #replace fulfilled clauses in greedy solution by empty array (solution needs to be a deep copy from original sol object!)
        clauses = [np.array([]) if sel in c else c for c in sol.inst.clauses]
        sol.inst.clauses = clauses


    def copy_empty(self, sol: MAXSATSolution):
        greedy_sol = deepcopy(sol)
        greedy_sol.x = np.full([sol.inst.n], -1, dtype=int)
        greedy_sol.obj_val_valid = False
        return greedy_sol

    def is_complete_solution(self, sol: MAXSATSolution):
        return not (-1 in sol.x)

    def candidate_list(self, sol: MAXSATSolution):
        candidates = dict()

        for v in [i for i,v in enumerate(sol.x,start=1) if v == -1]:
            candidates[v] = 0
            candidates[-v] = 0
            for c in sol.inst.variable_usage[v-1]:
                clause = sol.inst.clauses[c]
                if len(clause) == 0:
                    continue
                candidates[v] = candidates[v] + (v in clause)
                candidates[-v] = candidates[-v] + (-v in clause)

        return candidates


    def restricted_candidate_list(self, sol: Solution, cl: dict(), par):

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


    

class MISP(ProblemDefinition):

    def __init__(self):

        options = {Algorithm.GVNS: {
                                Option.CH: [(InitSolution.random.name, MISPSolution.construct, InitSolution.random.value)
                                            ,(InitSolution.greedy.name, self.construct_greedy, InitSolution.greedy.value)
                                            ],
                                Option.LI: [('two-exchange random fill neighborhood search', MISPSolution.local_improve, 2)],
                                Option.SH: [('remove k and random fill', MISPSolution.shaking, int)]
                                }
                                ,
                    Algorithm.GRASP: {
                                Option.CH: [('random', MISPSolution.construct, 0)],   #not needed for grasp
                                Option.LI: [('two-exchange random fill neighborhood search', MISPSolution.local_improve, 2)],
                                Option.RGC: [('k-best', self.greedy_randomized_construction, int),('alpha', self.greedy_randomized_construction, float)]
                              }
                    }

        super().__init__(Problem.MISP.name.lower(), options)

    def get_solution(self, instance_path):
        file_path = instance_path
        if instance_path.startswith('random'):
            file_path = "gnm" + instance_path[6:]
        instance = MISPInstance(file_path)
        return MISPSolution(instance)


    def get_instances(self, visualisation):
        if visualisation:
            inst = super().get_instances(True)
            return inst + ['random']
        else: 
            instances = os.listdir(demo_data_path)
            return [i for i in instances if i[-3:] == 'mis']

    
    def update_solution(self, sol: Solution, sel_elem):

        sol.x[sol.sel] = sel_elem
        sol.sel += 1
        sol.element_added_delta_eval()
        sol.sort_sel()


    def copy_empty(self, sol: Solution):

        greedy_sol = MISPSolution(deepcopy(sol.inst))
        greedy_sol.obj_val_valid = False
        return greedy_sol


    def is_complete_solution(self, sol: Solution):

        return not sol.may_be_extendible()


    def candidate_list(self, sol: Solution):
        # get all selected nodes and their neighbors
        covered = set(sol.x[:sol.sel])
        for n in sol.x[:sol.sel]:
            covered.update(sol.inst.graph.neighbors(n))

        # for each uncovered node get the number of uncovered neighbors
        candidates = {i:len(set(sol.inst.graph.neighbors(i)) - covered) for i,n in  enumerate(sol.covered) if n == 0}
        return candidates


    def restricted_candidate_list(self, sol: Solution, cl: dict, par):

        rcl = list()

        if type(par) == int:
            candidates = {k:v for k,v in cl.items()}
            k = min(len(candidates),par)
            for i in range(k):
                key = min(candidates, key=candidates.get)
                rcl.append(key)
                candidates.pop(key)

        if type(par) == float:
            mini = min(cl.values())
            maxi = max(cl.values())
            rcl = [k for k,v in cl.items() if v <= mini + par * (maxi-mini)]
    
        return np.array(rcl)



######################## overwrite GVNS to log necessary information 

class MyGVNS(GVNS):

    def __init__(self, sol: Solution, meths_ch: List[Method], meths_li: List[Method], meths_sh: List[Method],
                 own_settings: dict = None, consider_initial_sol=False,logger_step=True):
        super().__init__(sol, meths_ch, meths_li, meths_sh, own_settings, consider_initial_sol)
        self.logger_step = False
        if logger_step:
            self.logger_step = logging.getLogger("step-by-step")

    def perform_method(self, method: Method, sol: Solution, delayed_success=False) -> Result:
        """Perform method on given solution and returns Results object.

        Also updates incumbent, iteration and the method's statistics in method_stats.
        Furthermore checks the termination condition and eventually sets terminate in the returned Results object.

        :param method: method to be performed
        :param sol: solution to which the method is applied
        :param delayed_success: if set the success is not immediately determined and updated but at some later
                call of delayed_success_update()
        :returns: Results object
        """
        res = Result()
        obj_old = sol.obj()
        t_start = time.process_time()
        ##### logging for visualisation
        if self.logger_step:
            sol_str, inc_str = f'{sol}'.replace('\n',''), f'{self.incumbent}'.replace('\n','') # necessary because of automatic line breaks in str of numpy array
            step_info = f'START\nSOL: {sol_str}\nOBJ: {obj_old}\nM: {method.name}\nPAR: {method.par}\nINC: {inc_str}\nBEST: {self.incumbent.obj()}'
            self.logger_step.info(step_info)
        #################
        method.func(sol, method.par, res)
        t_end = time.process_time()
        if __debug__ and self.own_settings.mh_checkit:
            sol.check()
        ms = self.method_stats[method.name]
        ms.applications += 1
        ms.netto_time += t_end - t_start
        obj_new = sol.obj()
        if not delayed_success:
            ms.brutto_time += t_end - t_start
            if sol.is_better_obj(sol.obj(), obj_old):
                ms.successes += 1
                ms.obj_gain += obj_new - obj_old
        self.iteration += 1
        new_incumbent = self.update_incumbent(sol, t_end - self.time_start)
        ##### logging for visualisation
        if self.logger_step:
            sol_str, inc_str = f'{sol}'.replace('\n',''), f'{self.incumbent}'.replace('\n','') 
            step_info = f'END\nSOL: {sol_str}\nOBJ: {sol.obj()}\nM: {method.name}\nPAR: {method.par}\nINC: {inc_str}\nBEST: {self.incumbent.obj()}\nBETTER: {new_incumbent}'
            self.logger_step.info(step_info)
        #################
        terminate = self.check_termination()
        self.log_iteration(method.name, obj_old, sol, new_incumbent, terminate, res.log_info)
        if terminate:
            self.run_time = time.process_time() - self.time_start
            res.terminate = True
        return res


    def gvns(self, sol: Solution):
        """Perform general variable neighborhood search (GVNS) to given solution."""
        sol2 = sol.copy()
        if self.vnd(sol2) or not self.meths_sh:
            return
        use_vnd = bool(self.meths_li)
        while True:
            for m in self.next_method(self.meths_sh, repeat=True):
                t_start = time.process_time()
                res = self.perform_method(m, sol2, delayed_success=use_vnd)
                terminate = res.terminate
                if not terminate and use_vnd:
                    terminate = self.vnd(sol2)
                self.delayed_success_update(m, sol.obj(), t_start, sol2)
                ### log an entire cycle e.g. sh+li, rgc+li
                if self.logger_step:
                    self.logger_step.info('END_ITER')
                #########
                if sol2.is_better(sol):
                    sol.copy_from(sol2)
                    if terminate or res.terminate:
                        return
                    break
                else:
                    if terminate or res.terminate:
                        return
                    sol2.copy_from(sol)
            else:
                break

