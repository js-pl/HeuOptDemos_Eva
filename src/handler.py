"""
handler module which provides information for widgets in interface module and uses widget input
to issue pymhlib calls
"""

import sys
sys.path.append('..\\HeuOptDemos_Eva')

from abc import ABC, abstractmethod
import enum
import os
import logging
import re
import numpy as np
import random
import ast

# pymhlib imports
from pymhlib.settings import settings, parse_settings
from pymhlib.log import init_logger
from pymhlib.demos.maxsat import MAXSATInstance, MAXSATSolution
from pymhlib.demos.misp import MISPInstance, MISPSolution
from pymhlib.gvns import GVNS, Method

#needed for overwriting gnvs
from pymhlib.solution import Solution
from pymhlib.scheduler import Result
from typing import List
import time

from src.maxsat_methods import greedy_randomized_construction as maxsat_greedy_randomized_construction
from src.misp_methods import greedy_randomized_construction as misp_greedy_randomized_construction
from src.logdata import get_log_data


if not settings.__dict__: parse_settings(args='')





# TODO: pymhlib settings
settings.mh_titer = 100
settings.mh_lfreq = 1
#settings.mh_tciter = 30
settings.mh_out = "logs" + os.path.sep + "summary.log"
settings.mh_log = "logs" + os.path.sep + "iter.log"
init_logger()
logger_step = logging.getLogger("step-by-step")
logger_step.setLevel(logging.INFO)
instance_path = "instances" + os.path.sep

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

    def get_instances(self):
        instance_path = "instances" + os.path.sep
        if os.path.isdir(instance_path + self.name):
            instances = os.listdir(instance_path + self.name)
        return instances
    

    def get_method(self, algo:Algorithm, opt: Option, name: str, par):
        m = [t for t in self.options[algo][opt] if t[0] == name][0]
        method = Method(f'{opt.name.lower()}{par if type(m[2]) == type else m[2]}', m[1], par if type(m[2]) == type else m[2])
        return method



class MAXSAT(ProblemDefinition):

    def __init__(self, name: str):

        options = {Algorithm.GVNS: {
                                Option.CH: [('random', MAXSATSolution.construct, 0)],
                                Option.LI: [('k-flip neighborhood search', MAXSATSolution.local_improve, int)],
                                Option.SH: [('k random flip', MAXSATSolution.shaking, int)]
                                },
                    Algorithm.GRASP: {
                                Option.CH: [('random', MAXSATSolution.construct, 0)],   #not needed for grasp
                                Option.LI: [('k-flip neighborhood search', MAXSATSolution.local_improve, int)],
                                Option.RGC: [('k-best', maxsat_greedy_randomized_construction, int),('alpha', maxsat_greedy_randomized_construction, float)]
                                }
                    }

        super().__init__(name, options)

    def get_solution(self, instance_path: str):
        instance = MAXSATInstance(instance_path)
        return MAXSATSolution(instance)


class MISP(ProblemDefinition):

    def __init__(self, name: str):

        options = {Algorithm.GVNS: {
                                Option.CH: [('random', MISPSolution.construct, 0)],
                                Option.LI: [('two-exchange random fill neighborhood search', MISPSolution.local_improve, 2)],
                                Option.SH: [('remove k and random fill', MISPSolution.shaking, int)]
                                }
                                ,
                    Algorithm.GRASP: {
                                Option.CH: [('random', MISPSolution.construct, 0)],   #not needed for grasp
                                Option.LI: [('two-exchange random fill neighborhood search', MISPSolution.local_improve, 2)],
                                Option.RGC: [('k-best', misp_greedy_randomized_construction, int),('alpha', misp_greedy_randomized_construction, float)]
                              }
                    }

        super().__init__(name, options)

    def get_solution(self, instance_path):
        instance = MISPInstance(instance_path)
        return MISPSolution(instance)

    def get_instances(self):
        inst = super().get_instances()
        if len(inst) == 0:
            return ['random']
        return inst





# initialize problems (parameter 'name' corresponds to name of directories for instances/logs)
problems = {
            Problem.MAXSAT: MAXSAT('maxsat'),
            Problem.MISP: MISP('misp')
            }


# methods used by module interface to extract information for widgets
def get_problems():
    return [p.value for p,_ in problems.items()]

def get_instances(prob: Problem):
    return problems[prob].get_instances()

def get_algorithms(prob: Problem):
    return problems[prob].get_algorithms()

def get_options(prob: Problem, algo: Algorithm):
    return problems[prob].get_options(algo)


def run_algorithm(options: dict):

    iter_fh = logging.FileHandler(f"logs/iter.log", mode="w")
    iter_logger = logging.getLogger("pymhlib_iter")
    iter_logger.handlers = []
    iter_logger.addHandler(iter_fh)
    

    fh = logging.FileHandler(f"logs/{options['prob'].name.lower()}/{options['algo'].name.lower()}/{options['algo'].name.lower()}.log", mode="w")
    logger_step.handlers = []
    logger_step.addHandler(fh)
    logger_step.info(f"{options['prob'].name}\n{options['algo'].name}")

    file_path = instance_path + problems[options['prob']].name + os.path.sep + options['inst']

    if options['inst'] == 'random': #only available for misp so far
        file_path = "gnm-50-70"

    # initialize solution for problem
    solution = problems[options['prob']].get_solution(file_path)

    # run specified algorithm
    if options['algo'] == Algorithm.GVNS:
        run_gvns(solution, options)
        return get_log_data(options['prob'].name.lower(), Algorithm.GVNS.name.lower()), solution.inst
    if options['algo'] == Algorithm.GRASP:
        run_grasp(solution, options)
        return get_log_data(options['prob'].name.lower(), Algorithm.GRASP.name.lower()), solution.inst
   
    return [], None


######################### overwrite GVNS to log necessary information 

class MyGVNS(GVNS):

    def __init__(self, logger_step, sol: Solution, meths_ch: List[Method], meths_li: List[Method], meths_sh: List[Method],
                 own_settings: dict = None, consider_initial_sol=False):
        super().__init__(sol, meths_ch, meths_li, meths_sh, own_settings, consider_initial_sol)
        self.logger_step = logger_step

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
        step_info = f'START\nSOL: {"[ "+" ".join([str(i) for i in sol.x[:sol.sel]]) +" ]"}\nOBJ: {obj_old}\nM: {method.name}\nPAR: {method.par}'
        self.logger_step.info(step_info)
        self.logger_step.info(f'INC: {"[ "+" ".join([str(i) for i in self.incumbent.x[:self.incumbent.sel]]) +" ]"}\nBEST: {self.incumbent.obj()}')
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
        step_info = f'END\nSOL: {"[ "+" ".join([str(i) for i in sol.x[:sol.sel]]) +" ]"}\nOBJ: {sol.obj()}\nM: {method.name}\nPAR: {method.par}'
        self.logger_step.info(step_info)
        self.logger_step.info(f'INC: {"[ "+" ".join([str(i) for i in self.incumbent.x[:self.incumbent.sel]]) +" ]"}\nBEST: {self.incumbent.obj()}\nBETTER: {new_incumbent}')
        terminate = self.check_termination()
        self.log_iteration(method.name, obj_old, sol, new_incumbent, terminate, res.log_info)
        if terminate:
            self.run_time = time.process_time() - self.time_start
            res.terminate = True
        return res


########################################


def run_gvns(solution, options: dict):


    prob = problems[options['prob']]
    ch = [ prob.get_method(Algorithm.GVNS, Option.CH, m[0], m[1]) for m in options[Option.CH] ]
    li = [ prob.get_method(Algorithm.GVNS, Option.LI, m[0], m[1]) for m in options[Option.LI] ]
    sh = [ prob.get_method(Algorithm.GVNS, Option.SH, m[0], m[1]) for m in options[Option.SH] ]
    
    ### for now, the overwritten GVNS is called
    alg = MyGVNS(logger_step, solution, ch, li, sh)

    alg.run()
    alg.method_statistics()
    alg.main_results()
    logging.getLogger("pymhlib_iter").handlers[0].flush()


def run_grasp(solution, options: dict):
    
    prob = problems[options['prob']]

    ch = []
    li = [ prob.get_method(Algorithm.GRASP, Option.LI, m[0], m[1]) for m in options[Option.LI] ]
    RGC = [ prob.get_method(Algorithm.GRASP, Option.RGC, m[0], m[1]) for m in options[Option.RGC] ]

    # for now the overwritten gvns is called
    alg = MyGVNS(logger_step, solution,ch,li,RGC,consider_initial_sol=True)
    alg.run()
    alg.method_statistics()
    alg.main_results()
    logging.getLogger("pymhlib_iter").handlers[0].flush()

    

# only used for debugging
if __name__ == '__main__':
        log_data, _ = run_algorithm({'prob':Problem.MISP, Option.CH:[('random',0)], 'inst':'random','algo':Algorithm.GRASP,
           Option.LI:[('two-exchange random fill neighborhood search',2)],
          Option.RGC:[('k-best',5)]})

        #print(log_data)





        















            

