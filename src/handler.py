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
import pandas as pd
import random
import ast

# pymhlib imports
from pymhlib.settings import settings, parse_settings
from pymhlib.log import init_logger
from pymhlib.demos.maxsat import MAXSATInstance, MAXSATSolution
from pymhlib.demos.misp import MISPInstance, MISPSolution
from pymhlib.gvns import GVNS, Method
import pymhlib.demos as demos

#needed for overwriting gnvs
from pymhlib.solution import Solution
from pymhlib.scheduler import Result
from typing import List
import time

from src.maxsat_methods import greedy_randomized_construction as maxsat_greedy_randomized_construction
from src.problems import Problem, Algorithm, Option
from src.misp_methods import greedy_randomized_construction as misp_greedy_randomized_construction
from src.logdata import get_log_data


if not settings.__dict__: parse_settings(args='')





# TODO: pymhlib settings
settings.mh_titer = 1000
settings.mh_lfreq = 1
#settings.mh_tciter = 30
settings.mh_out = "logs" + os.path.sep + "summary.log"
settings.mh_log = "logs" + os.path.sep + "iter.log"
init_logger()
logger_step = logging.getLogger("step-by-step")
logger_step.setLevel(logging.INFO)
instance_path = "instances" + os.path.sep
demo_data_path = os.path.dirname(demos.__file__) + os.path.sep + 'data'




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

    def get_instances(self,visualisation):
        if visualisation:
            return super().get_instances(True)
        else: 
            instances = os.listdir(demo_data_path)
            return [i for i in instances if i[-3:] == 'cnf']

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
        file_path = instance_path
        if instance_path.startswith('random'):
            file_path = "gnm" + instance_path[6:] + "-27" #add seed TODO may allow user to set seed
        instance = MISPInstance(file_path)
        return MISPSolution(instance)

    def get_instances(self, visualisation):
        if visualisation:
            inst = super().get_instances(True)
            return inst + ['random']
        else: 
            instances = os.listdir(demo_data_path)
            return [i for i in instances if i[-3:] == 'mis']





# initialize problems (parameter 'name' corresponds to name of directories for instances/logs)
problems = {
            Problem.MAXSAT: MAXSAT('maxsat'),
            Problem.MISP: MISP('misp')
            }


# methods used by module interface to extract information for widgets
def get_problems():
    return [p.value for p,_ in problems.items()]

def get_instances(prob: Problem,visualisation):
    return problems[prob].get_instances(visualisation)

def get_algorithms(prob: Problem):
    return problems[prob].get_algorithms()

def get_options(prob: Problem, algo: Algorithm):
    return problems[prob].get_options(algo)


def run_algorithm_visualisation(options: dict):
    settings.mh_titer = 100
    fh = logging.FileHandler(f"logs/{options['prob'].name.lower()}/{options['algo'].name.lower()}/{options['algo'].name.lower()}.log", mode="w")
    logger_step.handlers = []
    logger_step.addHandler(fh)
    logger_step.info(f"{options['prob'].name}\n{options['algo'].name}")

    file_path = instance_path + problems[options['prob']].name + os.path.sep + options['inst']
    if options['inst'].startswith('random'):
        file_path = options['inst']
    # initialize solution for problem
    #solution = problems[options['prob']].get_solution(file_path)
    solution = run_algorithm(options, file_path)
    return get_log_data(options['prob'].name.lower(), options['algo'].name.lower()), solution.inst



def run_algorithm_comparison(configs: list()):
    settings.mh_titer = 500

    log_df = pd.DataFrame({'iteration':[]})
    
    for config in configs:

        iter_fh = logging.FileHandler(f"logs/iter.log", mode="w")
        iter_logger = logging.getLogger("pymhlib_iter")
        iter_logger.handlers = []
        iter_logger.addHandler(iter_fh)

        file_path = demo_data_path + os.path.sep + config['inst']

        name = config['name']
        _ = run_algorithm(config,file_path, visualisation=False)

        df = pd.read_csv('logs' + os.path.sep + 'iter.log', sep=r'\s+', usecols=['iteration','obj_new'])
        df.rename(columns = {'obj_new':name}, inplace=True)
        log_df = pd.merge(log_df, df, how = 'outer', on = 'iteration')

    return log_df


def run_algorithm(options: dict, file_path: str, visualisation=True):

    # initialize solution for problem
    solution = problems[options['prob']].get_solution(file_path)

    # run specified algorithm
    if options['algo'] == Algorithm.GVNS:
        run_gvns(solution, options, visualisation)

    if options['algo'] == Algorithm.GRASP:
        run_grasp(solution, options, visualisation)
        
    return solution
######################### overwrite GVNS to log necessary information 

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
                logger_step.info('END_ITER')
                ###
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


########################################


def run_gvns(solution, options: dict, visualisation: bool):


    prob = problems[options['prob']]
    ch = [ prob.get_method(Algorithm.GVNS, Option.CH, m[0], m[1]) for m in options[Option.CH] ]
    li = [ prob.get_method(Algorithm.GVNS, Option.LI, m[0], m[1]) for m in options[Option.LI] ]
    sh = [ prob.get_method(Algorithm.GVNS, Option.SH, m[0], m[1]) for m in options[Option.SH] ]
    
    ### for now, the overwritten GVNS is called
    alg = MyGVNS(solution, ch, li, sh, consider_initial_sol=True, logger_step=visualisation)

    alg.run()
    alg.method_statistics()
    alg.main_results()
    logging.getLogger("pymhlib_iter").handlers[0].flush()


def run_grasp(solution, options: dict, visualisation):
    
    prob = problems[options['prob']]

    ch = []
    li = [ prob.get_method(Algorithm.GRASP, Option.LI, m[0], m[1]) for m in options[Option.LI] ]
    rgc = [ prob.get_method(Algorithm.GRASP, Option.RGC, m[0], m[1]) for m in options[Option.RGC] ]

    # for now the overwritten gvns is called
    alg = MyGVNS(solution,ch,li,rgc,consider_initial_sol=True, logger_step=visualisation)
    alg.run()
    alg.method_statistics()
    alg.main_results()
    logging.getLogger("pymhlib_iter").handlers[0].flush()


def run_comparison(configs: list()):
    log_df = pd.DataFrame({'iteration':[]})
    
    for config in configs:
        name = config['name']
        _,_ = run_algorithm(config,visualisation=False)
        print(name, 'done')

        df = pd.read_csv('logs' + os.path.sep + 'iter.log', sep=r'\s+', usecols=['iteration','obj_new'])
        df.rename(columns = {'obj_new':name}, inplace=True)
        log_df = pd.merge(log_df, df, how = 'outer', on = 'iteration')
    return log_df

    

# only used for debugging
if __name__ == '__main__':
        filepath = instance_path + 'maxsat' + os.path.sep + 'cnf_7_50.cnf'
        _ = run_algorithm_visualisation({'prob':Problem.MAXSAT, Option.CH:[('random',0)], 'inst':'cnf_7_50.cnf','algo':Algorithm.GRASP,
           Option.LI:[('two-exchange random fill neighborhood search',2)],
          Option.RGC:[('k-best',5)]})

        #print(log_data)





        















            

