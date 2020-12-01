"""
handler module which provides information for widgets in interface module and uses widget input
to issue pymhlib calls
"""

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

from src.methods import greedy_randomized_construction


if not settings.__dict__: parse_settings(args='')

logger_step = logging.getLogger("step-by-step")



# TODO: pymhlib settings
settings.mh_titer = 100
settings.mh_lfreq = 1
settings.mh_tciter = 30
settings.mh_out = "logs" + os.path.sep + "summary.log"
settings.mh_log = "logs" + os.path.sep + "iter.log" # TODO logging to file not working
init_logger()

instance_path = "instances" + os.path.sep

# extend enums as needed, they hold the string values which are used for representation in widgets
class Problem(enum.Enum):
    MAXSAT = 'MAX-SAT'
    MISP = 'MAX-Independent Set'

class Algorithm(enum.Enum):
    GVNS = 'GVNS'
    GRASP = 'GRASP'

class Option(enum.Enum):
    CH = 'Construction'
    LI = 'Local Improvement'
    SH = 'Shaking'
    RCL = 'Restricted Candidate List'



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
        method = Method(f'{opt.name.lower()[0:2]}{par if type(m[2]) == type else m[2]}', m[1], par)
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
                                Option.RCL: [('k-best', greedy_randomized_construction, int),('alpha', greedy_randomized_construction, float)]
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
                                Option.SH: [('remove k and random flip', MISPSolution.shaking, int)]
                                }
                                ,
                    Algorithm.GRASP: {
                                Option.CH: [('random', MISPSolution.construct, 0)],   #not needed for grasp
                                Option.LI: [('two-exchange random fill neighborhood search', MAXSATSolution.local_improve, 2)],
                                Option.RCL: [('k-best', None, int),('alpha', None, float)]
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

    fh = logging.FileHandler(f"logs/{options['prob'].name.lower()}/{options['algo'].name.lower()}/{options['algo'].name.lower()}.log", mode="w")
    logger_step.handlers = []
    logger_step.addHandler(fh)
    logger_step.setLevel(logging.INFO)
    logger_step.info(f"{options['prob'].name}\n{options['algo'].name}")

    file_path = instance_path + problems[options['prob']].name + os.path.sep + options['inst']

    if options['inst'] == 'random': #only available for misp so far
        file_path = "gnm-30-60"

    # initialize solution for problem
    solution = problems[options['prob']].get_solution(file_path)

    # run specified algorithm
    if options['algo'] == Algorithm.GVNS:
        run_gvns(solution, options)
        return get_gvns_log_data(options['prob'], Algorithm.GVNS), solution.inst
    if options['algo'] == Algorithm.GRASP:
        run_grasp(solution, options)
        return get_gvns_log_data(options['prob'], Algorithm.GRASP), solution.inst
   
    return [], None


def get_gvns_log_data(prob: Problem, alg: Algorithm):
    file_path = 'logs' + os.path.sep + prob.name.lower() + os.path.sep + alg.name.lower() + os.path.sep + alg.name.lower() + '.log'

    data = list()
    entry_start = {}
    entry_end = {}
    entry = {'start': {}, 'end': {}}
    rcl_data = {}
    with open(file_path) as logf:
        status = ''
        method = ''
        for i,line in enumerate(logf):
            if i <2:
                data.append(line.strip())
                continue

            if line.startswith('START') or line.startswith('END'):
                status = line.split(':')[0].lower()
                current = cast_solution(line)
                entry[status]['status'] = status
                entry[status]['current'] = current
            if line.startswith('OBJ'):
                obj = int(line.split(':')[1].strip())    #TODO could be necessary to cast to float, depends on problems
                entry[status]['obj'] = obj
            if line.startswith('INC'):
                inc = cast_solution(line)
                entry[status]['inc'] = inc
            if line.startswith('M:'):
                m = line.split(':')[1].strip()
                entry[status]['m'] = m
                method = line.split(':')[1].strip()
            if line.startswith('BETTER'):
                better = line.split(':')[1].strip()
                entry[status]['better'] = True if better=='True' else False
            if line.startswith('BEST'):
                best = int(line.split(':')[1].strip())    #TODO could be necessary to cast to float, depends on problems
                entry[status]['best'] = best
                data.append(entry[status])
                entry[status] = {}

            ### TODO specific to maxsat, has to be adapted to other problems!!!!
            if line.startswith('CL'):
                status = 'rcl'
                rcl_data['cl'] = cast_solution(line)
            if line.startswith('X'):
                rcl_data['x'] = cast_solution(line)
            if line.startswith('UNFUL'):
                rcl_data['unful'] = int(line.split(':')[1].strip())
            if line.startswith('THRESH'):
                rcl_data['thresh'] = float(line.split(':')[1].strip())
            if line.startswith('MAX'):
                rcl_data['max'] = int(line.split(':')[1].strip())
            if line.startswith('RCL'):
                rcl_data['rcl'] = cast_solution(line)
            if line.startswith('SEL'):
                rcl_data['sel'] = int(line.split(':')[1].strip())
            if line.startswith('ADDED'):
                rcl_data['added'] = int(line.split(':')[1].strip())
                data.append({'status':'cl', 'cl':rcl_data['cl'], 'unful':rcl_data['unful'], 'x':rcl_data['x']})
                par = method[2:]
                rcl = {'status':'rcl', 'cl':rcl_data['cl'], 'rcl':rcl_data['rcl'], 'x':rcl_data['x']}
                if 'max' in rcl_data:
                    rcl['max'] = rcl_data['max']
                    rcl['thresh'] = rcl_data['thresh']
                    rcl['alpha'] = float(par)
                else:
                    rcl['k'] = int(par)
                data.append(rcl)
                x = rcl_data['x']
                x[abs(rcl_data['sel'])-1] = 0 if rcl_data['sel'] < 0 else 1
                data.append({'status':'sel', 'cl':rcl_data['cl'], 'rcl':rcl_data['rcl'], 'x':x, 'sel':rcl_data['sel'], 'added':rcl_data['added']})
                rcl_data = {}

        logf.close()
    return data



def cast_solution(sol: str):

    x = re.search(r'(?<=\[)(.*?)(?=\])', sol.strip())

    if x: #solution is a list
        x = x.group()
        x = ' '.join(x.split())
        x = "[" + x.replace(" ",",") + "]"
        return ast.literal_eval(x)
    # string is dict
    x = re.search(r'(?<=\{)(.*?)(?=\})', sol.strip())

    x = "{" + x.group() + "}"
    return ast.literal_eval(x)


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
        step_info = f'START: {sol}\nOBJ: {obj_old}\nM: {method.name}\nINC: {self.incumbent}\nBEST: {self.incumbent.obj()}'
        self.logger_step.info(step_info)
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
        step_info = f'END: {sol}\nOBJ: {sol.obj()}\nM: {method.name}\nBETTER: {new_incumbent}\nINC: {self.incumbent}\nBEST: {self.incumbent.obj()}'
        self.logger_step.info(step_info)
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
    
    ### TODO for now, the overwritten GVNS is called
    alg = MyGVNS(logger_step, solution, ch, li, sh)

    alg.run()
    alg.method_statistics()
    alg.main_results()


def run_grasp(solution, options: dict):
    
    prob = problems[options['prob']]

    #ch = [ prob.get_method(Algorithm.GRASP, Option.CH, 'random', 0) ]
    ch = []
    li = [ prob.get_method(Algorithm.GRASP, Option.LI, m[0], m[1]) for m in options[Option.LI] ]
    rcl = [ prob.get_method(Algorithm.GRASP, Option.RCL, m[0], m[1]) for m in options[Option.RCL] ]

    alg = MyGVNS(logger_step, solution,ch,li,rcl,consider_initial_sol=True)
    alg.run()
    alg.method_statistics()
    alg.main_results()



        















            

