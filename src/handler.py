"""
handler module which provides information for widgets in interface module and uses widget input
to issue pymhlib calls
"""


import os
import logging
import numpy as np
import pandas as pd

# pymhlib imports
from pymhlib.settings import settings, parse_settings, seed_random_generators
from pymhlib.log import init_logger
from pymhlib import demos
from pymhlib.gvns import GVNS
from pymhlib.ts import TS
from pymhlib.sa import SA, Method

# module imports
from .problems import Problem, Algorithm, Option, MAXSAT, MISP, Configuration, ProblemDefinition
from .logdata import get_log_data


if not settings.__dict__: parse_settings(args='')

# pymhlib settings
settings.mh_lfreq = 1


vis_instance_path = "instances" + os.path.sep
demo_data_path = os.path.dirname(demos.__file__) + os.path.sep + 'data'
step_log_path = "logs" + os.path.sep + "step.log"
iter_log_vis_path = "logs" + os.path.sep + "iter_vis.log"
sum_log_vis_path = "logs" + os.path.sep + "summary_vis.log"
iter_log_path = "logs" + os.path.sep + "iter.log"
sum_log_path = "logs" + os.path.sep + "summary.log"


# get available problems
problems = {p.name: p for p in [prob() for prob in ProblemDefinition.__subclasses__()]}

# methods used by module interface to extract information for widgets
def get_problems():
    return [p.value for p in problems.keys()]

def get_instances(prob: Problem,visualisation):
    return problems[prob].get_instances(visualisation)

def get_algorithms(prob: Problem):
    return problems[prob].get_algorithms()

def get_options(prob: Problem, algo: Algorithm):
    return problems[prob].get_options(algo)


def run_algorithm_visualisation(config: Configuration):
    settings.mh_out = sum_log_vis_path
    settings.mh_log = iter_log_vis_path
    settings.mh_log_step = step_log_path 
    init_logger()

    settings.seed =  config.seed
    seed_random_generators()

    solution = run_algorithm(config,True)
    return get_log_data(config.problem.name.lower(), config.algorithm.name.lower()), solution.inst



def run_algorithm_comparison(config: Configuration):
    settings.mh_out = sum_log_path
    settings.mh_log = iter_log_path
    settings.mh_log_step = 'None'
    init_logger()
    settings.seed =  config.seed
    seed_random_generators()

    for i in range(config.runs):
        _ = run_algorithm(config)
    log_df = read_iter_log(config.name)
    summary = read_sum_log()

    return log_df, summary


def read_sum_log():
    idx = []
    with open(sum_log_path) as f: 
        for i, line in enumerate(f):
            if not line.startswith('S '):
                idx.append(i)
        f.close()
        
    df = pd.read_csv(sum_log_path, sep=r'\s+',skiprows=idx)
    df.drop(labels=['S'], axis=1,inplace=True)
    idx = df[ df['method'] == 'method' ].index
    df.drop(idx , inplace=True)

    n = len(df[df['method'] == 'SUM/AVG'])
    m = int(len(df) / n)
    df['run'] = (np.array([i for i in range(1,n+1)]).repeat(m))
    df.set_index(['run','method'], inplace=True)
    return df


def read_iter_log(name):

        df = pd.read_csv(iter_log_path, sep=r'\s+', header=None)

        df.drop(df[ df[1] == '0' ].index , inplace=True) #drop initialisation line
        df = df[4].reset_index().drop(columns='index') #extract 'obj_new'
        indices = list((df[df[4] == 'obj_new'].index)) + [len(df)] #get indices of start of each run
        list_df = []
        #split data in single dataframes
        for i in range(len(indices) - 1):
            j = indices[i+1]-1
            frame = df.loc[indices[i]:j]
            frame = frame.reset_index().drop(columns='index')
            list_df.append(frame)
        full = pd.concat(list_df,axis=1) #concatenate dataframes
        full.columns = [i for i in range(1,len(full.columns)+1)] #rename columns to run numbers
        full.columns = pd.MultiIndex.from_tuples(zip([name]*len(full.columns), full.columns)) # set level of column
        full = full.drop([0]) #drop line that holds old column names
        full = full.astype(float)
       
        return full



def run_algorithm(config: Configuration, visualisation: bool=False):

    settings.mh_titer = config.iterations

    # initialize solution for problem
    solution = problems[config.problem].get_solution(config.get_inst_path(visualisation))

    # run specified algorithm
    if config.algorithm == Algorithm.GVNS:
        run_gvns(solution, config)

    if config.algorithm == Algorithm.GRASP:
        run_grasp(solution, config)

    if config.algorithm == Algorithm.TS:
        run_ts(solution, config)
    
    if config.algorithm == Algorithm.SA:
        run_sa(solution, config)
    return solution


def run_gvns(solution, config: Configuration):

    prob = problems[config.problem]
    ch = [ prob.get_method(Algorithm.GVNS, Option.CH, m[0], m[1]) for m in config.options[Option.CH] ]
    li = [ prob.get_method(Algorithm.GVNS, Option.LI, m[0], m[1]) for m in config.options[Option.LI] ]
    sh = [ prob.get_method(Algorithm.GVNS, Option.SH, m[0], m[1]) for m in config.options[Option.SH] ]
    
    alg = GVNS(solution, ch, li, sh, consider_initial_sol=False)

    alg.run()
    alg.method_statistics()
    alg.main_results()
    logging.getLogger("pymhlib_iter").handlers[0].flush()



def run_grasp(solution, config: Configuration):
    if config.options[Option.RGC][0][0] == 'k-best':
        settings.mh_grc_par = True
        settings.mh_grc_k = config.options[Option.RGC][0][1]
    else:
        settings.mh_grc_par = False
        settings.mh_grc_alpha = config.options[Option.RGC][0][1]

    
    prob = problems[config.problem]

    ch = []
    li = [ prob.get_method(Algorithm.GRASP, Option.LI, m[0], m[1]) for m in config.options[Option.LI] ]
    rgc = [ prob.get_method(Algorithm.GRASP, Option.RGC, m[0], m[1]) for m in config.options[Option.RGC] ]

    alg = GVNS(solution,ch,li,rgc,consider_initial_sol=True)
    alg.run()
    alg.method_statistics()
    alg.main_results()
    logging.getLogger("pymhlib_iter").handlers[0].flush()


def run_ts(solution, config: Configuration):
    print(config.options)

    prob = problems[config.problem]
    ch = [ prob.get_method(Algorithm.TS, Option.CH, m[0], m[1]) for m in config.options[Option.CH] ]
    li = [ prob.get_method(Algorithm.TS, Option.LI, m[0], m[1]) for m in config.options[Option.LI] ]
    mini, maxi, change = config.options[Option.TL][0][1], config.options[Option.TL][1][1], config.options[Option.TL][2][1]

    alg = TS(solution, ch, li, mini, maxi, change)
    alg.run()
    alg.method_statistics()
    alg.main_results()
    logging.getLogger("pymhlib_iter").handlers[0].flush()


def run_sa(solution, config : Configuration): # todo maybe add construction option
    own_settings = {
        'ma_sa_T_init': config.options[Option.TEMP],
        'mh_sa_alpha': config.options[Option.ALPHA],
        'mn_sa_equi_iter': config.options[Option.EQUI_ITER]
    }
    solution_class = type(solution)

    alg = SA(solution, [Method("ch", solution_class.construct, 0)], solution_class.random_move_delta_eval,
            solution_class.apply_neighborhood_move, iter_cb = None, own_settings = own_settings )
    alg.run()
    alg.method_statistics()
    alg.main_results()
    logging.getLogger("pymhlib_iter").handlers[0].flush()






        















            

