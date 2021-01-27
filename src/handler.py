"""
handler module which provides information for widgets in interface module and uses widget input
to issue pymhlib calls
"""

import os
import logging
import numpy as np
import pandas as pd

# pymhlib imports

from .pymhlib.settings import settings, parse_settings, seed_random_generators
from .pymhlib.log import init_logger
from .pymhlib import demos
from .pymhlib.gvns import GVNS

# module imports
from .problems import Problem, Algorithm, Option, MAXSAT, MISP
from .logdata import get_log_data


if not settings.__dict__: parse_settings(args='')

# TODO: pymhlib settings
settings.mh_lfreq = 1
settings.mh_out = "logs" + os.path.sep + "summary.log"
settings.mh_log = "logs" + os.path.sep + "iter.log"
settings.mh_log_step = 'None'
   


#logger_step = logging.getLogger("step-by-step")
#logger_step.setLevel(logging.INFO)
instance_path = "instances" + os.path.sep
demo_data_path = os.path.dirname(demos.__file__) + os.path.sep + 'data'
step_log_path = "logs" + os.path.sep


# initialize available problems
problems = {
            Problem.MAXSAT: MAXSAT(),
            Problem.MISP: MISP()
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

    #fh = logging.FileHandler(f"logs/{options['prob'].name.lower()}/{options['algo'].name.lower()}/{options['algo'].name.lower()}.log", mode="w")
    #logger_step.handlers = []
    #logger_step.addHandler(fh)
    #logger_step.info(f"{options['prob'].name}\n{options['algo'].name}")
    settings.mh_log_step = step_log_path + os.path.sep.join([options['prob'].name.lower(),options['algo'].name.lower(),options['algo'].name.lower()]) + ".log"
    init_logger()

    settings.seed =  options.get('settings').get('seed',0)
    seed_random_generators()

    file_path = instance_path + problems[options['prob']].name + os.path.sep + options['inst']
    if options['inst'].startswith('random'):
        file_path = options['inst']

    solution = run_algorithm(options, file_path)
    return get_log_data(options['prob'].name.lower(), options['algo'].name.lower()), solution.inst



def run_algorithm_comparison(config: dict):

    settings.mh_log_step = 'None'
    init_logger()

    s = config['settings']
    file_path = demo_data_path + os.path.sep + config['inst'] + \
        (f'-{s["seed"]}' if config['inst'].startswith('random') and s['seed'] > 0 else '')
    name = config['name']
 
    settings.seed =  s.get('seed',0)
    seed_random_generators()
    for i in range(s.get('runs',1)):
        _ = run_algorithm(config, file_path, visualisation=False)
    log_df = read_iter_log(name)
    summary = read_sum_log()

    return log_df, summary

def read_sum_log():
    idx = []
    with open('logs' + os.path.sep + 'summary.log') as f: 
        for i, line in enumerate(f):
            if not line.startswith('S '):
                idx.append(i)
        f.close()
        
    df = pd.read_csv('logs' + os.path.sep + 'summary.log',sep=r'\s+',skiprows=idx)
    df.drop(labels=['S'], axis=1,inplace=True)
    idx = df[ df['method'] == 'method' ].index
    df.drop(idx , inplace=True)

    n = len(df[df['method'] == 'SUM/AVG'])
    m = int(len(df) / n)
    df['run'] = (np.array([i for i in range(1,n+1)]).repeat(m))
    df.set_index(['run','method'], inplace=True)
    return df


def read_iter_log(name):

        df = pd.read_csv('logs' + os.path.sep + 'iter.log', sep=r'\s+', header=None)

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

        


def run_algorithm(options: dict, file_path: str, visualisation=True):

    if options.get('settings', False):
        settings.mh_titer = options['settings'].get('iterations',100)


    # initialize solution for problem
    solution = problems[options['prob']].get_solution(file_path)

    # run specified algorithm
    if options['algo'] == Algorithm.GVNS:
        run_gvns(solution, options, visualisation)

    if options['algo'] == Algorithm.GRASP:
        run_grasp(solution, options, visualisation)
        
    return solution


def run_gvns(solution, options: dict, visualisation: bool):

    prob = problems[options['prob']]
    ch = [ prob.get_method(Algorithm.GVNS, Option.CH, m[0], m[1]) for m in options[Option.CH] ]
    li = [ prob.get_method(Algorithm.GVNS, Option.LI, m[0], m[1]) for m in options[Option.LI] ]
    sh = [ prob.get_method(Algorithm.GVNS, Option.SH, m[0], m[1]) for m in options[Option.SH] ]
    
    alg = GVNS(solution, ch, li, sh, consider_initial_sol=True)

    alg.run()
    alg.method_statistics()
    alg.main_results()
    logging.getLogger("pymhlib_iter").handlers[0].flush()



def run_grasp(solution, options: dict, visualisation):
    
    prob = problems[options['prob']]

    ch = []
    li = [ prob.get_method(Algorithm.GRASP, Option.LI, m[0], m[1]) for m in options[Option.LI] ]
    rgc = [ prob.get_method(Algorithm.GRASP, Option.RGC, m[0], m[1]) for m in options[Option.RGC] ]

    alg = GVNS(solution,ch,li,rgc,consider_initial_sol=True)
    alg.run()
    alg.method_statistics()
    alg.main_results()
    logging.getLogger("pymhlib_iter").handlers[0].flush()

    

# only used for debugging
if __name__ == '__main__':
        filepath = instance_path + 'maxsat' + os.path.sep + 'cnf_7_50.cnf'
        _ = run_algorithm_visualisation({'prob':Problem.MAXSAT, Option.CH:[('random',0)], 'inst':'cnf_7_50.cnf','algo':Algorithm.GRASP,
           Option.LI:[('two-exchange random fill neighborhood search',2)],
          Option.RGC:[('k-best',5)]})

        #print(log_data)





        















            

