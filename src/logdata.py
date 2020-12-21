
import ast
import re
import os
import sys
sys.path.append('..\\HeuOptDemos_Eva')

import enum
import time
from src.problems import Problem, Algorithm, Option
from pymhlib.demos.misp import MISPInstance
from pymhlib.demos.maxsat import MAXSATInstance

class Log(enum.Enum):
        StepInter = 'step-by-step (intermediate steps)' # start-frame and end-frame for each step
        StepNoInter = 'step-by-step (no intermediate steps)' # start and end combined in one frame
        NewInc = 'new incumbents' # like StepNoInter, but only if new incumbent was found
        Update = 'updated solutions' # result of a phase e.g. li(vnd)-cycle, complete rgc in one frame
        Cycle = 'major cycles' # one entire cycle of algorithm, e.g. sh+li, rgc+li, per frame

# global variables
len_start = 7
len_end = 8


class LogData():

    def __init__(self, log_data: list):
        self.full_data = log_data
        self.levels = self.init_levels()
        self.current_level = Log.StepInter
        self.log_data = log_data # holds logdata for currently active log level

    def init_levels(self):
        levels = dict()
        levels[Log.StepInter] = list(range(2, len(self.full_data)))
        levels[Log.StepNoInter] = [i for i in levels[Log.StepInter] if self.full_data[i].get('status') != 'start']
        levels[Log.NewInc] = [i for i in levels[Log.StepNoInter] if self.full_data[i].get('better',False)]
        update = list()
        for i in levels[Log.StepNoInter][:-1]:
            if self.full_data[i].get('m', False) and (self.full_data[i].get('m') != self.full_data[i+1].get('m')):
                update.append(i)
        else:
            update.append(levels[Log.StepNoInter][-1])

        levels[Log.Update] = update
        levels[Log.Cycle] = [ i for i in levels[Log.StepNoInter] if self.full_data[i].get('end', False)]

        return levels


    def change_granularity(self, i: int, granularity: Log):
        # sets self.log_data to requested granularity and returns the number of iteration to show next
        self.log_data = self.full_data[:2] + [self.full_data[i] for i in self.levels[granularity]]
        current_iter = self.levels[self.current_level][i]
        next_iter = len(self.levels[granularity]) -1  if current_iter > self.levels[granularity][-1] else next(i for i,val in enumerate(self.levels[granularity]) if val >= current_iter) 
        self.current_level = granularity
        return next_iter



def get_log_data(prob: str, alg: str):

    filepath = (os.path.sep).join(['logs', prob, alg, alg+'.log'])
    data = list()
    with open(filepath, 'r') as logfile:
        for line in logfile:
            data.append(cast_line(line.strip()))
        logfile.close()
    
    return create_log_data(data)


def create_log_data(data: list()):

    vis_data = data[:2]
    i = 2
    while i < len(data):
        # find slice from start to end
        if data[i] == 'end_iter':
            vis_data[-1]['end'] = True
            i += 1
            if i >= len(data):
                break
        start = i
        while not data[start] == 'start':
            start += 1
        end = start
        while not data[end] == 'end':
            end +=1

        method = data[start+3]['m']

        # create data from start-end-slices according to method
        if method in ['ch','li', 'sh']:
            vis_data.append(create_gvns_data(data[start:end]))
            vis_data.append(create_gvns_data(data[end:end+len_end]))
        if method in ['rgc']:
            vis_data += create_grasp_data(data[start:end+len_end])

        i = end + len_end

    return vis_data


def create_gvns_data(data: list()):

    entries = {list(d.keys())[0]: list(d.values())[0] for d in data[1:]}
    entries['status'] = data[0] 
    return entries


def create_grasp_data(data: list()):

    entries = [create_gvns_data(data[:len_start])]

    greedy_data = data[len_start:-len_end]

    for i in range(0, len(greedy_data),5):
        rcl_data = {list(d.keys())[0]:list(d.values())[0] for d in greedy_data[i:i+5]}

        entries.append({'status':'cl', 'cl':rcl_data['cl'], 'sol':rcl_data['sol']})
        entries.append({'status':'rcl', 'cl':rcl_data['cl'], 'rcl':rcl_data['rcl'], 'sol':rcl_data['sol'], 'par':rcl_data['par']})
        sol =  data[-7]['sol']  if i == len(greedy_data) -5 else greedy_data[i+5]['sol']
        entries.append({'status':'sel', 'cl':rcl_data['cl'], 'rcl':rcl_data['rcl'], 'sol':sol, 'sel':rcl_data['sel']})

    entries.append(create_gvns_data(data[-len_end:]))
    return entries

        
def cast_line(line: str):
    if not ':' in line:
        return line.lower()
    idx = line.find(':')
    name, data = line[:idx].strip().lower(), line[idx+1:].strip()

    if re.match("^[+-]?[0-9]+(\.)?[0-9]*", data): #numerical data
        return {name: cast_number(data)}

    if re.match("^[a-z]+[0-9]+(\.)?[0-9]*", data): # method with parameter
        #extract method name
        x = re.search("^[a-z]+", data)
        return {name: x.group() }

    if data in ['False', 'True']:
        return {name: False if data=='False' else True}

    x = re.search(r'(?<=\[)(.*?)(?=\])', data)
    if x: #list
        x = x.group()
        x = ' '.join(x.split())
        x = "[" + x.replace(" ",",") + "]"
        return {name: ast.literal_eval(x)} 

    x = re.search(r'(?<=\{)(.*?)(?=\})', data)
    if x: #dict
        x = "{" + x.group() + "}"
        return {name: ast.literal_eval(x)}

    return {name: data}


def cast_number(data: str):

    if re.match("^[-+]?[0-9]+$", data): #int
        return int(data)

    if re.match("^[-+]?[0-9]+\.[0-9]*", data): #float
        return float(data)


def save_visualisation(params: dict, graph=None):
    # if instance==random, create instance file from graph, save in instance folder and keep filename
    inst_filename = params['inst']
    if inst_filename.startswith('random'):
        # for now only available for misp
        if params['prob'].name == 'MISP':
            inst_filename = save_misp_instance(graph)

    # get current log file according to problem and algo and copy content
    logfile = os.path.sep.join( ['logs', params['prob'].name.lower(), params['algo'].name.lower(), params['algo'].name.lower() + '.log'] )
    with open(logfile, 'r') as source:
        timestamp = time.strftime('_%Y%m%d_%H%M%S')
        with open(os.path.sep.join(['logs','saved',params['prob'].name.lower()+ '_' + params['algo'].name.lower() + timestamp + '.log']), 'w') as destination:
            data = source.read()
            # prepend description block to log file (instance filename, options)
            destination.write('I: ' + inst_filename + '\n')
            for k,v in params.items():
                if type(k) == Option:
                    destination.writelines( [f'O: {k.name} {o}\n' for o in v] )
            destination.write(data)
            source.close()
            destination.close()


def save_misp_instance(graph):
    filename = '_'.join(['gnm', str(graph.order()), str(graph.size()), time.strftime('%Y%m%d%H%M%S')]) + '.mis'
    pathname = 'instances' + os.path.sep + 'misp' + os.path.sep + filename
    with open(pathname, 'w') as inst_file:
        inst_file.writelines(['c "source: networkx.gnm_random_graph()"\n', f'p edge {graph.order()} {graph.size()}\n'])
        for u,v in graph.edges():
            inst_file.write(f'e {u+1} {v+1}\n')
        inst_file.close()
    return filename


def read_from_logfile(filename: str):
    data = list()
    instance_file = ''
    with open('logs' + os.path.sep + 'saved' + os.path.sep + filename, 'r') as logfile:
        for line in logfile:
            if line.startswith('I:'):
                instance_file = line.split(' ')[1].strip()
                continue
            if line.startswith('O:'):
                continue
            data.append(cast_line(line.strip()))
        logfile.close()
    
    instance_path = 'instances' + os.path.sep + data[0].lower() + os.path.sep + instance_file
    inst = None
    if data[0] == 'misp':
        inst = MISPInstance(instance_path)
    if data[0] == 'maxsat':
        inst = MAXSATInstance(instance_path)

    return create_log_data(data), inst

def get_log_description(filename: str):
    if not filename:
        return ''
    description = []
    with open('logs' + os.path.sep + 'saved' + os.path.sep + filename, 'r') as logfile:
        for line in logfile:
            if line.startswith('I:'):
                description.append('Instance: ' + line[2:].strip())
            elif line.startswith('O:'):
                description.append(line[2:].strip())
            else:
                break
    
    return '\n'.join(description)



# only used for debugging
if __name__ == '__main__':
    data = get_log_data('misp', 'grasp')
    i = 0
    for d in data:
        print(d)
        i += 1
        if i > 10:
            break
