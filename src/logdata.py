
import ast
import re
import os
import sys
sys.path.append('..\\HeuOptDemos_Eva')

import enum

class Log(enum.Enum):
        StepInter = 'step-by-step (intermediate steps)' # start-frame and end-frame for each step
        StepNoInter = 'step-by-step (no intermediate steps)' # start and end combined in one frame
        Update = 'updated solutions' # result of a phase e.g. li(vnd)-cycle, complete rgc in one frame
        NewInc = 'new incumbents' # like Update, but only if new incumbent was found
        FullCycle = 'major cycles' # one full cycle of algorithm, e.g. sh+li, rgc+li, per frame



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
            vis_data.append(create_gvns_data(data[end:end+8]))
        if method in ['rgc']:
            vis_data += create_grasp_data(data[start:end+8])

        i = end + 8

    return vis_data


def create_gvns_data(data: list()):

    entries = {list(d.keys())[0]: list(d.values())[0] for d in data[1:]}
    entries['status'] = data[0]
    return entries


def create_grasp_data(data: list()):

    entries = [create_gvns_data(data[:7])]

    greedy_data = data[7:-8]

    for i in range(0, len(greedy_data),5):
        rcl_data = {list(d.keys())[0]:list(d.values())[0] for d in greedy_data[i:i+5]}

        entries.append({'status':'cl', 'cl':rcl_data['cl'], 'sol':rcl_data['sol']})
        entries.append({'status':'rcl', 'cl':rcl_data['cl'], 'rcl':rcl_data['rcl'], 'sol':rcl_data['sol'], 'par':rcl_data['par']})
        sol =  data[-7]['sol']  if i == len(greedy_data) -5 else greedy_data[i+5]['sol']
        entries.append({'status':'sel', 'cl':rcl_data['cl'], 'rcl':rcl_data['rcl'], 'sol':sol, 'sel':rcl_data['sel']})

    entries.append(create_gvns_data(data[-8:]))
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


def get_filtered_logdata(i: int, log_data: list, granularity: Log):
    print(i,granularity)
    return log_data

# only used for debugging
if __name__ == '__main__':
    data = get_log_data('misp', 'grasp')
    i = 0
    for d in data:
        print(d)
        i += 1
        if i > 10:
            break
