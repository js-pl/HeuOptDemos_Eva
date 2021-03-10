
from pymhlib.demos.maxsat import MAXSATInstance
from pymhlib.demos.misp import MISPInstance

import matplotlib.image as mpimg
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import statistics
import math

from .problems import InitSolution, Problem, Algorithm, Option
from abc import ABC, abstractmethod
from .logdata import read_from_logfile, Log
from matplotlib.lines import Line2D
from dataclasses import dataclass



plt.rcParams['figure.figsize'] = (12,6)
plt.rcParams['figure.dpi'] = 80
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.facecolor'] = 'w'
pc_dir = 'pseudocode'


@dataclass
class CommentParameters:
        n: int = 0
        m: int = 0
        par: int = 0
        gain = 0
        better: bool = False
        no_change: bool = False

        # algorithm specific parameters
        remove: set = None
        add: set = None
        flip: list = None
        k: int = None
        alpha: float = None
        thres: float = None
        ll: int = 0
        asp: bool = False
        temp: float = 0
        delta: int = 0
        accept: bool = False

class Draw(ABC):




        phases = {'ch':'Construction', 'li': 'Local Search', 'sh': 'Shaking', 'rgc': 'Randomized Greedy Construction', 
                        'cl':'Candidate List', 'rcl': 'Restricted Candidate List', 'sel':'Selection from RCL',
                        'sa': 'Random neighborhood move'}

        plot_description = {'phase': '',
                                'comment': [],
                                'best': 0,
                                'obj': 0,
                                }
        grey = str(210/255) #'lightgrey'
        darkgrey = str(125/255)
        white = 'white'
        black = 'black'
        blue = 'royalblue'
        red = 'tomato'
        green = 'limegreen'
        yellow = 'gold'
        orange = 'darkorange'

        def __init__(self, prob: Problem, alg: Algorithm, instance, log_granularity: Log):
                self.problem = prob
                self.algorithm = alg
                self.graph = self.init_graph(instance)

                # create figure
                plt.close() # close any previously drawn figures
                self.fig, (self.ax, self.img_ax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, 
                        num = f'Solving {prob.value} with {alg.value}')
                self.img_ax.axis('off')
                self.ax.axis('off')
                self.log_granularity = log_granularity



        @abstractmethod
        def init_graph(self, instance):
                return None

        def get_animation(self, i: int, log_data: list):
                self.reset_graph()
                comment = None
                if self.algorithm == Algorithm.TS:
                        comment = self.get_ts_animation(i,log_data)
                if self.algorithm == Algorithm.GVNS:
                        comment = self.get_gvns_animation(i,log_data)
                if self.algorithm == Algorithm.GRASP:
                        comment = self.get_grasp_animation(i,log_data)
                if self.algorithm == Algorithm.SA:
                        comment = self.get_sa_animation(i, log_data)

                self.add_description(i, log_data)
                self.add_legend()
                self.load_pc_img(log_data[i], comment)


        @abstractmethod
        def get_grasp_animation(self, i: int, log_data: list):
                pass


        @abstractmethod
        def get_gvns_animation(self, i:int, log_data:list):
                pass

        @abstractmethod
        def get_ts_animation(self, i:int, log_data:list):
                pass

        @abstractmethod
        def get_sa_animation(self, i: int, log_data: list):
                pass

        def add_description(self, i, log_info: list):
                data = log_info[i]

                if data.get('status') == 'start' and i == 0:
                        data['best'] = 0
                        data['obj'] = 0
                        self.plot_description['phase'] = f'{self.problem.value} Instance'
                else:
                        self.plot_description['phase'] = self.phases.get(data.get('m',''),'')

                phase = self.plot_description['phase']
                
                if self.log_granularity == Log.Cycle and not data.get('m','').startswith('ch'):
                        if self.algorithm == Algorithm.GVNS:
                                phase = 'Shaking + Local Search'
                        if self.algorithm == Algorithm.GRASP:
                                phase = 'Randomized Greedy Construction + Local Search'


                self.ax.text(0,1, '\n'.join((
                '%s: %s' % (phase, self.plot_description['comment'] ),
                'Best Objective: %d' % (data.get('best',self.plot_description['best']), ),
                'Current Objective: %d' % (data.get('obj',self.plot_description['obj']),))), horizontalalignment='left', verticalalignment='top', transform=self.ax.transAxes)


                self.plot_description.update({'phase': '', 'comment': [], 'best':0, 'obj':0}) #reset description


        @abstractmethod
        def add_legend(self):
                pass

        def load_pc_img(self, log_info: dict, comment: CommentParameters):
                level = Log.StepInter.name.lower()
                m = log_info.get('m','')
                status = log_info.get('status','') if not log_info.get('end',False) else 'enditer'
                better = 'better' if m == 'li' and not comment.no_change else ''
                better = better if m == 'li' and status == 'end' else ''
                path = lambda level,m,status,better: f'{level}{"_" + m if m != "" else ""}{"_" + status}{"_" + better if better != "" else ""}'
                img_path = (os.path.sep).join( [pc_dir, self.algorithm.name.lower(), path(level,m,status,better) + '.PNG'] )
                img = mpimg.imread(img_path)
                self.img_ax.set_aspect('equal', anchor='E')
                self.img_ax.imshow(img)#,extent=[0, 1, 0, 1])

        @abstractmethod
        def reset_graph(self):
                pass
        @abstractmethod
        def draw_graph(self):
                pass





class MISPDraw(Draw):

        comments = {
                        Option.CH:{
                                'start': lambda params: f'{params.n} nodes, {params.m} edges',
                                'end': lambda params: f'initial solution={InitSolution(params.par).name}'
                        },
                        Option.LI: {
                                'start': lambda params: f'k={params.par}, remove {len(params.remove)} node(s), add {len(params.add)} node(s)',
                                'end': lambda params: f'objective gain={params.gain}{", no improvement - reached local optimum" if params.no_change else ""}{", found new best solution" if params.better else ""}'
                        },
                        Option.SH: {
                                'start': lambda params: f'k={params.par}, remove {len(params.remove)} node(s), add {len(params.add)} node(s)',
                                'end': lambda params: f'objective gain={params.gain}{", found new best solution" if params.better else ""}'
                        },
                        Option.RGC:{
                                'start': lambda params: 'start with empty solution',
                                'end': lambda params: f'created complete solution{", found new best solution" if params.better else ""}',
                                'cl': lambda params: 'candidate list=remaining degree (number of unblocked neigbors)',
                                'rcl': lambda params: f'restricted candidate list=' + (f'{params.k}-best' if params.k else f'alpha: {params.alpha}, threshold: {params.thres}'),
                                'sel': lambda params: f'selection=random, objective gain={params.gain}'
                        },
                        Option.TL:{
                                'start': lambda params: f'k={params.par} remove {len(params.remove)} node(s), add {len(params.add)} node(s){", apply aspiration criterion" if params.asp else ""}',
                                'end': lambda params: f'size of tabu list={params.ll}, objective gain={params.gain}{", all possible exchanges are tabu" if params.no_change else ""}{", found new best solution" if params.better else ""}'
                        }
                        
                        }

        def create_comment(self, option: Option, status: str, params: CommentParameters):
                # TODO create comments according to log granularity
                return self.comments[option][status](params)


        def __init__(self, prob: Problem, alg: Algorithm, instance, log_granularity: Log):
                super().__init__(prob,alg,instance,log_granularity)

        def init_graph(self, instance):
                graph = instance.graph
                

                nodelist = list(graph.nodes())
                nodelist = sorted(nodelist, key=lambda n:len(list(graph.neighbors(n))))
                nodelist.reverse()
                i = len(nodelist)
                nl = []
                while i > 0:
                        j = int(i/2)
                        nl.append(nodelist[j:i])
                        i = j
                nl.reverse()
                #pos = nx.shell_layout(graph,nlist=nl)
                
                pos = nx.spring_layout(graph,k=1,iterations=30)
                pos = nx.kamada_kawai_layout(graph,pos=pos)
                nx.set_node_attributes(graph, {n:{'color':self.grey, 'label':'', 'tabu':False} for n in graph.nodes()})
                nx.set_node_attributes(graph, pos, 'pos')
                nx.set_edge_attributes(graph, self.grey, 'color')
                return graph

        def get_gvns_animation(self, i:int, log_data: list):
                data = log_data[i]
                status = data.get('status','start')
                comment_params = CommentParameters()
                done = self.get_gvns_and_ts_animation(i,log_data,comment_params)
                if done:
                        return comment_params

                self.plot_description['comment'] = self.create_comment(Option[data.get('m','li').upper()],status,comment_params)
                self.draw_graph(data['inc'])
                return comment_params

        def get_gvns_and_ts_animation(self, i:int, log_data: list, comment_params: CommentParameters):

                data = log_data[i]
                status = data.get('status','')
                sol = data.get('sol',[])

                if status == 'start' and (data.get('m') == 'ch' or i==0):
                        comment_params.n = len(self.graph.nodes())
                        comment_params.m = len(self.graph.edges())
                        self.plot_description['comment'] = self.create_comment(Option.CH,status,comment_params)
                        log_data[i]['best'] = log_data[i]['obj'] = 0
                        self.draw_graph()
                        return True
                #set color of nodes and edges
                remove,add = self.get_removed_and_added_nodes(i,log_data)
                #self.color_nodes_and_edges(data,remove,add)
                nx.set_node_attributes(self.graph, {n:self.blue for n in sol}, 'color')
                nx.set_node_attributes(self.graph, {n:{'label':'+','color':self.green if status == 'start' else self.blue} for n in add})
                nx.set_node_attributes(self.graph, {n:{'label':'-','color':self.red if status == 'start' else self.grey} for n in remove})
                nx.set_node_attributes(self.graph, {n:self.yellow for n in data.get('inc') if data.get('better',False)}, 'color')
                nx.set_edge_attributes(self.graph, {e:'black' for n in add for e in self.graph.edges(n)} if status == 'end' else {}, 'color')

                # fill parameters for plot description
                comment_params.remove = remove
                comment_params.add = add
                comment_params.par = data.get('par',1)
                comment_params.gain = data["obj"] - log_data[i-1]["obj"]
                comment_params.no_change = not (add or remove)
                comment_params.better = data.get('better',False)
                return False


        def get_removed_and_added_nodes(self, i, log_data):
                status = log_data[i].get('status','start')
                compare_i = i + (status == 'start') - (status == 'end')
                compare_sol = set(log_data[compare_i].get('sol'))
                current_sol = set(log_data[i].get('sol',[]))
                remove = current_sol - compare_sol if status == 'start' else compare_sol - current_sol
                add = current_sol - compare_sol if status == 'end' else compare_sol - current_sol
                return remove,add


        def get_grasp_animation(self, i:int, log_data: list):
                if log_data[i].get('m','') in ['ch', 'li']:
                        comment = self.get_gvns_animation(i,log_data)
                        return comment

                comment_params = CommentParameters()
                data = log_data[i] 
                status = data.get('status','')

                if status == 'start' or status == 'end':
                        comment_params.better = data.get('better',False)
                        self.plot_description['comment'] = self.create_comment(Option.RGC,status,comment_params)
                        nx.set_node_attributes(self.graph, {n:self.yellow if data.get('better',False) else self.blue for n in data.get('sol') if status == 'end'}, name='color')
                        self.draw_graph(data.get('inc') if status == 'end' else [])
                        return comment_params

                nx.set_node_attributes(self.graph, data.get('cl',{}), 'label')
                nx.set_node_attributes(self.graph, {n: self.green if data.get('sel',-1) == n else self.blue for n in data.get('sol', [])}, name='color')
                selected = set() if not data.get('sel',False) else set(self.graph.neighbors(data.get('sel')))
                n_unsel = selected.intersection(set(data['cl'].keys()))
                nx.set_node_attributes(self.graph, {n:self.orange for n in n_unsel},'color')
                nx.set_edge_attributes(self.graph, {(data.get('sel',n),n):'black' for n in n_unsel}, 'color')

                par = data.get('par',1)
                mn = min(data.get('cl').values())
                mx = max(data.get('cl').values())
                if type(par) == int:
                        comment_params.k = par
                else:
                        comment_params.alpha = par
                        comment_params.thres = round(mn + par * (mx-mn),2)
                comment_params.gain = 1

                j = i
                while not (log_data[j]['status'] in ['start','end']):
                        j -= 1

                self.plot_description.update({'best': log_data[j]['best'], 'obj':len(data.get('sol'))})
                self.plot_description['comment'] = self.create_comment(Option.RGC,status,comment_params)
                self.draw_graph(data.get('rcl',[]), sel_color='black')
                return comment_params

        def get_ts_animation(self, i:int, log_data: list):

                data = log_data[i]
                status = data.get('status','')
                comment_params = CommentParameters()
                done = self.get_gvns_and_ts_animation(i,log_data,comment_params)
                if done:
                        return comment_params

                tabu_list = data.get('tabu',[])
                asp_nodes = set()
                for ta in tabu_list:
                        tabu_nodes = list(ta[0])
                        life = ta[1]
                        nx.set_node_attributes(self.graph, {n: {'label':life,'tabu':True} for n in tabu_nodes})
                        if data.get('status','') == 'start' and set(tabu_nodes).issubset(comment_params.add):
                                asp_nodes = asp_nodes.union(set(tabu_nodes).intersection(comment_params.add))

                comment_params.asp = len(asp_nodes) > 0
                comment_params.ll = data.get('ll',0)

                self.plot_description['comment'] = self.create_comment(Option.CH if data.get('m').startswith('ch') else Option.TL,status,comment_params)
                self.draw_graph(data.get('inc',[]))
                return comment_params

        def get_sa_animation(self, i: int, log_data: list):
                raise NotImplementedError

        def add_legend(self):

                legend_elements = tuple()
                description = tuple()

                legend_elements = (
                        Line2D([0], [0],linestyle='none'),
                        Line2D([0],[0],marker='o', color='w',
                                markerfacecolor=self.red, markersize=13),
                        Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=self.yellow, markersize=13),
                        Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=self.blue, markersize=13),
                        Line2D([0],[0],marker='o', color='w',
                                markerfacecolor=self.green, markersize=13),
                        Line2D([0], [0], marker='o', linestyle='none',
                                markerfacecolor='w',markeredgecolor=self.yellow,markeredgewidth=2, markersize=11)     
                        )
                description = ('','','','current solution','remove/add node', 'best solution')

                if self.algorithm == Algorithm.TS:
                        legend_elements = list(legend_elements)
                        legend_elements.insert(3,(Line2D([0], [0],linestyle='none')))
                        legend_elements.append(Line2D([0],[0],marker='X', color='w',
                                        markerfacecolor='black', markersize=13))
                        legend_elements = tuple(legend_elements)
                        description = list(description)
                        description.insert(3,'')
                        description.append('tabu attribute')
                        description = tuple(description)


                if self.algorithm == Algorithm.GRASP:
                        legend_elements = list(legend_elements)
                        legend_elements.insert(3,(Line2D([0], [0],linestyle='none')))
                        legend_elements.append(Line2D([0],[0],marker='o', color='w',
                                        markerfacecolor=self.orange, markersize=13))
                        legend_elements = tuple(legend_elements)
                        description = list(description)
                        description.insert(3,'')
                        description.append('blocked neighbor')
                        description = tuple(description)

                self.ax.legend(legend_elements, description,  ncol=2, handlelength=1, borderpad=0.7, columnspacing=0, loc='lower left')

        def reset_graph(self):
                nx.set_node_attributes(self.graph, self.grey, name='color')
                nx.set_edge_attributes(self.graph, self.grey, name='color')
                nx.set_node_attributes(self.graph, '', name='label')
                nx.set_node_attributes(self.graph,False,name='tabu')

        def draw_graph(self, pos_change: list() = [], sel_color='gold'):
                self.ax.clear()
                self.ax.set_ylim(bottom=-1.5,top=1.3)
                self.ax.set_xlim(left=-1.1,right=1.1)
                for pos in ['right', 'top', 'bottom', 'left']: 
                        self.ax.spines[pos].set_visible(False) 

                nodelist = self.graph.nodes()

                pos = nx.get_node_attributes(self.graph,'pos')

                color = [self.graph.nodes[n]['color'] for n in nodelist]
                linewidth = [3 if n in pos_change else 0 for n in nodelist]
                lcol = [sel_color for _ in nodelist]
                labels = {v:l for v,l in nx.get_node_attributes(self.graph,'label').items() if not self.graph.nodes[v]['tabu']}
                edges = list(self.graph.edges())
                e_cols = [self.graph.edges[e]['color'] for e in edges]
                
                nx.draw_networkx(self.graph, pos, nodelist=nodelist, with_labels=True, labels=labels,font_weight='bold', font_size=14, ax=self.ax,  
                                node_color=color, edgecolors=lcol, edgelist=edges, edge_color=e_cols, linewidths=linewidth, node_size=500)

                # drawings for tabu search
                nodes_labels_tabu = {v:l for v,l in nx.get_node_attributes(self.graph,'label').items() if self.graph.nodes[v]['tabu']}
                tabu_nodes = list(nodes_labels_tabu.keys())
                if len(tabu_nodes) == 0:
                        return
                x_pos = {k:[v[0]-0.03,v[1]-0.03] for k,v in pos.items()}
                nx.draw_networkx_nodes(self.graph, x_pos, nodelist=tabu_nodes,node_color='black', node_shape='X', node_size=150,ax=self.ax)
                nx.draw_networkx_labels(self.graph, pos, labels=nodes_labels_tabu, ax=self.ax, font_size=12, font_weight='bold', 
                                        font_color='black',horizontalalignment='left',verticalalignment='baseline')






class MAXSATDraw(Draw):

        comments = {
                Option.CH:{
                        'start': lambda params: f'{params.n} variables, {params.m} clauses',
                        'end': lambda params: f'initial solution={InitSolution(params.par).name}'
                },
                Option.LI: {
                        'start': lambda params: f'k={params.par}, flipping {len(params.flip)} variable(s)',
                        'end': lambda params: f'objective gain={params.gain}{", no improvement - reached local optimum" if params.no_change else ""}{", found new best solution" if params.better else ""}'
                },
                Option.SH: {
                        'start': lambda params: f'k={params.par}, flipping {len(params.flip)} variable(s)',
                        'end': lambda params: f'objective gain={params.gain}{", found new best solution" if params.better else ""}'
                },
                Option.RGC:{
                        'start': lambda params: 'start with empty solution',
                        'end': lambda params: f'created complete solution{", found new best solution" if params.better else ""}',
                        'cl': lambda params: 'candidate list=number of additionally fulfilled clauses',
                        'rcl': lambda params: f'restricted candidate list=' + (f'{params.k}-best' if params.k else f'alpha: {params.alpha}, threshold: {params.thres}'),
                        'sel': lambda params: f'selection=random, objective gain={params.gain}'
                },
                Option.TL:{
                        'start': lambda params: f'k={params.par} flipping {len(params.flip)} variable(s){", apply aspiration criterion" if params.asp else ""}',
                        'end': lambda params: f'size of tabu list={params.ll}, objective gain={params.gain}{", all possible flips are tabu" if params.no_change else ""}{", found new best solution" if params.better else ""}'
                }
                }

        def create_comment(self, option: Option, status: str, params: CommentParameters):
                # TODO create comments according to log granularity

                return self.comments[option][status](params)

        def __init__(self, prob: Problem, alg: Algorithm, instance, log_granularity: Log):
                super().__init__(prob,alg,instance,log_granularity)


        def init_graph(self, instance):

                n = instance.n #variables
                m = instance.m #clauses
                clauses = [i for i in range(1,1+m)]
                variables = [i + m for i in range(1,1+n)]
                incumbent = [i + n for i in variables]

                #sort clauses by barycentric heuristic (average of variable positions)
                def avg_clause(clause):
                        i = clause-1
                        vl = list(map(abs,instance.clauses[i]))
                        l = len(instance.clauses[i])
                        return sum(vl)/l

                clauses_sorted = sorted(clauses, key=avg_clause)

                ### calculate positions for nodes
                step = 2/(n+1)
                y_pos = 0.2 if self.algorithm == Algorithm.TS else 0.4
                pos = {v:[-1 + i*step, y_pos] for i,v in enumerate(variables, start=1)}
                pos.update({i:[-1 +j*step,y_pos+0.2] for j,i in enumerate(incumbent, start=1)})
                step = 2/(m+1)
                pos.update({c:[-1+ i*step,-0.4] for i,c in enumerate(clauses_sorted, start=1)})

                # create nodes with data
                v = [(x, {'type':'variable', 'nr':x-m, 'color':self.grey, 'pos':pos[x], 'label':'','usage':clause,'alpha':1.,'tabu':False}) for x,clause in enumerate(instance.variable_usage, start=m+1)]  #[m+1,...,m+n]
                c = [(x, {'type':'clause', 'nr':x, 'color': self.grey, 'pos':pos[x], 'label':f'c{x}', 'clause':clause}) for x,clause in enumerate(instance.clauses, start=1)]   #[1,..,m]
                i = [(x, {'type':'incumbent', 'nr':x-m-n, 'color':self.white, 'pos':pos[x], 'label':f'x{x-m-n}','alpha':1.}) for x in incumbent]               #[1+m+n,...,2n+m]

                # create graph by adding nodes and edges
                graph = nx.Graph()
                graph.add_nodes_from(c)
                graph.add_nodes_from(v)
                graph.add_nodes_from(i)

                for i,cl in enumerate(instance.clauses, start=1):
                        graph.add_edges_from([(i,abs(x)+ m,{'style':'dashed' if x < 0 else 'solid', 'color':self.grey}) for x in cl])

                return graph

        def get_gvns_animation(self, i:int, log_data:list):
                data = log_data[i]
                status = data.get('status','')
                comment_params = CommentParameters()
                done, lit_info = self.get_gvns_and_ts_animation(i,log_data,comment_params)
                if done:
                        return comment_params
                
                self.plot_description['comment'] = self.create_comment(Option[data.get('m','li').upper()],status,comment_params)
                flipped_nodes = [] if status == 'end' else comment_params.flip
                flipped_nodes += [n for n,t in self.graph.nodes(data='type') if t=='incumbent'] if data.get('better',False) else []
                self.draw_graph(flipped_nodes + list(comment_params.add.union(comment_params.remove)))
                self.write_literal_info(lit_info)
                self.add_sol_description(i,data)
                return comment_params

        def get_gvns_and_ts_animation(self, i:int, log_data: list, comment_params: CommentParameters):

                incumbent = [i for i,t in self.graph.nodes(data='type') if t=='incumbent']
                variables = [i for i,t in self.graph.nodes(data='type') if t=='variable']
                clauses = [i for i,t in self.graph.nodes(data='type') if t=='clause']
                data = log_data[i]
                status = data.get('status','')

                if status == 'start' and (data.get('m') == 'ch' or i==0):
                        comment_params.n = len(variables)
                        comment_params.m = len(clauses)
                        self.plot_description['comment'] = self.create_comment(Option.CH,status, comment_params)
                        log_data[i]['best'] = log_data[i]['obj'] = 0
                        nx.set_node_attributes(self.graph,{n:'' for n,t in self.graph.nodes(data='type') if t=='incumbent'}, name='label')
                        self.draw_graph([])
                        return True, {}
                if data.get('m') == 'ch' and status=='end':
                        log_data[0]['sol'] = [-1 for _ in data['sol']]

                nx.set_node_attributes(self.graph, {k: self.red if data['inc'][self.graph.nodes[k]['nr']-1] == 0 else self.blue for k in incumbent}, name='color')
                nx.set_node_attributes(self.graph, {k: self.red if data['sol'][self.graph.nodes[k]['nr']-1] == 0 else self.blue for k in variables}, name='color')
                added, removed, pos_literals = self.color_and_get_changed_clauses(i, log_data, status == 'start')
                flipped_nodes = self.get_flipped_variables(i,log_data)

                comment_params.flip = flipped_nodes
                comment_params.add = added
                comment_params.remove = removed
                comment_params.par = data.get('par',1)
                comment_params.gain = data["obj"] - log_data[i-1]["obj"]
                comment_params.better = data.get('better',False)
                comment_params.no_change = len(flipped_nodes) == 0

                return False, pos_literals
                



        def get_flipped_variables(self, i: int, log_data: list):
                info = log_data[i]
                comp = i + (info['status'] == 'start') - (info['status'] == 'end')
                flipped_variables = [n for n,v in enumerate(info['sol'],start=1) if v != log_data[comp]['sol'][n-1]]
                flipped_variables = [n for n, data in self.graph.nodes(data=True) if data['nr'] in flipped_variables and data['type']=='variable']

                fullfilled_clauses = [n for n in self.graph.nodes if self.graph.nodes[n]['type'] == 'clause' and self.graph.nodes[n]['color']==self.green]
                for c in fullfilled_clauses:
                        for v in self.graph.neighbors(c):
                                col = self.graph.nodes[v]['color']
                                style = self.graph.edges[(v,c)]['style']
                                if (col == self.blue and style=='solid') or (col==self.red and style=='dashed'):
                                        self.graph.edges[(v,c)]['color'] = self.darkgrey
        
                if info['status'] == 'end' and info.get('m','') != 'ch':
                        nx.set_edge_attributes(self.graph, {edge: 'black' for edge in self.graph.edges() if set(edge) & set(flipped_variables)}, 'color')
                        #nx.set_edge_attributes(self.graph, {edge: self.blue if self.graph.edges[edge]['style'] == 'solid' else self.red for edge in self.graph.edges() if set(edge) & set(flipped_variables)}, 'color')
                return flipped_variables

        def color_and_get_changed_clauses(self,value,log_data, start=True):

                i = value - (not start)
                pos_start,literals = self.color_clauses_and_count_literals(log_data[i], literals=start)

                if start:
                        return set(),set(),literals

                pos_end,literals = self.color_clauses_and_count_literals(log_data[value])

                return pos_end-pos_start, pos_start-pos_end,literals

        def color_clauses_and_count_literals(self, log_data: dict, literals=True):

                clauses = nx.get_node_attributes(self.graph, 'clause')
                fulfilled = set()
                num_literals = dict()

                for n,clause in clauses.items():
                        for v in clause:
                                if log_data['sol'][abs(v)-1] == (1 if v > 0 else 0):
                                        fulfilled.add(n)
                                        self.graph.nodes[n]['color'] = self.green
                                        break
                                else:
                                        self.graph.nodes[n]['color'] = self.grey

                if literals:
                        num_literals = dict.fromkeys(clauses,0)
                        for fc in fulfilled:
                                for v in clauses[fc]:
                                        if log_data['sol'][abs(v)-1] == (1 if v > 0 else 0):
                                                num_literals[fc] += 1
                
                return fulfilled,num_literals

        def write_literal_info(self, literal_info: dict):

                literal_info.update( {n:(v,[self.graph.nodes[n]['pos'][0],self.graph.nodes[n]['pos'][1]-0.1]) for n,v in literal_info.items()})

                for _,data in literal_info.items():
                        self.ax.text(data[1][0],data[1][1],data[0],{'color': 'black', 'ha': 'center', 'va': 'center', 'fontsize':'small'})

        def get_grasp_animation(self, i:int, log_data: list):
                if log_data[i].get('m','') in ['ch', 'li']:
                        comment = self.get_gvns_animation(i,log_data)
                        return comment

                incumbent = [i for i,t in self.graph.nodes(data='type') if t=='incumbent']
                variables = [i for i,t in self.graph.nodes(data='type') if t=='variable']
                clauses = [i for i,t in self.graph.nodes(data='type') if t=='clause']
                data = log_data[i]
                status = data.get('status','')
                comment_params = CommentParameters()
                comment_params.better = data.get('better',False)
                if status == 'end':
                        self.plot_description['comment'] = self.create_comment(Option.RGC,status,comment_params)
                        nx.set_node_attributes(self.graph, {k: self.red if data['inc'][self.graph.nodes[k]['nr']-1] == 0 else self.blue for k in incumbent}, name='color')
                        nx.set_node_attributes(self.graph, {k: self.red if data['sol'][self.graph.nodes[k]['nr']-1] == 0 else self.blue for k in variables}, name='color')
                        _,_,pos_literals = self.color_and_get_changed_clauses(i,log_data)
                        self.get_flipped_variables(i,log_data)
                        self.draw_graph(incumbent if data.get('better',False) else [])
                        self.write_literal_info(pos_literals)
                        self.add_sol_description(i,data)
                        return comment_params

                nx.set_node_attributes(self.graph,{n:'' for n,t in self.graph.nodes(data='type') if t=='incumbent'}, name='label')

                if status == 'start':
                        self.plot_description['comment'] = self.create_comment(Option.RGC,status,comment_params)
                        log_data[i]['obj'] = 0
                        self.draw_graph([])
                        self.write_literal_info(dict.fromkeys(clauses,0))
                        self.add_sol_description(i,data)
                        return comment_params

                #map variable ids to node ids
                keys = {v:k for k,v in nx.get_node_attributes(self.graph,'nr').items() if self.graph.nodes[k]['type'] == 'variable'}
                rcl = [np.sign(v)*keys[abs(v)] for v in data.get('rcl',[])]
                cl = {np.sign(k)*keys[abs(k)]:v for k,v in data['cl'].items()}
                sel = keys.get(abs(data.get('sel',0)),0) * np.sign(data.get('sel',0))

                not_sel = set(abs(v) for v in cl.keys())
                selected = set(variables).difference(not_sel)
                selected.add(abs(sel))

                #set colors for edges and variables
                #set colors for clauses
                added,_,pos_literals = self.color_and_get_changed_clauses(i,log_data,start=not (data['status'] == 'sel'))
                nx.set_node_attributes(self.graph, {n: self.red if data['sol'][self.graph.nodes[n]['nr']-1] == 0 else self.blue for n in selected if n != 0}, name='color')
                self.get_flipped_variables(i,log_data)
                nx.set_edge_attributes(self.graph, {edge: 'black' for edge in self.graph.edges() if abs(sel) in edge}, 'color')
                

                mx = max(data['cl'].values())
                mn = min(data['cl'].values())
                par = data.get('par', 0)
                if type(par) == int:
                        comment_params.k = par
                else:
                        comment_params.alpha = par
                        comment_params.thres = round(mx - par * (mx-mn),2)
                comment_params.gain = len(added)

                j = i
                while not log_data[j]['status'] in ['start','end']:
                        j = j-1
                self.plot_description.update({
                        'best': log_data[j]['best'], 
                        'obj': sum(p > 0 for p in pos_literals.values())
                        })

                # draw graph and print textual information
                self.plot_description['comment'] = self.create_comment(Option.RGC,status, comment_params)
                self.draw_graph(([abs(sel)] if sel != 0 else []) + list(added))
                self.write_literal_info(pos_literals)
                self.write_cl_info(cl, rcl, sel)
                self.add_sol_description(i,data)
                return comment_params


        def get_ts_animation(self, i:int, log_data:list):

                data = log_data[i]
                status = data.get('status','start')
                comment_params = CommentParameters()
                done, lit_info = self.get_gvns_and_ts_animation(i,log_data,comment_params)
                if done:
                        return comment_params
                        
                flipped_nodes = comment_params.flip 
                
                tabu_list = data.get('tabu',[])
                asp = False
                for ta in tabu_list:
                        tabu_var = list(map(abs,ta[0]))
                        life = ta[1]
                        nodes = [n for n,t in self.graph.nodes(data='type') if t=='variable' and self.graph.nodes[n]['nr'] in tabu_var]
                        nx.set_node_attributes(self.graph, {n: {'tabu':True,'label':str(life)} for n in nodes})
                        if set(nodes).issubset(set(flipped_nodes)):
                                asp = True
                
                comment_params.asp = asp
                comment_params.ll = data.get('ll',0)
                
                self.plot_description['comment'] = self.create_comment(Option.CH if data.get('m').startswith('ch') else Option.TL,status,comment_params)
                flipped_nodes = [] if status == 'end' else comment_params.flip
                flipped_nodes += [n for n,t in self.graph.nodes(data='type') if t=='incumbent'] if data.get('better',False) else []
                self.draw_graph(flipped_nodes + list(comment_params.add.union(comment_params.remove)))
                self.write_literal_info(lit_info)
                self.add_sol_description(i,data)
                return comment_params


        def write_cl_info(self, cl: dict(), rcl: list(), sel: int):

                cl_positions = {n:pos for n,pos in nx.get_node_attributes(self.graph, 'pos').items() if self.graph.nodes[n]['type'] == 'variable'}

                col = {1:self.blue,-1:self.red,0:self.grey}

                for k,v in cl.items():
                        pos = cl_positions[abs(k)]
                        c = col[np.sign(k)] if len(rcl)==0 or k in rcl else col[0]
                        bbox = dict(boxstyle="circle",fc="white", ec=c, pad=0.2) if k == sel else None
                        self.ax.text(pos[0],pos[1]+0.2+(0.05*np.sign(k)), v, {'color': c, 'ha': 'center', 'va': 'center','fontweight':'bold','bbox': bbox})

        def get_sa_animation(self, i: int, log_data: list):
                raise NotImplementedError

        def add_legend(self):
                legend_elements = (
                                        Line2D([0], [0], marker='s', linestyle='none', markeredgewidth=0,
                                                markerfacecolor=self.blue, markersize=11),
                                        Line2D([0], [0], marker='o', color='w',
                                                markerfacecolor=self.grey, markersize=13), 
                                        Line2D([0], [0], marker='o', linestyle='none',
                                                markerfacecolor='w',markeredgecolor=self.yellow,markeredgewidth=2, markersize=11),

                                        Line2D([0], [0], marker='s', linestyle='none', markeredgewidth=0,
                                                markerfacecolor=self.red, markersize=11),
                                        Line2D([0], [0], marker='o', color='w',
                                                markerfacecolor=self.green, markersize=13),
                                        Line2D([0], [0], marker='s', linestyle='none',
                                                markerfacecolor='w',markeredgecolor=self.yellow,markeredgewidth=2, markersize=11),
                                        )
                description = ('','','','true/false variable','unfullfilled/fullfilled clause','change in clauses/solution')

                if self.algorithm == Algorithm.TS:
                        legend_elements = list(legend_elements)
                        legend_elements.insert(3, Line2D([0], [0],linestyle='none'))
                        legend_elements.append(Line2D([0], [0], marker='X', color='w',
                                                markerfacecolor='black', markersize=13))
                        legend_elements = tuple(legend_elements)
                        description = list(description)
                        description.insert(3,'')
                        description.append('tabu attribute')
                        description = tuple(description)

                self.ax.legend(legend_elements, description,  ncol=2, handlelength=1, borderpad=0.7, columnspacing=0, loc='lower left')


        def reset_graph(self):
                nx.set_node_attributes(self.graph, {n: self.grey if self.graph.nodes[n]['type'] != 'incumbent' else self.white for n in self.graph.nodes()}, name='color')
                nx.set_edge_attributes(self.graph, self.grey, name='color')
                nx.set_node_attributes(self.graph, 1., name='alpha')
                nx.set_node_attributes(self.graph, {n: f'x{self.graph.nodes[n]["nr"]}' for n in self.graph.nodes() if self.graph.nodes[n]['type']=='incumbent'}, name='label')
                nx.set_node_attributes(self.graph,False,name='tabu')

        def draw_graph(self, pos_change):
                self.ax.clear()
                self.ax.set_ylim(bottom=-1,top=1.2)
                self.ax.set_xlim(left=-1,right=1)
                for pos in ['right', 'top', 'bottom', 'left']: 
                        self.ax.spines[pos].set_visible(False) 

                var_inc_nodes = [n for n,t in nx.get_node_attributes(self.graph, 'type').items() if  t in ['variable', 'incumbent']]
                var_inc_color = [self.graph.nodes[n]['color'] for n in var_inc_nodes]
                var_inc_lcol = ['black' if self.graph.nodes[n]['type'] == 'variable' else self.yellow for n in var_inc_nodes if n in pos_change]
                var_inc_lw = [4 if n in pos_change else 0 for n in var_inc_nodes]
                var_inc_alpha = [self.graph.nodes[n]['alpha'] for n in var_inc_nodes]

                cl_nodes = [n for n,t in nx.get_node_attributes(self.graph, 'type').items() if  t =='clause']
                cl_color = [self.graph.nodes[n]['color'] for n in cl_nodes]
                cl_lcol = [self.yellow for n in cl_nodes]
                cl_lw = [3 if n in pos_change else 0 for n in cl_nodes]

                edges = list(self.graph.edges())
                # draw gray and black edges seperately to avoid overpainting black edges
                e_list_gray = [e for i,e in enumerate(edges) if self.graph.edges[e]['color'] !='black']
                e_color_gray = [self.graph.edges[e]['color'] for e in e_list_gray]
                e_style_gray = [self.graph.edges[e]['style'] for e in e_list_gray]

                e_list_black = [e for i,e in enumerate(edges) if self.graph.edges[e]['color'] =='black']
                e_style_black = [self.graph.edges[e]['style'] for e in e_list_black]

                var_labels = {v:l for v,l in nx.get_node_attributes(self.graph,'label').items() if self.graph.nodes[v]['type'] == 'incumbent'}

                pos = nx.get_node_attributes(self.graph, 'pos')

                nx.draw_networkx_nodes(self.graph, pos, nodelist=var_inc_nodes, node_color=var_inc_color, edgecolors=var_inc_lcol, 
                                        alpha=var_inc_alpha, linewidths=var_inc_lw, node_shape='s', node_size=500,ax=self.ax)
                nx.draw_networkx_nodes(self.graph, pos, nodelist=cl_nodes, node_color=cl_color, edgecolors=cl_lcol, linewidths=cl_lw, node_shape='o',node_size=150,ax=self.ax)
                nx.draw_networkx_edges(self.graph, pos, edgelist=e_list_gray, style=e_style_gray,ax=self.ax, edge_color=e_color_gray)
                nx.draw_networkx_edges(self.graph, pos, edgelist=e_list_black, style=e_style_black,ax=self.ax, edge_color='black')
                nx.draw_networkx_labels(self.graph, pos, labels=var_labels,ax=self.ax)

                # drawings for tabu search
                var_labels_tabu = {v:l for v,l in nx.get_node_attributes(self.graph,'label').items() if self.graph.nodes[v]['tabu']}
                tabu_nodes = list(var_labels_tabu.keys())
                x_pos = {k:[v[0]-0.04,v[1]-0.04] for k,v in pos.items()}
                nx.draw_networkx_nodes(self.graph, x_pos, nodelist=tabu_nodes,node_color='black', node_shape='X', node_size=200,ax=self.ax)
                nx.draw_networkx_labels(self.graph, pos, labels=var_labels_tabu, ax=self.ax, font_size=14, font_weight='bold', 
                                        font_color='black',horizontalalignment='left',verticalalignment='baseline')
                
        def add_sol_description(self, i, data: dict):
                best = 'best' if i > 0 and data.get('status','') in ['start','end']  else ''
                inc_pos = [pos for n,pos in nx.get_node_attributes(self.graph, 'pos').items() if self.graph.nodes[n]['type'] == 'incumbent'][0]
                curr_pos = [pos for n,pos in nx.get_node_attributes(self.graph, 'pos').items() if self.graph.nodes[n]['type'] == 'variable'][0]
                self.ax.text(-1,inc_pos[1], best)
                self.ax.text(-1,curr_pos[1], 'current')


class TSPDraw(Draw):
        comments = {
                Option.CH:{
                        'start': lambda params: f'Constructing initial solution',
                        'end': lambda params: f'initial solution={InitSolution(params.par).name}'
                },
                Option.LI: {
                        'start': lambda params: f'k={params.par}, {params.par}-opt search',
                        'end': lambda params: f'objective gain={params.gain}{", no improvement - reached local optimum" if params.no_change else ""}{", found new best solution" if params.better else ""}'
                },
                Option.SH: {
                        'start': lambda params: f'k={params.par}, swapping two nodes k times',
                        'end': lambda params: f'objective gain={params.gain}{", found new best solution" if params.better else ""}'
                },
                Option.SA: {
                        'start': lambda params: f'Choosing a random 2-opt neighborhood move', # todo: stopped here
                        'end': lambda params: f'T={round(params.temp, 2)} ' + (
                                f'Improved solution with delta={params.delta}' if params.accept and params.delta < 0 
                                else f'Worse solution with delta={params.delta} accepted, P < {round(math.exp(-abs(params.delta) / params.temp), 5)}' 
                                if params.accept and params.delta >= 0 else 
                                f'Worse solution with delta={params.delta} rejected, P > {round(math.exp(-abs(params.delta) / params.temp), 5)}'
                        )
                }
                }




        def init_graph(self, instance):
                graph = nx.Graph()
                nodes = [(i, {'pos': instance.coordinates[i]}) for i in list(instance.coordinates.keys())]
                graph.add_nodes_from(nodes)
                return graph


        def create_comment(self, option: Option, status: str, params: CommentParameters):
                # TODO create comments according to log granularity
                return self.comments[option][status](params)

        def get_grasp_animation(self, i: int, log_data: list):
                raise NotImplementedError
                pass


        def get_gvns_animation(self, i:int, log_data:list):
                data = log_data[i]
                edges = self.edges_from_tour(data['sol'])
                added = removed = []
                if i > 0:
                        added, removed = self.get_added_removed_edges(data['sol'], log_data[i-1]['sol'])

                self.draw_graph(edges, added, removed)

                comment_params = CommentParameters()           
                comment_params.par = data.get('par')
                comment_params.gain = log_data[i-1].get('obj') - data.get('obj')
                comment_params.no_change = (added == [])

                self.plot_description['comment'] = self.create_comment(Option[data.get('m','li').upper()],data.get('status'),comment_params)
                return comment_params

        def edges_from_tour(self, tour):
                edges = []
                for j in range(len(tour)):
                        if tour[j] < tour[(j+1) % len(tour)]: # ensures nodes in edge tuple is ascending, useful for set() manipulation
                                edges.append((tour[j], tour[(j+1) % len(tour)]))
                        else:
                                edges.append((tour[(j+1) % len(tour)],tour[j]))
                return edges

        def get_added_removed_edges(self, tour, prev_tour):
                edges = set(self.edges_from_tour(tour))
                prev_edges = set(self.edges_from_tour(prev_tour))

                added = edges.difference(prev_edges)
                removed = prev_edges.difference(edges)
                return list(added), list(removed)


        def get_ts_animation(self, i:int, log_data:list):
                raise NotImplementedError
                pass


        def get_sa_animation(self, i: int, log_data: list):
                data = log_data[i]
                edges = self.edges_from_tour(data['sol'])
                added = removed = []

                comment_params = CommentParameters()           
                comment_params.par = data.get('par')
                comment_params.gain = log_data[i-1].get('obj') - data.get('obj')
                comment_params.no_change = (added == [])


                if Option[data['m'].upper()] == Option.SA:
                        comment_params.temp = data.get('temp', log_data[i+1])
                        comment_params.delta = data.get('delta', 0)
                        comment_params.accept = data.get('accept', True)
                        if data.get('accept', True):
                                added, removed = self.get_added_removed_edges(data['sol'], log_data[i-1]['sol'])
                        else:
                                removed, added = self.get_added_removed_edges(data['disc_sol'], data['sol'])
                        self.draw_graph(edges, added, removed, data.get('accept', True))
                else:
                        self.draw_graph(edges, added, removed)
                        
                self.plot_description['comment'] = self.create_comment(Option[data.get('m').upper()], data.get('status'), comment_params)
                return comment_params

        def add_legend(self):
                legend_elements = (
                                Line2D([0], [0], marker=None, linestyle='-', color=self.grey),
                                Line2D([0], [0], marker=None, linestyle='--', color=self.green), 
                                Line2D([0], [0], marker=None, linestyle='--', color=self.red),
                                Line2D([0], [0], marker=None, linestyle='--', color=self.yellow),
                                Line2D([0], [0], marker=None, linestyle='--', color=self.blue),
                )
                description = ('Tour','Added','Removed', 'Rejected added edges', 'Rejected removed edges')

                self.ax.legend(legend_elements, description,  ncol=1, handlelength=1, borderpad=0.7, columnspacing=0, loc='lower left')



        def reset_graph(self):
                self.graph.remove_edges_from(list(self.graph.edges()))

        def draw_graph(self, edges = [], added = [], removed = [], accept=True):
                """
                
                """
                self.ax.clear()
                self.ax.set_aspect('equal')

                if accept:
                        added_color = self.green
                        removed_color = self.red
                        added_style = "dashed"
                        removed_style = "dashed"
                else:
                        added_color = self.blue
                        removed_color = self.yellow
                        added_style = "dashed"
                        removed_style = "dashed"

                pos = nx.get_node_attributes(self.graph, 'pos')
                nx.draw_networkx_nodes(self.graph, pos, ax = self.ax, node_color = self.black, node_size = 75)
                nx.draw_networkx_edges(self.graph, pos, ax = self.ax, edgelist = edges, edge_color = self.grey, width = 3)
                nx.draw_networkx_edges(self.graph, pos, ax = self.ax, edgelist = added,
                                        style = added_style, edge_color = added_color, width = 1.5)
                nx.draw_networkx_edges(self.graph, pos, ax = self.ax, edgelist = removed,
                                        style = removed_style, edge_color = removed_color, width = 1.5)


def get_visualisation(prob: Problem, alg: Algorithm, instance, log_granularity: Log):
    prob_class = globals()[prob.name + 'Draw']
    prob_instance = prob_class(prob, alg, instance, log_granularity)
    return prob_instance



    