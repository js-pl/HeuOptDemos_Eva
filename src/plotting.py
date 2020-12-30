import sys
sys.path.append('..\\HeuOptDemos_Eva')
import matplotlib.image as mpimg
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import statistics
from pymhlib.demos.maxsat import MAXSATInstance
from pymhlib.demos.misp import MISPInstance
from src.problems import Problem, Algorithm, Option
from abc import ABC, abstractmethod
from src.logdata import read_from_logfile


plt.rcParams['figure.figsize'] = (12,5)
plt.rcParams['figure.dpi'] = 80
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False

pc_dir = 'pseudocode'

class Draw(ABC):

        phases = {'ch':'Construction', 'li': 'Local Search', 'sh': 'Shaking', 'rgc': 'Randomized Greedy Construction', 
                        'cl':'Candidate List', 'rcl': 'Restricted Candidate List', 'sel':'Selection from RCL'}

        plot_description = {'phase': '',
                                'comment': [],
                                'best': 0,
                                'obj': 0,
                                }

        def __init__(self, prob: Problem, alg: Algorithm, instance):
                self.problem = prob
                self.algorithm = alg
                self.graph = self.init_graph(instance)

                # create figure
                plt.close() # close any previously drawn figures
                self.fig, (self.ax, self.img_ax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, 
                        num = f'Solving {prob.value} with {alg.value}')
                self.img_ax.axis('off')
                self.ax.axis('off')

        @abstractmethod
        def init_graph(self, instance):
                return None

        def get_animation(self, i: int, log_data: list):
                if log_data[i+2].get('m','').startswith('rgc') or log_data[i+2].get('status','') in ['cl', 'rcl', 'sel']:
                        self.get_grasp_animation(i,log_data[2:])
                else:
                        self.get_gvns_animation(i,log_data[2:])

        @abstractmethod
        def get_grasp_animation(self, i: int, log_data: list):
                pass

        @abstractmethod
        def get_gvns_animation(self, i:int, log_data:list):
                pass

        def add_description(self, log_info: dict):

                if log_info.get('status') == 'start' and log_info.get('m').startswith('ch'):
                        log_info['best'] = 0
                        log_info['obj'] = 0
                else:
                        self.plot_description['phase'] = self.phases.get(log_info['status'],'') + self.phases.get(log_info.get('m','   '),'')


                self.ax.text(0,1, '\n'.join((
                '%s: %s' % (self.plot_description['phase'], ', '.join(self.plot_description['comment']) ),
                'Best Objective: %d' % (log_info.get('best',self.plot_description['best']), ),
                'Current Objective: %d' % (log_info.get('obj',self.plot_description['obj']),))), horizontalalignment='left', verticalalignment='top', transform=self.ax.transAxes)

                self.load_pc_img(log_info)
                self.plot_description.update({'phase': '', 'comment': [], 'best':0, 'obj':0})

        def load_pc_img(self, log_info: dict):
                # TODO: load correct image according to current step
                img_path = (os.path.sep).join( [pc_dir, self.algorithm.name.lower(), self.algorithm.name.lower() + '.PNG'] )
                img = mpimg.imread(img_path)
                self.img_ax.set_aspect('equal', anchor='E')
                self.img_ax.imshow(img)



class MISPDraw(Draw):

    def __init__(self, prob: Problem, alg: Algorithm, instance):
        super().__init__(prob,alg,instance)

    def init_graph(self, instance):
        graph = instance.graph
        #TODO find better layout
        nodelist = graph.nodes()
        pos = nx.kamada_kawai_layout(graph)
        nx.set_node_attributes(graph, 'lightgray', name='color')
        nx.set_node_attributes(graph, pos, 'pos')
        nx.set_edge_attributes(graph, 'lightgray', 'color')
        return graph

    def get_gvns_animation(self, i:int, log_data: list):
        self.reset_graph()
        comments = {'ch':'random','sh':'start from incumbent', 'li':'improve solution'}

        if i == 0:
                self.plot_description.update({'phase': f'MAX-Independent Set Instance',
                                                'comment': [f'{len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges']})
                log_data[i]['best'] = log_data[i]['obj'] = 0
                self.draw_graph()
                self.add_description(log_data[i])
                return

        info = log_data[i]

        compare_i = i + (info.get('status') == 'start') - (info.get('status') == 'end')
        compare_sol = set(log_data[compare_i].get('sol'))
        current_sol = set(info.get('sol'))

        #set color of sol solution
        nx.set_node_attributes(self.graph, {n:'green' for n in info.get('sol')}, 'color')

        remove = current_sol - compare_sol if info.get('status') == 'start' else compare_sol - current_sol
        add = current_sol - compare_sol if info.get('status') == 'end' else compare_sol - current_sol
        if info.get('status') == 'start':
                self.plot_description['comment'].append(comments[info['m']])
                self.plot_description['comment'].append(f'k={info.get("par")}')
                #set labels for elements to be removed/added
                nx.set_node_attributes(self.graph, {n:'+' for n in add}, 'label')
                nx.set_node_attributes(self.graph, {n:'-' for n in remove}, 'label')
                self.plot_description['comment'].append(f'remove {len(remove)} node(s), add {len(add)} node(s)')
        if info.get('status') == 'end' and not (add or remove) and info.get('m') == 'li':
                self.plot_description['comment'].append('no improvement - reached local optimum')


        if info.get('better',False):
                self.plot_description['comment'].append('found new incumbent')
                nx.set_node_attributes(self.graph, {n:'gold' for n in info.get('inc')}, 'color')

        if info.get('status') == 'end':
                self.plot_description['comment'].append(f'objective gain: {info["obj"] - log_data[i-1]["obj"]}')

        self.draw_graph(info['inc'])
        self.add_description(info)

    def get_grasp_animation(self, i:int, log_data: list):
        self.reset_graph()
        info = log_data[i] 

        if info['status'] == 'start':
                self.plot_description.update({'comment':['start with empty solution'], 'best':info.get('best')})
                self.draw_graph()
                self.add_description(log_data[i])
                return
        if info['status'] == 'end':
                self.plot_description.update({'comment':['created complete solution'], 'best':info.get('best'), 'obj':info.get('obj')})
                if info.get('better',False):
                        self.plot_description['comment'].append('found new incumbent')
                        nx.set_node_attributes(self.graph, {n:'gold' for n in info.get('sol')}, name='color')
                else:
                        nx.set_node_attributes(self.graph, {n:'green' for n in info.get('sol')}, name='color')
                self.draw_graph(info.get('inc'))
                self.add_description(log_data[i])
                return
        #set labels according to candidate list
        nx.set_node_attributes(self.graph, info.get('cl',{}), 'label')

        # set color of selected nodes
        nx.set_node_attributes(self.graph, {n: 'green' for n in info.get('sol', [])}, name='color')
        if info.get('status') == 'sel':
                self.graph.nodes[info.get('sel')]['color'] = 'gold'
                n_unsel = set(self.graph.neighbors(info.get('sel'))).intersection(set(info['cl'].keys()))
                nx.set_node_attributes(self.graph, {n:'darksalmon' for n in n_unsel},'color')
                nx.set_edge_attributes(self.graph, {(info.get('sel'),n):'black' for n in n_unsel}, 'color')


        j = i
        while not (log_data[j]['status'] in ['start','end']):
                j -= 1

        par = info.get('par',0.)
        mn = min(info.get('cl').values())
        mx = max(info.get('cl').values())
        comments = {'cl': 'remaining degree (number of unblocked neighbors)',
                        'rcl':  f'{par}-best' if type(par)==int else f'alpha: {par}, threshold: {round(mn + par * (mx-mn),2)}',
                        'sel': f'random, objective gain: 1'}

        self.plot_description['comment'].append(comments.get(info['status']))
        self.plot_description.update({'best': log_data[j]['best'], 'obj':len(info.get('sol'))})
        self.draw_graph(info.get('rcl',[]), sel_color='black')
        self.add_description(log_data[i])

    def reset_graph(self):
        nx.set_node_attributes(self.graph, 'lightgray', name='color')
        nx.set_edge_attributes(self.graph, 'lightgray', name='color')
        nx.set_node_attributes(self.graph, '', name='label')

    def draw_graph(self, pos_change: list() = [], sel_color='gold'):
        self.ax.clear()
        self.ax.set_ylim(bottom=-1.1,top=1.2)
        self.ax.set_xlim(left=-1.1,right=1.1)

        nodelist = self.graph.nodes()

        pos = nx.get_node_attributes(self.graph,'pos')

        color = [self.graph.nodes[n]['color'] for n in nodelist]
        linewidth = [3 if n in pos_change else 0 for n in nodelist]
        lcol = [sel_color for n in nodelist]
        #labels = {n:f'{n}{l}' for n,l in nx.get_node_attributes(graph, 'label').items()}
        labels = nx.get_node_attributes(self.graph, 'label')
        edges = list(self.graph.edges())
        e_cols = [self.graph.edges[e]['color'] for e in edges]
        
        nx.draw_networkx(self.graph, pos, nodelist=nodelist, with_labels=True, labels=labels,font_weight='heavy', font_size=14, ax=self.ax,  
                        node_color=color, edgecolors=lcol, edgelist=edges, edge_color=e_cols, linewidths=linewidth, node_size=500)






class MAXSATDraw(Draw):

    def __init__(self, prob: Problem, alg: Algorithm, instance):
        super().__init__(prob,alg,instance)

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
        pos = {v:[-1 + i*step, 0.4] for i,v in enumerate(variables, start=1)}
        pos.update({i:[-1 +j*step,0.6] for j,i in enumerate(incumbent, start=1)})
        step = 2/(m+1)
        pos.update({c:[-1+ i*step,-0.4] for i,c in enumerate(clauses_sorted, start=1)})

        # create nodes with data
        v = [(x, {'type':'variable', 'nr':x-m, 'color':'lightgray', 'pos':pos[x], 'label':f'x{x-m}','usage':clause}) for x,clause in enumerate(instance.variable_usage, start=m+1)]  #[m+1,...,m+n]
        c = [(x, {'type':'clause', 'nr':x, 'color': 'lightgray', 'pos':pos[x], 'label':f'c{x}', 'clause':clause}) for x,clause in enumerate(instance.clauses, start=1)]   #[1,..,m]
        i = [(x, {'type':'incumbent', 'nr':x-m-n, 'color':'white', 'pos':pos[x], 'label':f'i{x-m-n}'}) for x in incumbent]               #[1+m+n,...,2n+m]

        # create graph by adding nodes and edges
        graph = nx.Graph()
        graph.add_nodes_from(c)
        graph.add_nodes_from(v)
        graph.add_nodes_from(i)

        for i,cl in enumerate(instance.clauses, start=1):
                graph.add_edges_from([(i,abs(x)+ m,{'style':'dashed' if x < 0 else 'solid', 'color':'lightgray'}) for x in cl])

        return graph

    def get_gvns_animation(self, i:int, log_data: list):
        
        self.reset_graph()

        incumbent = [i for i,t in self.graph.nodes(data='type') if t=='incumbent']
        variables = [i for i,t in self.graph.nodes(data='type') if t=='variable']
        clauses = [i for i,t in self.graph.nodes(data='type') if t=='clause']

        if i == 0:
                self.plot_description['phase'] = f'MAX-SAT Instance'
                self.plot_description['comment'] = [f'{len(variables)} variables, {len(clauses)} clauses']
                self.draw_graph([])
                self.add_description(log_data[i])
                return
        
        info = log_data[i]
        if i == 1:
                log_data[0]['sol'] = [-1 for _ in info['sol']]
                self.plot_description['comment'].append('random') # TODO make generic to be able to used with other methods than random
        flipped_nodes = []

        nx.set_node_attributes(self.graph, {k: 'r' if info['inc'][self.graph.nodes[k]['nr']-1] == 0 else 'b' for k in incumbent}, name='color')
        nx.set_node_attributes(self.graph, {k: 'r' if info['sol'][self.graph.nodes[k]['nr']-1] == 0 else 'b' for k in variables}, name='color')
        
        added, removed, pos_literals = self.color_and_get_changed_clauses(i, log_data, info['status'] == 'start')

        if i > 1:
                comp = i + (info['status'] == 'start') - (info['status'] == 'end')
                flipped_nodes = [i+1 for i in range(len(variables)) if info['sol'][i]!=log_data[comp]['sol'][i] ]
                flipped_nodes = [n for n in variables if self.graph.nodes[n]['nr'] in flipped_nodes]

                if info['status'] == 'start':
                        self.plot_description['comment'].append(f'k={info.get("par")}')
                        self.plot_description['comment'].append(f'flipping {len(flipped_nodes)} variable(s)')          
                else:
                        
                        nx.set_edge_attributes(self.graph, {edge: 'black' for edge in self.graph.edges() if set(edge) & set(flipped_nodes)}, 'color')
                        prev = log_data[i-1]['obj'] if i > 1 else 0
                        self.plot_description['comment'].append(f"objective gain: {info['obj'] - prev}")
                        if len(flipped_nodes) == 0 and info['m'] == 'li':
                                self.plot_description['comment'].append('no improvement - reached local optimum')
                        flipped_nodes = []


        if info.get('better', False):
                flipped_nodes += [n for n in incumbent]
                self.plot_description['comment'].append('found new incumbent')

        self.draw_graph(flipped_nodes + list(added.union(removed)))
        self.write_literal_info(pos_literals)
        self.add_description(info)

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
                                    self.graph.nodes[n]['color'] = 'green'
                                    break
                          #  else:
                           #         self.graph.nodes[n]['color'] = 'lightgray'

            if literals:
                    num_literals = dict.fromkeys(clauses,0)
                    for fc in fulfilled:
                            for v in clauses[fc]:
                                if log_data['sol'][abs(v)-1] == (1 if v > 0 else 0):
                                        num_literals[fc] += 1
            
            return fulfilled,num_literals

    def write_literal_info(self, literal_info: dict):

        literal_info.update( {n:(v,[self.graph.nodes[n]['pos'][0],self.graph.nodes[n]['pos'][1]-0.1]) for n,v in literal_info.items()})

        for n,data in literal_info.items():
                self.ax.text(data[1][0],data[1][1],data[0],{'color': 'black', 'ha': 'center', 'va': 'center', 'fontsize':'small'})

    def get_grasp_animation(self, i:int, log_data: list):
        self.reset_graph()

        incumbent = [i for i,t in self.graph.nodes(data='type') if t=='incumbent']
        variables = [i for i,t in self.graph.nodes(data='type') if t=='variable']
        clauses = [i for i,t in self.graph.nodes(data='type') if t=='clause']
        info = log_data[i]

        if log_data[i]['status'] == 'start':
                self.plot_description['comment'].append('start with empty solution')
                info['obj'] = 0
                self.draw_graph([])
                self.write_literal_info(dict.fromkeys(clauses,0))
                self.add_description(info)
                return

        if log_data[i]['status'] == 'end':
                self.plot_description['comment'].append(f'constructed complete solution' + (', found new incumbent' if info['better'] else ''))
                nx.set_node_attributes(self.graph, {k: 'r' if info['inc'][self.graph.nodes[k]['nr']-1] == 0 else 'b' for k in incumbent}, name='color')
                nx.set_node_attributes(self.graph, {k: 'r' if info['sol'][self.graph.nodes[k]['nr']-1] == 0 else 'b' for k in variables}, name='color')
                _,_,pos_literals = self.color_and_get_changed_clauses(i,log_data)
                self.draw_graph(incumbent if info['better'] else [])
                self.write_literal_info(pos_literals)
                self.add_description(info)
                return


        #map variable ids to node ids
        keys = {v:k for k,v in nx.get_node_attributes(self.graph,'nr').items() if self.graph.nodes[k]['type'] == 'variable'}
        rcl = [np.sign(v)*keys[abs(v)] for v in info.get('rcl',[])]
        cl = {np.sign(k)*keys[abs(k)]:v for k,v in info['cl'].items()}
        sel = keys.get(abs(info.get('sel',0)),0) * np.sign(info.get('sel',0))

        not_sel = set(abs(v) for v in cl.keys())
        selected = set(variables).difference(not_sel)
        selected.add(abs(sel))

        #set colors for edges and variables
        nx.set_edge_attributes(self.graph, {edge: 'black' for edge in self.graph.edges() if abs(sel) in edge}, 'color')
        nx.set_node_attributes(self.graph, {n: 'r' if info['sol'][self.graph.nodes[n]['nr']-1] == 0 else 'b' for n in selected if n != 0}, name='color')
        #set colors for clauses
        added,_,pos_literals = self.color_and_get_changed_clauses(i,log_data,start=not (info['status'] == 'sel'))

        j = i
        while not log_data[j]['status'] in ['start','end']:
                j = j-1


        mx = max(info['cl'].values())
        par = info.get('par', 0)
        comments = {'cl': 'number of additionally fulfilled clauses',
                        'rcl':  f'{par}-best' if type(par)==int else f'alpha: {par}, threshold: {round(mx*par,2)}',
                        'sel': f'random, objective gain: {len(added)}'}

        self.plot_description.update({
                'best': log_data[j]['best'], 
                'obj': sum(p > 0 for p in pos_literals.values())
                })
        self.plot_description['comment'].append(comments.get(info['status']))
        # draw graph and print textual information
        self.draw_graph(([abs(sel)] if sel != 0 else []) + list(added))
        self.write_literal_info(pos_literals)
        self.write_cl_info(cl, rcl, sel)
        self.add_description(info)

    def write_cl_info(self, cl: dict(), rcl: list(), sel: int):

        cl_positions = {n:pos for n,pos in nx.get_node_attributes(self.graph, 'pos').items() if self.graph.nodes[n]['type'] == 'variable'}

        col = {1:'b',-1:'r',0:'lightgray'}

        for k,v in cl.items():
                pos = cl_positions[abs(k)]
                c = col[np.sign(k)] if len(rcl)==0 or k in rcl else col[0]
                bbox = dict(boxstyle="circle",fc="white", ec=c, pad=0.2) if k == sel else None
                self.ax.text(pos[0],pos[1]+0.2+(0.05*np.sign(k)), v, {'color': c, 'ha': 'center', 'va': 'center','fontweight':'bold','bbox': bbox})

    def reset_graph(self):
        nx.set_node_attributes(self.graph, {n: 'lightgray' if self.graph.nodes[n]['type'] != 'incumbent' else 'white' for n in self.graph.nodes()}, name='color')
        nx.set_edge_attributes(self.graph, 'lightgray', name='color')

    def draw_graph(self, pos_change):
        self.ax.clear()
        self.ax.set_ylim(bottom=-1,top=1.2)
        self.ax.set_xlim(left=-1,right=1)

        var_inc_nodes = [n for n,t in nx.get_node_attributes(self.graph, 'type').items() if  t in ['variable', 'incumbent']]
        var_inc_color = [self.graph.nodes[n]['color'] for n in var_inc_nodes]
        var_inc_lcol = ['black' if self.graph.nodes[n]['type'] == 'variable' else 'gold' for n in var_inc_nodes if n in pos_change]
        var_inc_lw = [4 if n in pos_change else 0 for n in var_inc_nodes]

        cl_nodes = [n for n,t in nx.get_node_attributes(self.graph, 'type').items() if  t =='clause']
        cl_color = [self.graph.nodes[n]['color'] for n in cl_nodes]
        cl_lcol = ['gold' for n in cl_nodes]
        cl_lw = [3 if n in pos_change else 0 for n in cl_nodes]

        edges = list(self.graph.edges())
        # draw gray and black edges seperately to avoid overpainting black edges
        e_list_gray = [e for i,e in enumerate(edges) if self.graph.edges[e]['color'] !='black']
        e_style_gray = [self.graph.edges[e]['style'] for e in e_list_gray]

        e_list_black = [e for i,e in enumerate(edges) if self.graph.edges[e]['color'] =='black']
        e_style_black = [self.graph.edges[e]['style'] for e in e_list_black]

        var_labels = {v:self.graph.nodes[v]['label'] for v in self.graph.nodes() if self.graph.nodes[v]['type'] == 'variable'}

        pos = nx.get_node_attributes(self.graph, 'pos')

        nx.draw_networkx_nodes(self.graph, pos, nodelist=var_inc_nodes, node_color=var_inc_color, edgecolors=var_inc_lcol, linewidths=var_inc_lw, node_shape='s', node_size=500,ax=self.ax)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=cl_nodes, node_color=cl_color, edgecolors=cl_lcol, linewidths=cl_lw, node_shape='o',node_size=150,ax=self.ax)
        nx.draw_networkx_edges(self.graph, pos, edgelist=e_list_gray, style=e_style_gray,ax=self.ax, edge_color='lightgray')
        nx.draw_networkx_edges(self.graph, pos, edgelist=e_list_black, style=e_style_black,ax=self.ax, edge_color='black')
        nx.draw_networkx_labels(self.graph, pos, labels=var_labels,ax=self.ax)





def get_visualisation(prob: Problem, alg: Algorithm, instance):
    prob_class = globals()[prob.name + 'Draw']
    prob = prob_class(prob, alg, instance)
    return prob



# only used for debugging
if __name__ == '__main__':
    log_data, instance = read_from_logfile('maxsat_gvns_20201229_205432.log')
    draw = get_visualisation(Problem.MAXSAT, Algorithm.GVNS, instance)
    for i in range(0, 100):
        draw.get_animation(i,log_data)
    