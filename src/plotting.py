import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import statistics        
from IPython.display import display, display_html 
import ipywidgets as widgets
from pymhlib.solution import Solution
from pymhlib.demos.maxsat import MAXSATInstance, MAXSATSolution
from pymhlib.demos.misp import MISPInstance, MISPSolution
from src.handler import Problem, Algorithm

# plot settings
plt.rcParams['figure.figsize'] = (12,4)
plt.rcParams['figure.dpi'] = 80
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.facecolor'] = 'w'
f = None
ax = None

plot_description = {'phase': 'Starting',
                        'comment': 'press "play" to start',
                        'best': 0,
                        'current': 0,
                        }

phases = {'ch':'Construction', 'li': 'Local Search', 'sh': 'Shaking', 'rc': 'Randomized Greedy Construction', 'cl':'Candidate List', 'rcl': 'Restricted Candidate List', 'sel':'Selection from RCL'}

def get_visualisation(prob: Problem, alg: Algorithm, instance):
        global f
        global ax
        if not f:
                f = plt.figure(num = f'Solving {prob.value} with {alg.value}')
                ax = f.add_subplot(111)
        else:
                f = plt.gcf()
                ax = f.gca()
                f.canvas.set_window_title(f'Solving {prob.value} with {alg.value}')

        if prob == Problem.MAXSAT and (alg == Algorithm.GVNS or alg == Algorithm.GRASP):
                #if alg == Algorithm.GRASP:
                 #       f.set_size_inches(5,5, forward=True)
                return init_maxsat_graph(instance)



def get_animation(i: int, log_data: list(), graph):

        if log_data[0] == 'MAXSAT' and log_data[1] == 'GVNS':
                get_gvns_maxsat_animation(i, log_data[2:], graph)
        if log_data[0] == 'MAXSAT' and log_data[1] == 'GRASP':
                get_grasp_maxsat_animation(i, log_data[2:], graph)
                



def add_description():

        ax.text(0,1, '\n'.join((
        '%s%s' % (plot_description['phase'], '' if plot_description['comment'] == '' else ': ' + plot_description['comment'] ),
        'Best Objective: %d' % (plot_description['best'], ),
        'Current Objective: %d' % (plot_description['current'],))), horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)



def get_gvns_maxsat_animation(value: int, log_data: list(), graph):

        reset_maxsat_graph(graph)

        incumbent = [i for i,t in graph.nodes(data='type') if t=='incumbent']
        variables = [i for i,t in graph.nodes(data='type') if t=='variable']
        clauses = [i for i,t in graph.nodes(data='type') if t=='clause']

        if value == 0:
                plot_description['phase'] = f'MAX-SAT Instance: {len(variables)} variables, {len(clauses)} clauses'
                plot_description['comment'] = f'press "play" to start'
                draw_maxsat_graph(graph, [])
                return
        

        info = log_data[value]
        if value == 1:
                log_data[0]['current'] = [-1 for _ in info['current']]
        flipped_nodes = []
        comment = 'random' # TODO make generic to be able to used with other methods than random
        phase = phases['ch'] if value == 1 else phases[info['m'][:2]]


        nx.set_node_attributes(graph, {k: 'r' if info['inc'][graph.nodes[k]['nr']-1] == 0 else 'b' for k in incumbent}, name='color')
        nx.set_node_attributes(graph, {k: 'r' if info['current'][graph.nodes[k]['nr']-1] == 0 else 'b' for k in variables}, name='color')
        
        added, removed, pos_literals = color_and_get_changed_clauses(value, log_data, graph, info['status'] == 'start')

        if value > 1:
                comp = 0
                if info['status'] == 'start':
                        comp = value + 1
                else:
                        comp = value -1
                flipped_nodes = [i+1 for i in range(len(variables)) if info['current'][i]!=log_data[comp]['current'][i] ]
                flipped_nodes = [n for n in variables if graph.nodes[n]['nr'] in flipped_nodes]

                if info['status'] == 'start':
                        comment = f'flipping {len(flipped_nodes)} variable(s)'          
                else:
                        prev = log_data[value-1]['obj'] if value > 1 else 0
                        comment = f"objective gain: {info['obj'] - prev}"
                        nx.set_edge_attributes(graph, {edge: 'black' for edge in graph.edges() if set(edge) & set(flipped_nodes)}, 'color')
                        comment += ', no improvement - reached local optimum' if len(flipped_nodes) == 0 else ''
                        flipped_nodes = []


        if info.get('better', False):
                flipped_nodes += [n for n in incumbent]
                comment += ', found new incumbent'

        # fill description with text
        plot_description['phase'] = phase
        plot_description['comment'] = comment
        plot_description['best'] = info.get('best', 0)
        plot_description['current'] = info.get('obj', 0)

        draw_maxsat_graph(graph, flipped_nodes + list(added.union(removed)))
        write_literal_info(pos_literals,graph)

def color_and_get_changed_clauses(value,log_data, graph, start=True):

        i = value - (not start)
        pos_start,literals = color_clauses_and_count_literals(log_data[i], graph, literals=start)

        if start:
                return set(),set(),literals

        pos_end,literals = color_clauses_and_count_literals(log_data[value],graph)

        return pos_end-pos_start, pos_start-pos_end,literals


def color_clauses_and_count_literals(log_data: dict(), graph, literals=True):

        clauses = nx.get_node_attributes(graph, 'clause')
        fulfilled = set()
        num_literals = dict()

        for n,clause in clauses.items():
                for v in clause:
                        if log_data['current'][abs(v)-1] == (1 if v > 0 else 0):
                                fulfilled.add(n)
                                graph.nodes[n]['color'] = 'green'
                                break
                        else:
                                graph.nodes[n]['color'] = 'lightgray'

        if literals:
                num_literals = dict.fromkeys(clauses,0)
                for fc in fulfilled:
                        for v in clauses[fc]:
                             if log_data['current'][abs(v)-1] == (1 if v > 0 else 0):
                                     num_literals[fc] += 1
        
        return fulfilled,num_literals



def draw_maxsat_graph(graph, pos_change):

        ax.clear()
        plt.axis('off')
        add_description()
        ax.set_ylim(bottom=-0.14,top=0.37)
        ax.set_xlim(left=-1,right=1)

        for i in ['right','top','bottom','left']:
            plt.gca().spines[i].set_visible(False)

        var_inc_nodes = [n for n,t in nx.get_node_attributes(graph, 'type').items() if  t in ['variable', 'incumbent']]
        var_inc_color = [graph.nodes[n]['color'] for n in var_inc_nodes]
        var_inc_lcol = ['black' if graph.nodes[n]['type'] == 'variable' else 'yellow' for n in var_inc_nodes if n in pos_change]
        var_inc_lw = [4 if n in pos_change else 0 for n in var_inc_nodes]

        cl_nodes = [n for n,t in nx.get_node_attributes(graph, 'type').items() if  t =='clause']
        cl_color = [graph.nodes[n]['color'] for n in cl_nodes]
        cl_lcol = ['yellow' for n in cl_nodes]
        cl_lw = [3 if n in pos_change else 0 for n in cl_nodes]

        edges = list(graph.edges())
        # draw gray and black edges seperately to avoid overpainting black edges
        e_list_gray = [e for i,e in enumerate(edges) if graph.edges[e]['color'] !='black']
        e_style_gray = [graph.edges[e]['style'] for e in e_list_gray]

        e_list_black = [e for i,e in enumerate(edges) if graph.edges[e]['color'] =='black']
        e_style_black = [graph.edges[e]['style'] for e in e_list_black]

        var_labels = {v:graph.nodes[v]['label'] for v in graph.nodes() if graph.nodes[v]['type'] == 'variable'} # nicht unbedingt notwendig?

        pos = nx.get_node_attributes(graph, 'pos')

        nx.draw_networkx_nodes(graph, pos, nodelist=var_inc_nodes, node_color=var_inc_color, edgecolors=var_inc_lcol, linewidths=var_inc_lw, node_shape='s', node_size=500,ax=ax)
        nx.draw_networkx_nodes(graph, pos, nodelist=cl_nodes, node_color=cl_color, edgecolors=cl_lcol, linewidths=cl_lw, node_shape='o',node_size=150,ax=ax)
        nx.draw_networkx_edges(graph,pos, edgelist=e_list_gray, style=e_style_gray,ax=ax, edge_color='lightgray')
        nx.draw_networkx_edges(graph,pos, edgelist=e_list_black, style=e_style_black,ax=ax, edge_color='black')
        nx.draw_networkx_labels(graph,pos,labels=var_labels,ax=ax)




def reset_maxsat_graph(graph):
        
        nx.set_node_attributes(graph, {n: 'lightgray' if graph.nodes[n]['type'] != 'incumbent' else 'white' for n in graph.nodes()}, name='color')
        nx.set_edge_attributes(graph, 'lightgray', name='color')


def get_grasp_maxsat_animation(i: int, log_data: list(), graph):

        if log_data[i].get('m','').startswith('rc') or log_data[i].get('status','') not in ['start','end']:
                
                get_grasp_maxsat_rcl_animation(i, log_data, graph)
        else:
                get_gvns_maxsat_animation(i, log_data, graph)

               


def get_grasp_maxsat_rcl_animation(i: int, log_data: list(), graph):
        reset_maxsat_graph(graph)

        incumbent = [i for i,t in graph.nodes(data='type') if t=='incumbent']
        variables = [i for i,t in graph.nodes(data='type') if t=='variable']
        clauses = [i for i,t in graph.nodes(data='type') if t=='clause']
        info = log_data[i]

        if log_data[i]['status'] == 'start':
                plot_description.update({'phase': phases['rc'],
                        'comment': 'start with empty solution',
                        'best': log_data[i]['best'],
                        'current': 0,
                        })
                draw_maxsat_graph(graph, [])
                write_literal_info(dict.fromkeys(clauses,0),graph)
                return

        if log_data[i]['status'] == 'end':
                plot_description.update({'phase': phases['rc'],
                'comment': f'constructed complete solution' + (', found new incumbent' if info['better'] else ''),
                'best': info['best'],
                'current': info['obj']
                })
                nx.set_node_attributes(graph, {k: 'r' if info['inc'][graph.nodes[k]['nr']-1] == 0 else 'b' for k in incumbent}, name='color')
                nx.set_node_attributes(graph, {k: 'r' if info['current'][graph.nodes[k]['nr']-1] == 0 else 'b' for k in variables}, name='color')
                _,_,pos_literals = color_and_get_changed_clauses(i,log_data, graph)
                draw_maxsat_graph(graph, incumbent if info['better'] else [])
                write_literal_info(pos_literals,graph)
                return


        mx = max(info['cl'].values())
        comments = {'cl': 'number of additionally fulfilled clauses',
                        'rcl':  ''.join([str(info.get('k', 'alpha: ')) + str(info.get('alpha',' best'))] + [f', threshold: {round(info.get("alpha",1) * mx,2)}' if info.get('alpha') else '']),
                        'sel': 'random'}

        #map variable ids to node ids
        keys = {v:k for k,v in nx.get_node_attributes(graph,'nr').items() if graph.nodes[k]['type'] == 'variable'}
        rcl = [np.sign(v)*keys[abs(v)] for v in info.get('rcl',[])]
        cl = {np.sign(k)*keys[abs(k)]:v for k,v in info['cl'].items()}
        sel = keys.get(abs(info.get('sel',0)),0) * np.sign(info.get('sel',0))

        not_sel = set(abs(v) for v in cl.keys())
        selected = set(variables).difference(not_sel)
        selected.add(abs(sel))

        #set colors for edges and variables
        nx.set_edge_attributes(graph, {edge: 'black' for edge in graph.edges() if abs(sel) in edge}, 'color')
        nx.set_node_attributes(graph, {n: 'r' if info['current'][graph.nodes[n]['nr']-1] == 0 else 'b' for n in selected if n != 0}, name='color')
        #set colors for clauses
        added,_,pos_literals = color_and_get_changed_clauses(i,log_data,graph,start=not (info['status'] == 'sel'))

        j = i
        while log_data[j]['status'] != 'start':
                j = j-1

        plot_description.update({'phase': phases[info['status']],
                'comment': comments.get(info['status']) + (f', objective gain: {len(added)}' if info['status'] == 'sel' else ''),
                'best': log_data[j]['best'], 
                'current': sum(p > 0 for p in pos_literals.values())
                })

        # draw graph and print textual information
        draw_maxsat_graph(graph,([abs(sel)] if sel != 0 else []) + list(added))
        write_literal_info(pos_literals,graph)
        write_cl_info(cl, rcl, sel, graph)



def write_literal_info(literal_info: dict(),graph):

        literal_info.update( {n:(v,[graph.nodes[n]['pos'][0],graph.nodes[n]['pos'][1]-0.025]) for n,v in literal_info.items()})

        for n,data in literal_info.items():
                ax.text(data[1][0],data[1][1],data[0],{'color': 'black', 'ha': 'center', 'va': 'center', 'fontsize':'small'})


def write_cl_info(cl: dict(), rcl: list(), sel: int, graph):

        cl_positions = {n:pos for n,pos in nx.get_node_attributes(graph, 'pos').items() if graph.nodes[n]['type'] == 'variable'}

        col = {1:'b',-1:'r',0:'lightgray'}

        for k,v in cl.items():
                pos = cl_positions[abs(k)]
                c = col[np.sign(k)] if len(rcl)==0 or k in rcl else col[0]
                bbox = dict(boxstyle="circle",fc="white", ec=c, pad=0.2) if k == sel else None
                ax.text(pos[0],pos[1]+0.06+(0.015*np.sign(k)), v, {'color': c, 'ha': 'center', 'va': 'center','fontweight':'bold','bbox': bbox})


def init_maxsat_graph(instance: MAXSATInstance):

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
        pos = {v:[-1 + i*step, 0.2] for i,v in enumerate(variables, start=1)}
        pos.update({i:[-1 +j*step,0.25] for j,i in enumerate(incumbent, start=1)})
        step = 2/(m+1)
        pos.update({c:[-1+ i*step,-0.1] for i,c in enumerate(clauses_sorted, start=1)})

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






