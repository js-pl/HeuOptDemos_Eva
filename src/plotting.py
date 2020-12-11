import sys
sys.path.append('..\\HeuOptDemos_Eva')

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
plt.rcParams['figure.figsize'] = (12,5)
plt.rcParams['figure.dpi'] = 80
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False
f = None
ax = None

plot_description = {'phase': '',
                        'comment': [],
                        'best': 0,
                        'obj': 0,
                        }

phases = {'ch':'Construction', 'li': 'Local Search', 'sh': 'Shaking', 'rgc': 'Randomized Greedy Construction', 
                'cl':'Candidate List', 'rcl': 'Restricted Candidate List', 'sel':'Selection from RCL'}

def get_visualisation(prob: Problem, alg: Algorithm, instance):
        global f
        global ax

        if not f:
                f = plt.figure(num = f'Solving {prob.value} with {alg.value}')
                ax = f.add_subplot(111)
        else:
                f = plt.gcf()
                ax = f.gca()

        #f.set_size_inches(7,5) # resets the window title..?


        f.canvas.set_window_title(f'Solving {prob.value} with {alg.value}')

        if prob == Problem.MAXSAT and (alg == Algorithm.GVNS or alg == Algorithm.GRASP):
                return init_maxsat_graph(instance)

        if prob == Problem.MISP and (alg == Algorithm.GVNS or alg == Algorithm.GRASP):

                return init_misp_graph(instance)



def get_animation(i: int, log_data: list(), graph):

        if log_data[0] == Problem.MAXSAT.name.lower() and log_data[1] == Algorithm.GVNS.name.lower():
                get_gvns_maxsat_animation(i, log_data[2:], graph)
        if log_data[0] == Problem.MAXSAT.name.lower() and log_data[1] == Algorithm.GRASP.name.lower():
                get_grasp_maxsat_animation(i, log_data[2:], graph)
        if log_data[0] == Problem.MISP.name.lower() and log_data[1] == Algorithm.GVNS.name.lower():
                get_gvns_misp_animation(i, log_data[2:], graph)
        if log_data[0] == Problem.MISP.name.lower() and log_data[1] == Algorithm.GRASP.name.lower():
                get_grasp_misp_animation(i, log_data[2:], graph)
                



def add_description(log_info: dict()):

        if log_info.get('status') == 'start' and log_info.get('m').startswith('ch'):
                log_info['best'] = 0
                log_info['obj'] = 0
        else:
                plot_description['phase'] = phases.get(log_info['status'],'') + phases.get(log_info.get('m','   '),'')


        ax.text(0,1, '\n'.join((
        '%s: %s' % (plot_description['phase'], ', '.join(plot_description['comment']) ),
        'Best Objective: %d' % (log_info.get('best',plot_description['best']), ),
        'Current Objective: %d' % (log_info.get('obj',plot_description['obj']),))), horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

        #reset description
        plot_description.update({'phase': '', 'comment': [], 'best':0, 'obj':0})



def get_gvns_maxsat_animation(value: int, log_data: list(), graph):

        reset_maxsat_graph(graph)

        incumbent = [i for i,t in graph.nodes(data='type') if t=='incumbent']
        variables = [i for i,t in graph.nodes(data='type') if t=='variable']
        clauses = [i for i,t in graph.nodes(data='type') if t=='clause']

        if value == 0:
                plot_description['phase'] = f'MAX-SAT Instance'
                plot_description['comment'] = [f'{len(variables)} variables, {len(clauses)} clauses']
                draw_maxsat_graph(graph, [])
                add_description(log_data[value])
                return
        
        info = log_data[value]
        if value == 1:
                log_data[0]['sol'] = [-1 for _ in info['sol']]
                plot_description['comment'].append('random') # TODO make generic to be able to used with other methods than random
        flipped_nodes = []

        nx.set_node_attributes(graph, {k: 'r' if info['inc'][graph.nodes[k]['nr']-1] == 0 else 'b' for k in incumbent}, name='color')
        nx.set_node_attributes(graph, {k: 'r' if info['sol'][graph.nodes[k]['nr']-1] == 0 else 'b' for k in variables}, name='color')
        
        added, removed, pos_literals = color_and_get_changed_clauses(value, log_data, graph, info['status'] == 'start')

        if value > 1:
                comp = value + (info['status'] == 'start') - (info['status'] == 'end')
                flipped_nodes = [i+1 for i in range(len(variables)) if info['sol'][i]!=log_data[comp]['sol'][i] ]
                flipped_nodes = [n for n in variables if graph.nodes[n]['nr'] in flipped_nodes]

                if info['status'] == 'start':
                        plot_description['comment'].append(f'k={info.get("par")}')
                        plot_description['comment'].append(f'flipping {len(flipped_nodes)} variable(s)')          
                else:
                        
                        nx.set_edge_attributes(graph, {edge: 'black' for edge in graph.edges() if set(edge) & set(flipped_nodes)}, 'color')
                        prev = log_data[value-1]['obj'] if value > 1 else 0
                        plot_description['comment'].append(f"objective gain: {info['obj'] - prev}")
                        if len(flipped_nodes) == 0 and info['m'] == 'li':
                                plot_description['comment'].append('no improvement - reached local optimum')
                        flipped_nodes = []


        if info.get('better', False):
                flipped_nodes += [n for n in incumbent]
                plot_description['comment'].append('found new incumbent')

        draw_maxsat_graph(graph, flipped_nodes + list(added.union(removed)))
        write_literal_info(pos_literals,graph)
        add_description(info)

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
                        if log_data['sol'][abs(v)-1] == (1 if v > 0 else 0):
                                fulfilled.add(n)
                                graph.nodes[n]['color'] = 'green'
                                break
                        else:
                                graph.nodes[n]['color'] = 'lightgray'

        if literals:
                num_literals = dict.fromkeys(clauses,0)
                for fc in fulfilled:
                        for v in clauses[fc]:
                             if log_data['sol'][abs(v)-1] == (1 if v > 0 else 0):
                                     num_literals[fc] += 1
        
        return fulfilled,num_literals



def get_grasp_maxsat_animation(i: int, log_data: list(), graph):

        if log_data[i].get('m','').startswith('rgc') or log_data[i].get('status','') not in ['start','end']:
                
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
                plot_description['comment'].append('start with empty solution')
                info['obj'] = 0
                draw_maxsat_graph(graph, [])
                write_literal_info(dict.fromkeys(clauses,0),graph)
                add_description(info)
                return

        if log_data[i]['status'] == 'end':
                plot_description['comment'].append(f'constructed complete solution' + (', found new incumbent' if info['better'] else ''))
                nx.set_node_attributes(graph, {k: 'r' if info['inc'][graph.nodes[k]['nr']-1] == 0 else 'b' for k in incumbent}, name='color')
                nx.set_node_attributes(graph, {k: 'r' if info['sol'][graph.nodes[k]['nr']-1] == 0 else 'b' for k in variables}, name='color')
                _,_,pos_literals = color_and_get_changed_clauses(i,log_data, graph)
                draw_maxsat_graph(graph, incumbent if info['better'] else [])
                write_literal_info(pos_literals,graph)
                add_description(info)
                return


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
        nx.set_node_attributes(graph, {n: 'r' if info['sol'][graph.nodes[n]['nr']-1] == 0 else 'b' for n in selected if n != 0}, name='color')
        #set colors for clauses
        added,_,pos_literals = color_and_get_changed_clauses(i,log_data,graph,start=not (info['status'] == 'sel'))

        j = i
        while log_data[j]['status'] != 'start':
                j = j-1

        mx = max(info['cl'].values())
        par = info.get('par', 0)
        comments = {'cl': 'number of additionally fulfilled clauses',
                        'rcl':  f'{par}-best' if type(par)==int else f'alpha: {par}, threshold: {round(mx*par,2)}',
                        'sel': f'random, objective gain: {len(added)}'}

        plot_description.update({
                'best': log_data[j]['best'], 
                'obj': sum(p > 0 for p in pos_literals.values())
                })
        plot_description['comment'].append(comments.get(info['status']))
        # draw graph and print textual information
        draw_maxsat_graph(graph,([abs(sel)] if sel != 0 else []) + list(added))
        write_literal_info(pos_literals,graph)
        write_cl_info(cl, rcl, sel, graph)
        add_description(info)



def write_literal_info(literal_info: dict(),graph):

        literal_info.update( {n:(v,[graph.nodes[n]['pos'][0],graph.nodes[n]['pos'][1]-0.1]) for n,v in literal_info.items()})

        for n,data in literal_info.items():
                ax.text(data[1][0],data[1][1],data[0],{'color': 'black', 'ha': 'center', 'va': 'center', 'fontsize':'small'})


def write_cl_info(cl: dict(), rcl: list(), sel: int, graph):

        cl_positions = {n:pos for n,pos in nx.get_node_attributes(graph, 'pos').items() if graph.nodes[n]['type'] == 'variable'}

        col = {1:'b',-1:'r',0:'lightgray'}

        for k,v in cl.items():
                pos = cl_positions[abs(k)]
                c = col[np.sign(k)] if len(rcl)==0 or k in rcl else col[0]
                bbox = dict(boxstyle="circle",fc="white", ec=c, pad=0.2) if k == sel else None
                ax.text(pos[0],pos[1]+0.2+(0.05*np.sign(k)), v, {'color': c, 'ha': 'center', 'va': 'center','fontweight':'bold','bbox': bbox})


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


def draw_maxsat_graph(graph, pos_change):

        ax.clear()
        plt.axis('off')
        ax.set_ylim(bottom=-1,top=1.2)
        ax.set_xlim(left=-1,right=1)

        for i in ['right','top','bottom','left']:
            plt.gca().spines[i].set_visible(False)

        var_inc_nodes = [n for n,t in nx.get_node_attributes(graph, 'type').items() if  t in ['variable', 'incumbent']]
        var_inc_color = [graph.nodes[n]['color'] for n in var_inc_nodes]
        var_inc_lcol = ['black' if graph.nodes[n]['type'] == 'variable' else 'gold' for n in var_inc_nodes if n in pos_change]
        var_inc_lw = [4 if n in pos_change else 0 for n in var_inc_nodes]

        cl_nodes = [n for n,t in nx.get_node_attributes(graph, 'type').items() if  t =='clause']
        cl_color = [graph.nodes[n]['color'] for n in cl_nodes]
        cl_lcol = ['gold' for n in cl_nodes]
        cl_lw = [3 if n in pos_change else 0 for n in cl_nodes]

        edges = list(graph.edges())
        # draw gray and black edges seperately to avoid overpainting black edges
        e_list_gray = [e for i,e in enumerate(edges) if graph.edges[e]['color'] !='black']
        e_style_gray = [graph.edges[e]['style'] for e in e_list_gray]

        e_list_black = [e for i,e in enumerate(edges) if graph.edges[e]['color'] =='black']
        e_style_black = [graph.edges[e]['style'] for e in e_list_black]

        var_labels = {v:graph.nodes[v]['label'] for v in graph.nodes() if graph.nodes[v]['type'] == 'variable'}

        pos = nx.get_node_attributes(graph, 'pos')

        nx.draw_networkx_nodes(graph, pos, nodelist=var_inc_nodes, node_color=var_inc_color, edgecolors=var_inc_lcol, linewidths=var_inc_lw, node_shape='s', node_size=500,ax=ax)
        nx.draw_networkx_nodes(graph, pos, nodelist=cl_nodes, node_color=cl_color, edgecolors=cl_lcol, linewidths=cl_lw, node_shape='o',node_size=150,ax=ax)
        nx.draw_networkx_edges(graph,pos, edgelist=e_list_gray, style=e_style_gray,ax=ax, edge_color='lightgray')
        nx.draw_networkx_edges(graph,pos, edgelist=e_list_black, style=e_style_black,ax=ax, edge_color='black')
        nx.draw_networkx_labels(graph,pos,labels=var_labels,ax=ax)


def reset_maxsat_graph(graph):
        
        nx.set_node_attributes(graph, {n: 'lightgray' if graph.nodes[n]['type'] != 'incumbent' else 'white' for n in graph.nodes()}, name='color')
        nx.set_edge_attributes(graph, 'lightgray', name='color')



def init_misp_graph(instance: MISPInstance):

        graph = instance.graph
        #TODO find better layout
        nodelist = graph.nodes()
        #pos = nx.kamada_kawai_layout(graph, dist={n: {nn: 0.1 for nn in nodelist if nn != n} for n in nodelist})
        pos = nx.kamada_kawai_layout(graph)
        nx.set_node_attributes(graph, 'lightgray', name='color')
        nx.set_node_attributes(graph, pos, 'pos')
        nx.set_edge_attributes(graph, 'lightgray', 'color')

        return graph



def draw_misp_graph(graph, pos_change: list() = [], sel_color='gold'):
        ax.clear()
        plt.axis('off')
        ax.set_ylim(bottom=-1.1,top=1.2)
        ax.set_xlim(left=-1.1,right=1.1) #TODO maybe adapt these to positions of outer nodes

        nodelist = graph.nodes()

        pos = nx.get_node_attributes(graph,'pos')

        color = [graph.nodes[n]['color'] for n in nodelist]
        linewidth = [3 if n in pos_change else 0 for n in nodelist]
        lcol = [sel_color for n in nodelist]
        #labels = {n:f'{n}{l}' for n,l in nx.get_node_attributes(graph, 'label').items()}
        labels = nx.get_node_attributes(graph, 'label')
        
        nx.draw_networkx(graph, pos, nodelist=nodelist, with_labels=True, labels=labels,font_weight='heavy', font_size=14, ax=ax,  
                        node_color=color, edgecolors=lcol, edge_color='lightgray', linewidths=linewidth, node_size=500)



def get_gvns_misp_animation(i: int, log_data: list(), graph):
        reset_misp_graph(graph)
        comments = {'ch':'random','sh':'start from incumbent', 'li':'improve solution'}

        if i == 0:
                plot_description.update({'phase': f'MAX-Independent Set Instance',
                                                'comment': [f'{len(graph.nodes())} nodes, {len(graph.edges())} edges']})
                log_data[i]['best'] = log_data[i]['obj'] = 0
                draw_misp_graph(graph, [])
                add_description(log_data[i])
                return

        info = log_data[i]

        compare_i = i + (info.get('status') == 'start') - (info.get('status') == 'end')
        compare_sol = set(log_data[compare_i].get('sol'))
        current_sol = set(info.get('sol'))

        #set color of sol solution
        nx.set_node_attributes(graph, {n:'green' for n in info.get('sol')}, 'color')

        remove = current_sol - compare_sol if info.get('status') == 'start' else compare_sol - current_sol
        add = current_sol - compare_sol if info.get('status') == 'end' else compare_sol - current_sol
        if info.get('status') == 'start':
                plot_description['comment'].append(comments[info['m']])
                plot_description['comment'].append(f'k={info.get("par")}')
                #set labels for elements to be removed/added
                nx.set_node_attributes(graph, {n:'+' for n in add}, 'label')
                nx.set_node_attributes(graph, {n:'-' for n in remove}, 'label')
                plot_description['comment'].append(f'remove {len(remove)} node(s), add {len(add)} node(s)')
        if info.get('status') == 'end' and not (add or remove) and info.get('m') == 'li':
                plot_description['comment'].append('no improvement - reached local optimum')


        if info.get('better',False):
                plot_description['comment'].append('found new incumbent')
                nx.set_node_attributes(graph, {n:'gold' for n in info.get('inc')}, 'color')

        if info.get('status') == 'end':
                plot_description['comment'].append(f'objective gain: {info["obj"] - log_data[i-1]["obj"]}')

        draw_misp_graph(graph, info['inc'])
        add_description(info)


def get_grasp_misp_animation(i: int, log_data:list(), graph):

        if log_data[i].get('m','').startswith('rgc') or log_data[i].get('status','') not in ['start','end']:
                
                get_grasp_misp_rcl_animation(i, log_data, graph)
        else:
                get_gvns_misp_animation(i, log_data, graph)



def get_grasp_misp_rcl_animation(i:int, log_data: list(), graph):
        reset_misp_graph(graph)
        info = log_data[i] 
        # TODO add vis for start and end
        if info['status'] == 'start':
                plot_description.update({'comment':['start with empty solution'], 'best':info.get('best')})
                draw_misp_graph(graph)
                add_description(log_data[i])
                return
        if info['status'] == 'end':
                plot_description.update({'comment':['created complete solution'], 'best':info.get('best'), 'obj':info.get('obj')})
                if info.get('better',False):
                        plot_description['comment'].append('found new incumbent')
                        nx.set_node_attributes(graph, {n:'gold' for n in info.get('sol')}, name='color')
                else:
                        nx.set_node_attributes(graph, {n:'green' for n in info.get('sol')}, name='color')
                draw_misp_graph(graph,info.get('inc'))
                add_description(log_data[i])
                return
        #set labels according to candidate list
        nx.set_node_attributes(graph, info.get('cl',{}), 'label')

        # set color of selected nodes
        nx.set_node_attributes(graph, {n: 'green' for n in info.get('sol', [])}, name='color')
        if info.get('status') == 'sel':
                graph.nodes[info.get('sel')]['color'] = 'gold'

        j = i
        while not (log_data[j]['status'] == 'start'):
                j -= 1

        par = info.get('par',0.)
        mn = min(info.get('cl').values())
        comments = {'cl': 'remaining degree (number of unblocked neighbors)',
                        'rcl':  f'{par}-best' if type(par)==int else f'alpha: {par}, threshold: {round(mn*par,2)}',
                        'sel': f'random, objective gain: 1'}

        plot_description['comment'].append(comments.get(info['status']))
        plot_description.update({'best': log_data[j]['best'], 'obj':len(info.get('sol'))})
        draw_misp_graph(graph, info.get('rcl',[]), sel_color='black')
        add_description(log_data[i])


def reset_misp_graph(graph):
        nx.set_node_attributes(graph, 'lightgray', name='color')
        nx.set_edge_attributes(graph, 'lightgray', name='color')
        nx.set_node_attributes(graph, '', name='label')



#only used for debugging
if __name__ == '__main__':
        graph = init_misp_graph(MISPInstance('gnm-30-60'))
        get_animation(0,['MISP','GVNS'], graph)


