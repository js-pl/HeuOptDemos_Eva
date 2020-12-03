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
plt.rcParams['figure.figsize'] = (10,4)
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.facecolor'] = 'w'
f = None
ax = None

plot_description = {'phase': 'Starting',
                        'comment': 'press "play" to start',
                        'best': 0,
                        'current': 0,
                        }

phases = {'ch':'Construction', 'li': 'Local Search', 'sh': 'Shaking', 'rc': 'Randomized Greedy Construction'}

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
                return init_maxsat_graph(instance)


def get_animation(i: int, log_data: list(), graph):

        if log_data[0] == 'MAXSAT' and log_data[1] == 'GVNS':
                get_gvns_maxsat_animation(i, log_data[2:], graph)
        if log_data[0] == 'MAXSAT' and log_data[1] == 'GRASP':
                get_grasp_maxsat_animation(i, log_data[2:], graph)
                



def add_description():

        ax.text(0,1, '\n'.join((
        '%s%s' % (plot_description['phase'], '' if plot_description['comment'] == '' else ', ' + plot_description['comment'] ),
        'Best Objective: %d' % (plot_description['best'], ),
        'Current Objective: %d' % (plot_description['current'],))), horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)



def get_gvns_maxsat_animation(value: int, log_data: list(), graph):

        incumbent = [i for i,t in graph.nodes(data='type') if t=='incumbent']
        variables = [i for i,t in graph.nodes(data='type') if t=='variable']
        clauses = [i for i,t in graph.nodes(data='type') if t=='clause']

        if value == 0:
                plot_description['phase'] = f'MAX-SAT Instance: {len(variables)} variables, {len(clauses)} clauses'
                plot_description['comment'] = f'press "play" to start'
                reset_maxsat_graph(graph) 
                draw_graph(graph, [])
                return

        info = log_data[value]
        added_clauses = []
        flipped_nodes = []
        better = []
        comment = ''
        phase = phases[info['m'][:2]]


        nx.set_node_attributes(graph, {k: 'r' if info['inc'][graph.nodes[k]['nr']-1] == 0 else 'b' for k in incumbent}, name='color')
        nx.set_node_attributes(graph, {k: 'r' if info['current'][graph.nodes[k]['nr']-1] == 0 else 'b' for k in variables}, name='color')
        
        # find newly fulfilled clauses
        #TODO actually find new clauses and objective gain (maybe also show clauses that become negative due to the change)
        for n in clauses:
                for v in graph.nodes[n]['clause']:
                        if info['current'][abs(v)-1] == (1 if v > 0 else 0):
                                if info['status'] == 'end' and log_data[value-1]['current'][abs(v)-1] != (1 if v > 0 else 0):
                                        # the clause was not fulfilled previously
                                        #TODO find clauses after construction
                                        added_clauses.append(n)
                                
                                graph.nodes[n]['color'] = 'green'
                                break
                        else:
                                graph.nodes[n]['color'] = 'lightgray'


        if info['status'] == 'start' and not info['m'].startswith('ch') :
                flipped_nodes = [i+1 for i in range(len(variables)) if info['current'][i]!=log_data[value + 1]['current'][i] ]
                flipped_nodes = [n for n in variables if graph.nodes[n]['nr'] in flipped_nodes]
                comment = f'flipping {len(flipped_nodes)} variable(s)' if len(flipped_nodes)  > 0 or info['m'].startswith('sh') else 'no improvement found'

        if info.get('better', False):
                better = [n for n in incumbent]
                comment += 'found new incumbent'

        # fill description with text
        plot_description['phase'] = phase
        plot_description['comment'] = comment
        plot_description['best'] = info.get('best', 0)
        plot_description['current'] = info.get('obj', 0)

        draw_graph(graph, flipped_nodes + added_clauses + better)





def draw_graph(graph, pos_change):

        ax.clear()
        plt.axis('off')
        add_description()
        ax.set_ylim(bottom=-0.13,top=0.37)
        ax.set_xlim(left=-1,right=1)

        for i in ['right','top','bottom','left']:
            plt.gca().spines[i].set_visible(False)

        var_inc_nodes = [n for n,t in nx.get_node_attributes(graph, 'type').items() if  t !='clause']
        var_inc_color = [graph.nodes[n]['color'] for n in var_inc_nodes]
        var_inc_lcol = ['black' if graph.nodes[n]['type'] == 'variable' else 'yellow' for n in var_inc_nodes if n in pos_change]
        var_inc_lw = [4 if n in pos_change else 0 for n in var_inc_nodes]

        cl_nodes = [n for n,t in nx.get_node_attributes(graph, 'type').items() if  t =='clause']
        cl_color = [graph.nodes[n]['color'] for n in cl_nodes]
        cl_lcol = ['yellow' for n in cl_nodes]
        cl_lw = [3 if n in pos_change else 0 for n in cl_nodes]

        edges = list(graph.edges())
        e_style = [graph.edges[e]['style'] for e in edges]

        var_labels = {v:graph.nodes[v]['label'] for v in graph.nodes() if graph.nodes[v]['type'] == 'variable'} # nicht unbedingt notwendig?

        pos = nx.get_node_attributes(graph, 'pos')

        nx.draw_networkx_nodes(graph, pos, nodelist=var_inc_nodes, node_color=var_inc_color, edgecolors=var_inc_lcol, linewidths=var_inc_lw, node_shape='s', node_size=500,ax=ax)
        nx.draw_networkx_nodes(graph, pos, nodelist=cl_nodes, node_color=cl_color, edgecolors=cl_lcol, linewidths=cl_lw, node_shape='o',node_size=100,ax=ax)
        nx.draw_networkx_edges(graph,pos, edgelist=edges, style=e_style,ax=ax)
        nx.draw_networkx_labels(graph,pos,labels=var_labels,ax=ax)




def reset_maxsat_graph(graph):
        
        nx.set_node_attributes(graph, {n: 'lightgray' if graph.nodes[n]['type'] != 'incumbent' else 'white' for n in graph.nodes()}, name='color')


def get_grasp_maxsat_animation(i: int, log_data: list(), graph):

        if log_data[i].get('m','').startswith('rc') or log_data[i].get('status','') not in ['start','end']:
                get_grasp_maxsat_rcl_animation(i, log_data, graph)
        else:
                get_gvns_maxsat_animation(i, log_data, graph)

               


def get_grasp_maxsat_rcl_animation(i: int, log_data: list(), graph):
        #plt.clf()
        plt.axis('off')
        if log_data[i]['status'] in ['start', 'end']:
                #TODO something to start and end the greedy construction
                task = 'todo'
                #plt.suptitle('Starting Greedy Randomized Construction')
                reset_maxsat_graph(graph)
                draw_graph(graph, [])
                return


        # how to:
        # reset graph and only change the color of relevant nodes


        incumbent = [i for i,t in graph.nodes(data='type') if t=='incumbent']
        variables = [i for i,t in graph.nodes(data='type') if t=='variable']
        clauses = [i for i,t in graph.nodes(data='type') if t=='clause']
        n = len(variables)
        
        
        titles = {'ch':'Construction', 'li':'Local improvment', 'cl':'Candidate List', 'rc':'Restricted Candidate List'}
        methods = {'rc': '', 'li':'k-flip neigborhood search', 'cl': 'greedy'}
        #plt.clf()
        title = 'Solving MaxSat with GRASP'
        info = log_data[i]
        edges = list(graph.edges())
        e_style = [dict(graph[u][v])['style'] for u,v in edges] 
        pos = {n:pos for n,pos in graph.nodes(data='pos')}
        labels = {n:data for n,data in graph.nodes(data='label') if n in variables}

        not_sel = list(set(abs(k) for k,_ in info['cl'].items()))
        col_not_sel = ['gray']*len(not_sel)
        if info['status'] == 'sel':
                col_not_sel[not_sel.index(abs(info['sel']))] = 'r' if info['sel'] < 0 else 'b'
                #TODO color fulfilled clauses
        not_sel = [v + len(clauses) for v in not_sel]  

        selected = [v for v in variables if v not in not_sel]
        col_selected = ['r' if info['x'][v-len(clauses)-1] == 0 else 'b' for v in selected]


        nx.draw_networkx_nodes(graph, pos, nodelist=not_sel, node_color=col_not_sel, node_shape='s', node_size=500)
        nx.draw_networkx_nodes(graph, pos, nodelist=selected, node_color=col_selected, node_shape='s',node_size=500)
        nx.draw_networkx_nodes(graph,pos, nodelist=incumbent, node_color='white', alpha=1, node_size = 500, node_shape='s')
        nx.draw_networkx_nodes(graph, pos, nodelist=clauses, node_color='gray', node_shape='o',node_size=70)
        nx.draw_networkx_edges(graph,pos, style=e_style)
        nx.draw_networkx_labels(graph,pos,labels=labels)
        plt.suptitle('Candidate List' if info['status']=='cl' else 'Restricted Candidate List: '+ str(info.get('k','')) + str(info.get('alpha','')) )

        pos_inc = {n-len(variables):pos for n,pos in graph.nodes(data='pos') if n in incumbent and n-len(variables) in not_sel}
        for n,pos in pos_inc.items():
                text_pos = str(info['cl'][n-len(clauses)])
                text_neg = str(info['cl'][-(n-len(clauses))])
                if info['status'] == 'sel' and abs(info['sel']) + len(clauses) == n:
                        if info['sel'] < 0 :
                                plt.text(pos[0],pos[1]-0.02,text_neg,{'color': 'r', 'ha': 'center', 'va': 'center','fontweight':'bold','bbox': dict(boxstyle="round",fc="white", ec="r", pad=0.2)})
                                plt.text(pos[0],pos[1],text_pos,{'color': 'gray', 'ha': 'center', 'va': 'center','fontweight':'bold'})
                        else:
                                plt.text(pos[0],pos[1],text_pos,{'color': 'b', 'ha': 'center', 'va': 'center','fontweight':'bold','bbox': dict(boxstyle="round",fc="white", ec="b", pad=0.2)})
                                plt.text(pos[0],pos[1]-0.02,text_neg,{'color': 'gray', 'ha': 'center', 'va': 'center','fontweight':'bold'})
                if (info['status'] == 'rcl' or info['status'] == 'sel') and -(n-len(clauses)) in info['rcl']:
                                plt.text(pos[0],pos[1]-0.02,text_neg,{'color': 'r', 'ha': 'center', 'va': 'center','fontweight':'bold'})
                                plt.text(pos[0],pos[1],text_pos,{'color': 'gray', 'ha': 'center', 'va': 'center','fontweight':'bold'})
                                
                elif (info['status'] == 'rcl' or info['status'] == 'sel') and n-len(clauses) in info['rcl']:
                                plt.text(pos[0],pos[1],text_pos,{'color': 'b', 'ha': 'center', 'va': 'center','fontweight':'bold'})
                                plt.text(pos[0],pos[1]-0.02,text_neg,{'color': 'gray', 'ha': 'center', 'va': 'center','fontweight':'bold'})
                elif info['status'] == 'rcl':
                        plt.text(pos[0],pos[1],text_pos, {'color': 'gray', 'ha': 'center', 'va': 'center','fontweight':'bold'})

                        plt.text(pos[0],pos[1]-0.02,text_neg,{'color': 'gray', 'ha': 'center', 'va': 'center','fontweight':'bold'})
                elif info['status']=='cl':
                        plt.text(pos[0],pos[1],text_pos, {'color': 'b', 'ha': 'center', 'va': 'center','fontweight':'bold'})

                        plt.text(pos[0],pos[1]-0.02,text_neg,{'color': 'r', 'ha': 'center', 'va': 'center','fontweight':'bold'})
                else:
                        plt.text(pos[0],pos[1],text_pos, {'color': 'gray', 'ha': 'center', 'va': 'center','fontweight':'bold'})

                        plt.text(pos[0],pos[1]-0.02,text_neg,{'color': 'gray', 'ha': 'center', 'va': 'center','fontweight':'bold'})


        for i in ['right','top','bottom','left']:
                plt.gca().spines[i].set_visible(False)

        plt.axis('off')



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
                graph.add_edges_from([(i,abs(x)+ m,{'style':'dashed' if x < 0 else 'solid'}) for x in cl])


        return graph






