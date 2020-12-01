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



def get_visualisation(prob: Problem, alg: Algorithm, instance):

    if prob == Problem.MAXSAT and (alg == Algorithm.GVNS or alg == Algorithm.GRASP):
        return init_gvns_maxsat(instance)


def get_animation(iter: int, log_data: list(), graph):

        if log_data[0] == 'MAXSAT' and log_data[1] == 'GVNS':
                get_gvns_maxsat_animation(iter, log_data[2:], graph)
        if log_data[0] == 'MAXSAT' and log_data[1] == 'GRASP':
                get_grasp_maxsat_animation(iter, log_data[2:], graph)



def get_gvns_maxsat_animation(value: int, log_data: list(), graph):
        if log_data[0]['m'].startswith('ch'):
                log_data[0]['current'], log_data[0]['inc'], log_data[0]['obj'], log_data[0]['best']  = None, None, 0, 0

        incumbent = [i for i,data in list(graph.nodes(data='label')) if data.startswith('i')]
        variables = [i for i,data in list(graph.nodes(data='label')) if data.startswith('x')]
        clauses = [i for i,data in list(graph.nodes(data='label')) if data.startswith('c')]
        n = len(variables)
        
        
        titles = {'ch':'Construction', 'li':'Local improvment', 'sh':'Shaking'}
        methods = {'ch': 'random', 'li':'k-flip neigborhood search', 'sh': 'k-random-flip'}
        info = log_data[value]
        if info['m'].startswith('rc'):
                return
        plt.clf()
        title = 'Solving MaxSat with GNVS'
        
        m = info['m'][0:2]
        size = info['m'][2:]
        if m == 'ch' and info['status'] == 'end':
                title = f'{titles[m]}: {methods[m]}'
        if not m == 'ch':
                title =  f'{titles[m]}: {methods[m]}, k={size}'
                        
        best=info['best']
        text = 'incumbent'
        if info['status'] == 'end' and info['better']:
                text = text + ' new'
        plt.text(-1.2,0.25,text,fontsize=10)
        #edge_col = ['black' if v else 'r' for v in info['inc']] if info['inc'] else v_colors

        #linewidth = 4

        node_col_v = ['b' if v else 'r' for v in info['current']] if info['current'] else ['gray']*n
        node_col_i = ['b' if v else 'r' for v in info['inc']] if info['inc'] else ['gray']*n
        node_col_c = ['gray' for i in clauses]
        edge_col =None
        linewidth = 1
        #if info['status'] == 'start' and not info['m'].startswith('ch'):
                #edge_col = [node_col_v[i] if info['current'][i]== log_data[value+1]['current'][i] else 'gray' for i in range(n)]
                #linewidth = 4
        plt.suptitle(title)
        pos = {n:pos for n,pos in graph.nodes(data='pos')}
        if info['status'] == 'start' and not info['m'].startswith('ch'):
                diff = [i + len(clauses)+1 for i in range(n) if info['current'][i]!=log_data[value + 1]['current'][i]]
                edge = ['black' for i in diff]
                linew = 7
                nx.draw_networkx_nodes(graph,pos,nodelist=diff,edgecolors=edge,linewidths=linew, node_shape='s', node_size=500)
                
        if info['current']:
                #c = [(i,clause) for data for graph.nodes(data='clause')]
                c = nx.get_node_attributes(graph,'clause')   
                for i,clause in c.items():
                        for v in clause:
                                if info['current'][abs(v)-1] == (1 if v > 0 else 0):
                                        node_col_c[i-1] = 'green'
                                        break
                                else:
                                        node_col_c[i-1] = 'gray'
        plt.text(-1.2,-0.105,f'objective: {info["obj"]}',fontsize=10)
        #plt.text(-1.2,-0.13,f'best: {best}',fontsize=10)
        #plt.text(-1.2,0.25,'incumbent:',fontsize=10)
        plt.text(-1.2,0.23,f'objective: {best}',fontsize=10)
        labels = {n:data for n,data in graph.nodes(data='label') if n in variables}

        edges = list(graph.edges())
        e_style = [dict(graph[u][v])['style'] for u,v in edges]  

        nx.draw_networkx_nodes(graph,pos, nodelist=incumbent,node_color=node_col_i, node_shape='s', node_size=500)
        nx.draw_networkx_nodes(graph, pos, nodelist=variables, node_color=node_col_v, edgecolors=edge_col, linewidths=linewidth, node_shape='s', node_size=500)
        nx.draw_networkx_nodes(graph, pos, nodelist=clauses, node_color=node_col_c, node_shape='o',node_size=70)
        nx.draw_networkx_edges(graph,pos, style=e_style)
        nx.draw_networkx_labels(graph,pos,labels=labels)

        for i in ['right','top','bottom','left']:
            plt.gca().spines[i].set_visible(False)


def get_grasp_maxsat_animation(iter: int, log_data: list(), graph):


        if log_data[iter]['status'] in ['start', 'end']:
                get_gvns_maxsat_animation(iter, log_data, graph)
        else:
                get_grasp_maxsat_rcl_animation(iter, log_data, graph)


def get_grasp_maxsat_rcl_animation(iter: int, log_data: list(), graph):

        incumbent = [i for i,data in list(graph.nodes(data='label')) if data.startswith('i')]
        variables = [i for i,data in list(graph.nodes(data='label')) if data.startswith('x')]
        clauses = [i for i,data in list(graph.nodes(data='label')) if data.startswith('c')]
        n = len(variables)
        
        
        titles = {'ch':'Construction', 'li':'Local improvment', 'cl':'Candidate List', 'rc':'Restricted Candidate List'}
        methods = {'rc': '', 'li':'k-flip neigborhood search', 'cl': 'greedy'}
        plt.clf()
        title = 'Solving MaxSat with GRASP'
        info = log_data[iter]
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
        col_selected = ['r' if info['x'][v-len(clauses)-1] == 0 else 'b' for v in selected] #are drawn in current value, but with alpha/transparent



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



def init_gvns_maxsat(instance: MAXSATInstance):

        plt.axis('off')
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
        pos = {n:[-1 + i*step, 0.2] for i,n in enumerate(variables, start=1)}
        pos.update({i:[-1 +j*step,0.25] for j,i in enumerate(incumbent, start=1)})
        step = 2/(m+1)
        pos.update({m:[-1+ i*step,-0.1] for i,m in enumerate(clauses_sorted, start=1)})

        # create nodes with data
        v = [(x, {'color':'gray', 'pos':pos[x], 'label':f'x{x-m}','usage':clause}) for x,clause in enumerate(instance.variable_usage, start=m+1)]  #[m+1,...,m+n]
        c = [(x, {'color': 'gray', 'pos':pos[x], 'label':f'c{x}', 'clause':clause}) for x,clause in enumerate(instance.clauses, start=1)]   #[1,..,m]
        i = [(x, {'color':'gray', 'pos':pos[x], 'label':f'i{x-m-n}'}) for x in incumbent]               #[1+m+n,...,2n+m]

        # create graph by adding nodes and edges
        graph = nx.Graph()
        graph.add_nodes_from(c)
        graph.add_nodes_from(v)
        graph.add_nodes_from(i)

        for i,li in enumerate(instance.clauses, start=1):
                graph.add_edges_from([(i,abs(x)+ m,{'style':'dashed' if x < 0 else 'solid'}) for x in li])

        return graph






