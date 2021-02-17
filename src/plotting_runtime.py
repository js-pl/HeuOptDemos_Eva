from .problems import Problem
from .logdata import RunData
import matplotlib.pyplot as plt
from matplotlib import gridspec as gs
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from IPython.display import display
import pandas as pd

class PlotRuntime():
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.right'] = True
    plt.rcParams['axes.spines.top'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['figure.figsize'] = (12,7)
    plt.rcParams['figure.autolayout'] = True
    problem: Problem = None
    g = gs.GridSpec(3,2)
    ax = None
    ax_bp = None
    ax_sum = None

    def plot(self, i:int, lines, configurations: dict, iter_data, sum_data: dict):

        fig = plt.figure(num=f'{self.problem.value}',clear=True)

        self.ax = fig.add_subplot(self.g[0:2,:])
        self.ax_bp = fig.add_subplot(self.g[2,0])
        self.ax_sum = fig.add_subplot(self.g[2,1])
        
        sol_data = self.plot_obj(i, lines, configurations, iter_data)
        self.plot_bp(i, sol_data)
        self.plot_sum(sum_data)
        fig.suptitle('')

    def plot_obj(self, iter: int, lines: list, configurations, iter_data):
        legend_handles=[]
        sol = 'best_sol' if 'best_sol' in lines else 'current_sol'
        if sol == 'best_sol':
            iter_data = iter_data.cummax(axis=0) if self.problem in [Problem.MAXSAT,Problem.MISP] \
                                                else iter_data.cummin(axis=0)

        for c in list(iter_data.columns.get_level_values(0).unique()):

                col = f'C{int(c.split(".")[0]) % 10}'
                df = iter_data[c]

                maxi = df.max(axis=1)
                mini = df.min(axis=1)
                if 'max' in lines:
                        maxi.plot(color=col, ax=self.ax, linestyle='dashed')
                if 'min' in lines:
                        mini.plot(color=col, ax=self.ax, linestyle='dashdot')
                if 'polygon' in lines:
                        self.ax.fill_between(df.index, maxi, mini, where=maxi > mini , facecolor=col, alpha=0.2, interpolate=True)
                if 'mean' in lines:
                        m = df.mean(axis=1)
                        m.plot(color=col, ax=self.ax, linestyle=(0, (3, 5, 1, 5, 1, 5)))
                if 'median' in lines:
                        m = df.median(axis=1)
                        m.plot(color=col, ax=self.ax,linestyle='dotted')

                seed = configurations.get(c).seed
                legend_handles += [Patch(color=col,label=c + f' (n={len(df.columns)},s={seed})')]
    
        loc = 'upper right'
        loc2 = 'lower left'
        best = None

        if self.problem in [Problem.MAXSAT,Problem.MISP]:
                best = iter_data.cummax(axis=0).cummax(axis=1).iloc[:,-1:]
                loc = 'lower right'
                loc2 = 'upper left'
        else:
                best = iter_data.cummin(axis=0).cummin(axis=1).iloc[:,-1:]
        if 'best' in lines:
                best.plot(color='black',ax=self.ax, linestyle='solid')
        ax2 = self.ax.twinx()
        ax2.get_yaxis().set_visible(False)
        self.ax.legend(handles=legend_handles,loc=loc)
        ax2.legend(handles=[Line2D([0],[0],color='black',label=k,linestyle=v) for k,v in 
                                        {'max':'dashed','min':'dashdot','mean':(0, (3, 5, 1, 5, 1, 5)),'median':'dotted','best':'solid'}.items() if k in lines],loc=loc2)

        self.ax.axvline(x=iter)
        self.ax.set_ylabel('objective value')
        self.ax.set_xlabel('iterations')
        return iter_data

    def plot_bp(self, iter, iter_data):
        bp_data = iter_data
        bp_data = bp_data.loc[iter].reset_index(level=1, drop=True).reset_index()
        bp_data = bp_data.rename(columns={iter:f'iteration={iter}'})
        bp = bp_data.boxplot(by='index',rot=25,patch_artist=True,ax=self.ax_bp,boxprops=dict(edgecolor='black'),medianprops=dict(color='black'),whiskerprops=dict(color='black'))
        self.ax_bp.set_xlabel('')
        self.ax_bp.set_ylabel('objective value')
        labels = bp_data.groupby(['index']).sum().index
        for i,name in enumerate(labels):
                col = f'C{int(name.split(".")[0]) % 10}'
                bp.findobj(Patch)[i].set_facecolor(col)
        plt.show()
        




    def plot_sum(self, summaries: dict):
        to_plot = []
        color = []
        n = ''
        for name,df in summaries.items():
                n = df.name
                color.append(f'C{int(name.split(".")[0]) % 10}')
                data = df.rename(name)
                data = pd.to_numeric(data, errors='coerce')
                data = data.groupby(by='method').mean()
                data.drop('SUM/AVG',inplace=True)
                to_plot.append(data)
        to_plot = pd.concat(to_plot,axis=1)
        to_plot.plot.bar(color=color,ax=self.ax_sum,legend=False)
        self.ax_sum.set_ylabel(n)
        self.ax_sum.set_xlabel('methods')

