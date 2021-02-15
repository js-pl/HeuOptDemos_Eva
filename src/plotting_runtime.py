from .problems import Problem
from .logdata import RunData
import matplotlib.pyplot as plt
from matplotlib import gridspec as gs
from matplotlib.lines import Line2D
from IPython.display import display

class PlotRuntime():
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.right'] = True
    plt.rcParams['axes.spines.top'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['figure.figsize'] = (12,7)
    plt.rcParams['figure.autolayout'] = True
    problem: Problem = None
    #fig = plt.figure(num=f'{self.problem.value}',clear=True)
    g = gs.GridSpec(3,2)
    ax = None
    ax_bb = None
    ax_sum = None

    def plot(self, i:int, lines, sum_options, iter_data, sum_data):

        fig = plt.figure(num=f'{self.problem.value}',clear=True)

        self.ax = fig.add_subplot(self.g[0:2,:])
        self.ax_bb = fig.add_subplot(self.g[2,0])
        self.ax_sum = fig.add_subplot(self.g[2,1])
        
        self.plot_obj(i, lines, iter_data)
        self.plot_bb(i, iter_data)
        self.plot_sum(sum_options, sum_data)
        fig.suptitle('')

    def plot_obj(self, iter: int, lines: list, iter_data):
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
                    maxi.plot(color=col, ax=self.ax)
            if 'min' in lines:
                    mini.plot(color=col, ax=self.ax)
            if 'polygon' in lines:
                    self.ax.fill_between(df.index, maxi, mini, where=maxi > mini , facecolor=col, alpha=0.2, interpolate=True)
            if 'mean' in lines:
                    m = df.mean(axis=1)
                    m.plot(color=col, ax=self.ax, linestyle='dotted')
            if 'median' in lines:
                    m = df.median(axis=1)
                    m.plot(color=col, ax=self.ax,linestyle='dashed')
            legend_handles += [Line2D([0],[0],color=col,label=c + f' (n={len(df.columns)},s=addSeed!)')]#{self.configurations[c].seed})')]
    
        loc = 'upper right'
        best = None

        if self.problem in [Problem.MAXSAT,Problem.MISP]:
                best = iter_data.cummax(axis=0).cummax(axis=1).iloc[:,-1:]
                loc = 'lower right'
        else:
                best = iter_data.cummin(axis=0).cummin(axis=1).iloc[:,-1:]
                loc = 'upper right'
        if 'best' in lines:
                best.plot(color='black',ax=self.ax)
        self.ax.legend(handles=legend_handles,loc=loc)

        self.ax.axvline(x=iter)

    def plot_bb(self, iter, iter_data):
        bb_data = iter_data
        bb_data = bb_data.loc[iter].reset_index(level=1, drop=True).reset_index()
        bb_data = bb_data.rename(columns={iter:f'iteration={iter}'})
        bb_data.boxplot(by='index',rot=25,ax=self.ax_bb)
        self.ax_bb.set_xlabel('')
        self.ax_bb.set_ylabel('objective value')




    def plot_sum(self, sum_options, summaries):
            pass