"""
module which builds all necessary widgets for visualisation and runtime analysis based on information received from handler/logdata module

"""
import sys
sys.path.append("C:/Users/Eva/Desktop/BakkArbeit/pymhlib")

from pymhlib.demos.maxsat import MAXSATInstance, MAXSATSolution
from pymhlib.demos.misp import MISPInstance, MISPSolution

import networkx as nx
import matplotlib.pyplot as plt
import statistics        
from IPython.display import display, display_html 
import ipywidgets as widgets
from . import handler
from .problems import Problem, Algorithm, Option, InitSolution, Configuration
from . import plotting as p
from .logdata import Log, LogData, RunData, save_visualisation, read_from_logfile, get_log_description
from IPython.display import clear_output
import os
from matplotlib.lines import Line2D
import pandas as pd
from matplotlib import gridspec as gs
import time




def load_visualisation_settings():

        interface = InterfaceVisualisation()
        interface.display_main_selection()

def load_runtime_settings():

        interface = InterfaceRuntimeAnalysis()
        interface.display_widgets()



class InterfaceVisualisation():

        def __init__(self,visualisation=True):

                self.problemWidget = widgets.Dropdown(
                                        options = handler.get_problems(),
                                        description = 'Problem'
                                        )
                self.algoWidget =  widgets.Dropdown(
                                options = handler.get_algorithms(Problem(self.problemWidget.value)),
                                description = 'Algorithm'
                                )
                
                self.instanceWidget = widgets.Dropdown(
                                options = handler.get_instances(Problem(self.problemWidget.value),visualisation),
                                description = 'Instance'
                                )
                self.instanceBox = widgets.HBox([self.instanceWidget])

                self.optionsWidget = widgets.VBox() #container which holds all options

                self.optionsHandles = {} #used to store references to relevant option widgets since they are wrapped in boxes

                self.log_data = None

                self.plot_instance = None
                self.out = widgets.Output()
                self.settingsWidget = widgets.Accordion(selected_index=None)

                def on_change_settings(change):
                        if change.owner.description == 'seed' and change.new <0:
                                change.owner.value = 0
                        if (change.owner.description == 'iterations' or change.owner.description == 'runs') and change.new <1:
                                change.owner.value = 1
                seed = widgets.IntText(description='seed', value=0)
                iterations = widgets.IntText(description='iterations', value=100)
                runs = widgets.IntText(description='runs',value=1,layout=widgets.Layout(display='None'))
                use_runs = widgets.Checkbox(description='use saved runs',value=False,layout=widgets.Layout(display='None'))
                seed.observe(on_change_settings,names='value')
                iterations.observe(on_change_settings,names='value')
                runs.observe(on_change_settings,names='value')
                self.settingsWidget.children = (widgets.VBox([iterations,seed,runs,use_runs]),)
                self.settingsWidget.set_title(0, 'General settings')
                self.controls = self.init_controls()

                self.controls.layout.visibility = 'hidden'
                self.run_button =  widgets.Button(description = 'Run')
                self.save_button = None
                self.mainSelection = widgets.RadioButtons(options=['load from log file', 'generate new run'])
                self.logfileWidget = widgets.Dropdown(layout=widgets.Layout(display='None'))


        def init_controls(self):
        
                play = widgets.Play(interval=1000, value=0, min=0, max=100,
                        step=1, description="Press play")
                slider = widgets.IntSlider(value=0, min=0, max=100,
                        step=1, orientation='horizontal',)

                def click_prev(event):
                        slider.value = slider.value - 1
                
                def click_next(event):
                        slider.value = slider.value + 1

                prev_iter = widgets.Button(description='',icon='step-backward',tooltip='previous', layout=widgets.Layout(width='50px'))
                next_iter = widgets.Button(description='',icon='step-forward', tooltip='next', layout=widgets.Layout(width='50px'))
                prev_iter.on_click(click_prev)
                next_iter.on_click(click_next)
                log_granularity = widgets.Dropdown(description='Log granularity', options=[l.value for l in Log])
                log_granularity.observe(self.on_change_log_granularity, names='value')
                widgets.jslink((play, 'value'), (slider, 'value'))
                slider.observe(self.animate, names = 'value')

                return widgets.HBox([play, slider, prev_iter, next_iter, log_granularity])

        def on_change_log_granularity(self, change):
                next_iter = self.log_data.change_granularity(self.controls.children[1].value, Log(self.controls.children[4].value))
                self.plot_instance.log_granularity = self.log_data.current_level
                #set max,min,value of slider and controls to appropriate iteration number
                self.controls.children[0].max = self.controls.children[1].max = len(self.log_data.log_data) - 1
                self.controls.children[1].value = next_iter
                self.animate(None)



        def animate(self,event):
                with self.out:
                        #p.get_animation(self.controls.children[1].value, self.log_data.log_data, self.plot_instance)
                        self.plot_instance.get_animation(self.controls.children[1].value, self.log_data.log_data)
                        widgets.interaction.show_inline_matplotlib_plots()
                
        def on_change_problem(self, change):

                self.algoWidget.options = handler.get_algorithms(Problem(change.new))
                self.optionsWidget.children = self.get_options(Algorithm(self.algoWidget.value))
                self.instanceWidget.options = handler.get_instances(Problem(change.new), not isinstance(self, InterfaceRuntimeAnalysis))
                self.optionsHandles = {}
                self.on_change_algo(None)


    
        def on_change_algo(self, change):

                self.optionsHandles = {} #reset references
                self.optionsWidget.children = self.get_options(Algorithm(self.algoWidget.value))


        def run_visualisation(self, event):
                params = None
                log_data = list()
                if self.mainSelection.value == 'load from log file':
                        log_data, instance = read_from_logfile(self.logfileWidget.value)
                        params = Configuration(Problem[log_data[0]].value, Algorithm[log_data[1]].value, '')
                        log_data = log_data[2:]
                        
                else:
                        params = self.prepare_parameters()
                        # starts call to pymhlib in handler module
                        log_data, instance = handler.run_algorithm_visualisation(params)

                self.log_data = LogData(log_data)

                # initialize graph from instance
                with self.out:
                        self.out.clear_output()
                        self.plot_instance = p.get_visualisation(params.problem,params.algorithm, instance, self.log_data.current_level)
                        widgets.interaction.show_inline_matplotlib_plots()

                self.controls.children[1].value = 0
                self.controls.children[0].max = self.controls.children[1].max = len(self.log_data.log_data) -1
                self.controls.children[4].value = Log.StepInter.value

                # start drawing
                self.animate(None)
                self.controls.layout.visibility = 'visible'
                self.save_button.disabled = self.mainSelection.value == 'load from log file' 

        def prepare_parameters(self):

                instance = '-'.join([str(c.value) for c in self.instanceBox.children])
                params = Configuration(self.problemWidget.value, self.algoWidget.value, instance)

                # extend if further options are needed
                if Option.CH in self.optionsHandles:
                        params.options[Option.CH] = [(self.optionsHandles.get(Option.CH).value, InitSolution[self.optionsHandles.get(Option.CH).value].value)]
                if Option.LI in self.optionsHandles:
                        #TODO: make sure name splitting works if no 'k=' given (ok for now because k is always added)
                        params.options[Option.LI] = [(name.split(',')[0], int(name.split('=')[1])) for name in list(self.optionsHandles.get(Option.LI).options)]
                if Option.SH in self.optionsHandles:
                        params.options[Option.SH] = [(name.split(',')[0], int(name.split('=')[1])) for name in list(self.optionsHandles.get(Option.SH).options)]
                if Option.RGC in self.optionsHandles:
                        params.options[Option.RGC] = [(self.optionsHandles[Option.RGC].children[0].value,self.optionsHandles[Option.RGC].children[1].value)]
                if Option.TL in self.optionsHandles:
                        params.options[Option.TL] = [(o.description,o.value) for o in self.optionsHandles[Option.TL].children[1:]]
                # add settings params
                params.iterations = self.settingsWidget.children[0].children[0].value
                params.seed = self.settingsWidget.children[0].children[1].value
                if params.instance.startswith('random') and params.seed > 0 :
                        params.instance += f'-{params.seed}'

                return params


        def display_main_selection(self):

                options_box = self.display_widgets()
                options_box.layout.display = 'None'
                log_description = widgets.Output()

                def on_change_logfile(change):
                        with log_description:
                                log_description.clear_output()
                                print(get_log_description(change['new']))

                self.logfileWidget.observe(on_change_logfile, names='value')

                def on_change_main(change):
                        if change['new'] == 'load from log file':
                                self.logfileWidget.options = os.listdir('logs' + os.path.sep + 'saved')
                                self.run_button.disabled = not len(self.logfileWidget.options) > 0
                                self.logfileWidget.layout.display = 'flex'
                                log_description.layout.display = 'flex'
                                options_box.layout.display = 'None'

                        else: 
                                self.run_button.disabled = False
                                self.logfileWidget.layout.display = 'None'
                                log_description.layout.display = 'None'
                                options_box.layout.display = 'flex'


                self.mainSelection.observe(on_change_main, names='value')
                self.run_button.on_click(self.run_visualisation)
                display(self.mainSelection)
                on_change_main({'new': 'load from log file'})
                display(widgets.VBox([self.logfileWidget,log_description]))
                display(options_box)
                display(widgets.VBox([self.run_button, self.controls, self.out]))


        def display_widgets(self):

                self.problemWidget.observe(self.on_change_problem, names='value')
                self.instanceWidget.observe(self.on_change_instance, names='value')
                self.algoWidget.observe(self.on_change_algo, names='value')
                self.optionsWidget.children = self.get_options(Algorithm(self.algoWidget.value))


                self.save_button = widgets.Button(description='Save Visualisation', disabled=True)
                self.save_button.on_click(self.on_click_save)
                optionsBox = widgets.VBox([self.settingsWidget, self.problemWidget,self.instanceBox,self.algoWidget,self.optionsWidget])
                return widgets.VBox([optionsBox, self.save_button])

        def on_click_save(self,event):
                #TODO make sure params were not changed!!!!
                save_visualisation(self.prepare_parameters(), self.plot_instance.graph)
                self.save_button.disabled = True


        def on_change_instance(self,change):
                if change.new == 'random':
                        n = widgets.IntText(value=30, description='n:',layout=widgets.Layout(width='150px'))
                        m = widgets.IntText(value=50, description='m:',layout=widgets.Layout(width='150px'))
                        self.instanceBox.children = (self.instanceWidget,n,m)
                else:
                        self.instanceBox.children = (self.instanceWidget,)

                
        def get_options(self, algo: Algorithm):

                options = handler.get_options(Problem(self.problemWidget.value), algo)

                # create options widgets for selected algorithm
                if algo == Algorithm.GVNS:
                        return self.get_gvns_options(options)
                if algo == Algorithm.GRASP:
                        return self.get_grasp_options(options)
                if algo == Algorithm.TS:
                        return self.get_ts_options(options)
                return ()


        # define option widgets for each algorithm
        def get_gvns_options(self, options: dict):

                ch = widgets.Dropdown( options = [m[0] for m in options[Option.CH]],
                                description = Option.CH.value)
                ch_box = widgets.VBox([ch])
                self.optionsHandles[Option.CH] = ch

                li_box = self.get_neighborhood_options(options, Option.LI)
                sh_box = self.get_neighborhood_options(options, Option.SH)

                return (ch_box,li_box,sh_box)


        def get_grasp_options(self, options: dict):

                li_box = self.get_neighborhood_options(options, Option.LI)
                rcl = widgets.RadioButtons(options=[m[0] for m in options[Option.RGC]],
                                                description=Option.RGC.value)
                k_best = widgets.IntText(value=1,description='k:',layout=widgets.Layout(width='150px', display='None'))
                alpha = widgets.FloatSlider(value=0.85, description='alpha:',step=0.05,layout=widgets.Layout(display='None'), orientation='horizontal', min=0, max=1)

                param = widgets.HBox([k_best,alpha])
                rcl_box = widgets.VBox()

                def set_param(change):
                        if change['new'] == 'k-best':
                                k_best.layout.display = 'flex'
                                alpha.layout.display = 'None'
                                rcl_box.children = (rcl,k_best)
                        if change['new'] == 'alpha':
                                k_best.layout.display = 'None'
                                alpha.layout.display = 'flex'
                                rcl_box.children = (rcl,alpha)

                rcl.observe(set_param, names='value')
                self.optionsHandles[Option.RGC] = rcl_box
                set_param({'new':rcl.value})

                return (li_box,rcl_box)

        def get_ts_options(self, options: dict):

                ch = widgets.Dropdown( options = [m[0] for m in options[Option.CH]],
                        description = Option.CH.value)
                self.optionsHandles[Option.CH] = ch
                ch_box = widgets.VBox([ch])
                li_box = self.get_neighborhood_options(options, Option.LI)
                min_ll = widgets.IntText(value=5,description='min length', layout=widgets.Layout(width='150px'), disabled=True)
                max_ll = widgets.IntText(value=5,description='max length', layout=widgets.Layout(width='150px'))
                iter_ll = widgets.IntText(value=0,description='change (iteration)', layout=widgets.Layout(width='150px'))
                
                def on_change_min(change):
                        if change.new > max_ll.value:
                                min_ll.value = max_ll.value
                        if change.new <= 0:
                                min_ll.value = 1
                def on_change_max(change):
                        if change.new < min_ll.value and iter_ll.value > 0:
                                max_ll.value = min_ll.value
                        if change.new <= 0:
                                max_ll.value = 1
                        if iter_ll.value == 0:
                                min_ll.value = max_ll.value
                def on_change_iter(change):
                        if change.new < 0:
                                iter_ll.value = 0
                        if change.new > 0:
                                min_ll.disabled = False
                        if change.new == 0:
                                min_ll.value = max_ll.value
                                min_ll.disabled = True
                

                min_ll.observe(on_change_min, names='value')
                max_ll.observe(on_change_max, names='value')
                iter_ll.observe(on_change_iter, names='value')
                label = widgets.Label(value='Tabu List')
                ll_box = widgets.HBox([label,min_ll,max_ll,iter_ll])
                self.optionsHandles[Option.TL] = ll_box

                return (ch_box,li_box,ll_box)

        # helper functions to create widget box for li/sh neighborhoods
        def get_neighborhood_options(self, options: dict, phase: Option):
                available = widgets.Dropdown(
                                options = [m[0] for m in options[phase]],
                                description = phase.value
                )


                size = widgets.IntText(value=1,description='k: ',layout=widgets.Layout(width='150px'))
                add = widgets.Button(description='',icon='chevron-right',layout=widgets.Layout(width='60px'), tooltip='add ' + phase.name)
                remove = widgets.Button(description='',icon='chevron-left',layout=widgets.Layout(width='60px'), tooltip='remove ' + phase.name)
                up = widgets.Button(description='',icon='chevron-up',layout=widgets.Layout(width='30px'), tooltip='up ' + phase.name)
                down = widgets.Button(description='',icon='chevron-down',layout=widgets.Layout(width='30px'), tooltip='down ' + phase.name)
                
                def on_change_available(event):
                        opt = handler.get_options(Problem(self.problemWidget.value),Algorithm(self.algoWidget.value))
                        opt = [o for o in opt.get(phase) if o[0]==available.value][0]
                        size.value = 1 if opt[1] == None or type(opt[1]) == type else opt[1]
                        size.disabled = type(opt[1]) != type or opt[1] == None
                on_change_available(None)

                available.observe(on_change_available, names='value')

                selected = widgets.Select(
                                options = [],
                                description = 'Selected'
                )

                self.optionsHandles[phase] = selected

                add.on_click(self.on_add_neighborhood)
                remove.on_click(self.on_remove_neighborhood)
                up.on_click(self.on_up_neighborhood)
                down.on_click(self.on_down_neighborhood)

                middle = widgets.Box([size, add, remove],layout=widgets.Layout(display='flex',flex_flow='column',align_items='flex-end'))
                sort = widgets.VBox([up,down])
                
                return widgets.HBox([available,middle,selected,sort])        

                
        def on_add_neighborhood(self,event):
                phase = event.tooltip.split(' ')[1]
                descr = dict(Option.__members__.items()).get(phase).value
                n_block = [c for c in self.optionsWidget.children if c.children[0].description == descr][0]
                selected = n_block.children[2]
                size = n_block.children[1].children[0].value
                sel = n_block.children[0].value
                selected.options += (f'{sel}, k={max(1,size)}',)
                selected.index = len(selected.options) -1


        def on_remove_neighborhood(self,event):
                selected = self.get_selected_nh(event)
                
                if len(selected.options) == 0:
                        return

                to_remove = selected.index
                options = list(selected.options)
                del options[to_remove]
                selected.options = tuple(options)

        def on_up_neighborhood(self,event):
                selected = self.get_selected_nh(event)

                if len(selected.options) == 0:
                        return

                to_up = selected.index
                if to_up == 0:
                        return
                options = list(selected.options)
                options[to_up -1], options[to_up] = options[to_up], options[to_up-1]
                selected.options = tuple(options)
                selected.index = to_up -1

        def on_down_neighborhood(self,event):
                selected = self.get_selected_nh(event)

                if len(selected.options) == 0:
                        return

                to_down = selected.index
                if to_down == (len(selected.options) - 1):
                        return
                options = list(selected.options)
                options[to_down +1], options[to_down] = options[to_down], options[to_down+1]
                selected.options = tuple(options)
                selected.index = to_down +1

        def get_selected_nh(self, event):
                phase = event.tooltip.split(' ')[1]
                descr = dict(Option.__members__.items()).get(phase).value
                n_block = [c for c in self.optionsWidget.children if c.children[0].description == descr][0]
                return n_block.children[2]
                
                        



class InterfaceRuntimeAnalysis(InterfaceVisualisation):

        def __init__(self):
                super().__init__(visualisation=False)
                self.configurations = {}
                self.out = widgets.Output()

                self.line_checkboxes = widgets.VBox()
                self.run_button.description = 'Run configuration'
                self.iter_slider = widgets.IntSlider(layout=widgets.Layout(padding="2em 0 0 0"),description='iteration', value=1, min=1)
                self.plot_options = None
                plt.rcParams['axes.spines.left'] = True
                plt.rcParams['axes.spines.right'] = True
                plt.rcParams['axes.spines.top'] = True
                plt.rcParams['axes.spines.bottom'] = True
                plt.rcParams['figure.figsize'] = (12,7)
                plt.rcParams['figure.autolayout'] = True

                #self.iteration_df = pd.DataFrame()
                #self.summaries = {}
                self.run_data = RunData()


        def display_widgets(self):

                self.problemWidget.observe(self.on_change_problem, names='value')
                self.algoWidget.observe(self.on_change_algo, names='value')
                self.optionsWidget.children = self.get_options(Algorithm(self.algoWidget.value))
                self.run_button.on_click(self.run)
                self.settingsWidget.children[0].children[2].value = 5
                self.settingsWidget.children[0].children[2].layout.display = 'flex'
                self.settingsWidget.children[0].children[3].layout.display = 'flex'
                self.settingsWidget.children[0].children[0].value = 100
                reset = widgets.Button(description='Reset', icon='close',layout=widgets.Layout(width='100px',justify_self='end'))
                save_selected = widgets.Button(description='Save selected runs',layout=widgets.Layout(width='150px',justify_self='end'))

                def on_reset(event):
                        self.settingsWidget.children[0].children[0].value = 100
                        self.settingsWidget.children[0].children[1].value = 0
                        self.settingsWidget.children[0].children[2].value = 5
                        self.configurations = {}
                        for widget in [self.problemWidget,self.instanceWidget,self.settingsWidget.children[0].children[0]]:
                                widget.disabled = False

                        self.on_change_algo(None)
                        self.out.clear_output()
                        plt.close()
                        self.line_checkboxes.children = []
                        self.plot_options.layout.visibility = 'hidden'
                        #self.iteration_df = pd.DataFrame()
                        #self.summaries = {}
                        self.run_data.reset()

                def on_change_iter(change):
                        i = change.new if change.new > 0 else 1
                        self.plot_comparison(i)

                self.iter_slider.observe(on_change_iter,names='value')

                solutions = widgets.RadioButtons(options=['best solutions','current solutions'], layout=widgets.Layout(width='auto', grid_area='sol'))
                checkboxes = []
                for o in ['max','min','polygon','median','mean','best']:
                        checkboxes.append(self.init_checkbox(o))
                        checkboxes[-1].layout = widgets.Layout(width='auto', grid_area=o)
                        checkboxes[-1].indent= False

                def on_change_plotoptions(change):
                        self.plot_comparison(self.iter_slider.value)

                solutions.observe(on_change_plotoptions,names='value')
                self.iter_slider.layout.grid_area='iter'
                self.iter_slider.indent = False
                self.plot_options = widgets.GridBox(children=[solutions, self.iter_slider] + checkboxes, 
                                        layout=widgets.Layout(
                                                padding='1em',
                                                border='solid black 1px',
                                                visibility='hidden',
                                                width='30%',
                                                grid_template_rows='auto auto auto auto',
                                                grid_template_columns='40% 30% 30%',
                                                grid_template_areas='''
                                                "sol max median"
                                                "sol min mean"
                                                ". polygon best"
                                                " iter iter iter"
                                                '''))


                reset.on_click(on_reset)
                save_selected.on_click(self.save_runs)


                display(widgets.VBox([self.settingsWidget,self.problemWidget,self.instanceWidget,self.algoWidget,self.optionsWidget,
                        self.run_button, self.line_checkboxes, widgets.HBox([save_selected,reset])]))
                display(self.plot_options)
                display(self.out)
                on_reset(None)



        def save_runs(self,event):

                checked = {config.description for config in self.line_checkboxes.children if config.value}
                not_saved = {name for name,config in self.configurations.items() if config.runs > len(config.saved_runs)}
                to_save = checked.intersection(not_saved)

                for s in to_save:
                        config = self.configurations[s]
                        description = self.create_configuration_description(config)
                        name = f'i{config.iterations}_s{config.seed}_' + s[s.find('.')+1:].strip()
                        filepath = self.configuration_exists_in_saved(name, description)

                        if filepath:
                                if config.seed == 0:
                                        self.run_data.save_to_logfile(config,filepath,append=True)
                                        self.configurations[config.name].saved_runs = list(range(1,config.runs+1))
                                        return
                                elif config.runs <= len(config.saved_runs):
                                        # do nothing, only existing runs were loaded
                                        return
                                else:
                                        # overwrite existing file
                                        self.run_data.save_to_logfile(config,filepath)
                                        self.configurations[config.name].saved_runs = list(range(1,config.runs+1))
                                        return
                        # create new file and write to it
                        filepath = 'logs'+ os.path.sep + 'saved_runtime' + os.path.sep + config.problem.name.lower() + os.path.sep +\
                                          f'r{config.runs}_' + name + '_' + time.strftime('_%Y%m%d_%H%M%S') + '.log'
                        self.run_data.save_to_logfile(config,filepath,description=description)
                        self.configurations[config.name].saved_runs = list(range(1,config.runs+1))

                                        
        '''                              
        def save_to_logfile(self, config: Configuration, filepath: str, description: str=None, append: bool=False):

                mode = 'w' if description else 'r+'
                f = open(filepath, mode)

                if description:
                        f.write(description+'\n')
                        df = self.iteration_df[config.name].T
                        df.to_csv(f,sep=' ',na_rep='NaN', mode='a', line_terminator='\n')
                        f.write('S summary\n')
                        self.summaries[config.name].to_csv(f,na_rep='NaN', sep=' ',mode='a',line_terminator='\n')
                        f.close()
                else:
                        saved_runs = set(config.saved_runs)
                        runs = set(range(1,config.runs+1))
                        to_save = list(runs - saved_runs)
                        to_save.sort()
                        if len(to_save) == 0:
                                f.close()
                                return
                        data = f.readlines()
                        df = self.iteration_df[config.name][to_save].T
                        sm = self.summaries[config.name].loc[to_save]
                        existing_runs =int(data[0].split('=')[1].strip())
                        if append: #seed==0

                                idx = next((i for i,v in enumerate(data) if v.startswith('S ')), 0)
                                df.index = pd.Index(range(existing_runs+1,existing_runs+1+len(to_save)))
                                sm.index = pd.MultiIndex.from_tuples(zip(df.index.repeat(len(sm)/len(to_save)),sm.index.get_level_values(1)),names=sm.index.names)
                                sm.reset_index(inplace=True)
                                data.insert(idx, df.to_csv(sep=' ',line_terminator='\n',header=False))
                                data += [sm.to_csv(sep=' ',line_terminator='\n', index=False,header=False)]
                                data[0] = f'R runs={df.index[-1]}\n'
                                f.seek(0)
                                f.writelines(data)
                                f.truncate()
                                f.close()
                                
                                os.rename(filepath,filepath.replace(f'r{existing_runs}',f'r{existing_runs+len(to_save)}',1))
                           
                        else: # seed!= 0
                                if len(runs) <= existing_runs:
                                        f.close()
                                        return
                                data[0] = f"R runs={config.runs}\n"
                                idx = next((i for i,v in enumerate(data) if not v[0] in ['R','D']), 0)
                                data = data[:idx]
                                data += [df.to_csv(sep=' ',line_terminator='\n')]
                                data += ['S summary\n']
                                data += [sm.to_csv(sep=' ',line_terminator='\n')]
                                f.seek(0)
                                f.writelines(data)
                                f.truncate()
                                f.close()

                                os.rename(filepath,filepath.replace(f'r{existing_runs}',f'r{len(to_save)}',1))

                self.configurations[config.name].saved_runs = list(range(1,config.runs+1))
                       '''

        def configuration_exists_in_saved(self, name: str, description: str):
                description = description[description.find('D '):]
                path = 'logs'+ os.path.sep + 'saved_runtime' + os.path.sep + Problem(self.problemWidget.value).name.lower()
                log_files = os.listdir(path)
                for l in log_files:
                        n = l[l.find('i'):]
                        if n.startswith(name):
                                file_des = ''
                                with open(path + os.path.sep + l) as file:
                                        for line in file:
                                                if line.startswith('R'):
                                                        continue
                                                if line.startswith('D'):
                                                        file_des += line
                                                else:
                                                        break
                                if file_des.strip() == description:
                                        return path + os.path.sep + l
                return False
        

        def create_configuration_description(self, config: Configuration):
                s = f"R runs={config.runs}\n"
                s += f'D inst={config.instance}\n'
                s += f"D seed={config.seed}\n"
                s += f"D iterations={config.iterations}\n"

                for o,v in config.options.items():
                        v = v if type(v) == list else [v]
                        for i in v:
                                s += f'D {o.name}{i}\n'

                return s.strip()
                
                
        def init_checkbox(self,name: str):
                def on_change(change):
                        self.iter_slider.value = self.get_best_idx()
                        self.plot_comparison(self.iter_slider.value)
                        
                cb = widgets.Checkbox(description=name, value=True)
                cb.observe(on_change,names='value')
                return cb
        

        def run(self, event):
                text = widgets.Label(value='running...')
                display(text)
                # disable prob + inst + iteration + run button
                self.problemWidget.disabled = self.instanceWidget.disabled = True
                self.settingsWidget.children[0].children[0].disabled = True
                self.run_button.disabled = True

                # prepare params and name, save params in dict of configurations
                params = self.prepare_parameters()

                # run algorithm with params or load data from file
                log_df,summary = self.load_datafile_or_run_algorithm(params)

                text.layout.display = 'None'
                self.run_button.disabled = False

                self.run_data.iteration_df = pd.concat([self.run_data.iteration_df,log_df], axis=1)
                self.run_data.summaries[params.name] = summary

                # add name to checkbox list
                self.line_checkboxes.children +=(self.init_checkbox(params.name),)
                #self.iter_slider.layout.visibility = 'visible'
                self.plot_options.layout.visibility = 'visible'

                # plot checked data
                self.iter_slider.value = self.get_best_idx()
                self.iter_slider.max = len(self.run_data.iteration_df)
                self.plot_comparison(self.iter_slider.value)


        def load_datafile_or_run_algorithm(self,params: Configuration):
                #settings = params['settings']
                if params.use_runs:
                        name = f'i{params.iterations}_s{params.seed}_' + params.name[params.name.find('.')+1:].strip()
                        description = self.create_configuration_description(params)
                        file_name = self.configuration_exists_in_saved(name,description)
                        if file_name:
                                f = open(file_name, 'r')
                                ex_runs = int(f.readline().split('=')[1].strip())
                                if params.runs <= ex_runs:
                                        data, sm = self.run_data.load_datafile(file_name,params.runs)
                                        data.columns = pd.MultiIndex.from_tuples(zip([params.name]*len(data.columns), data.columns))
                                        self.configurations[params.name].saved_runs = list(data.columns.get_level_values(1))
                                        return data, sm
                                if params.seed == 0:
                                        runs = params.runs
                                        # load existing runs
                                        data, sm = self.run_data.load_datafile(file_name,ex_runs)
                                        data.columns = pd.MultiIndex.from_tuples(zip([params.name]*len(data.columns), data.columns))
                                        # generate runs-ex_runs new ones and set correct run numbers
                                        params.runs = runs-ex_runs
                                        new_data, new_sm = handler.run_algorithm_comparison(params)
                                        params.runs = runs
                                        new_data.columns = pd.MultiIndex.from_tuples([(n,int(r)+ex_runs) for (n,r) in new_data.columns])
                                        new_sm.index = pd.MultiIndex.from_tuples([(int(r)+ex_runs,m) for (r,m) in new_sm.index])

                                        self.configurations[params.name].saved_runs = list(data.columns.get_level_values(1))
                                        # concatenate them
                                        return pd.concat([data,new_data],axis=1),pd.concat([sm,new_sm])
                                
                return handler.run_algorithm_comparison(params)
        '''
        def load_datafile(self,filename,runs: int):
                f = open(filename, 'r')
                pos = 0
                while True:
                        l = f.readline()
                        if not l[0] in ['D','R']:
                                break
                        pos = f.tell()
                f.seek(pos)
                data = pd.read_csv(f, sep=r'\s+', nrows=runs).T
                data.reset_index(drop=True, inplace=True)
                data.index += 1
                f.seek(pos)
                
                while True:
                        l = f.readline()
                        if l[0] == 'S':
                                break
                sm = pd.read_csv(f,sep=r'\s+',index_col=['run','method'])
                sm = sm[sm.index.get_level_values('run') <= runs]
                f.close()

                return data,sm
        '''

        def get_best_idx(self):
                checked = [c.description for c in self.line_checkboxes.children if c.value]
                if checked == []:
                        return 1
                df = self.run_data.iteration_df[checked]
                m = 1
                if Problem(self.problemWidget.value) in [Problem.MAXSAT,Problem.MISP]:
                        m = df.max().max()
                else:
                        m = df.min().min()
                return df.loc[df.isin([m]).any(axis=1)].index.min()


        def prepare_parameters(self):
                params = super().prepare_parameters()
                params.runs = self.settingsWidget.children[0].children[2].value
                params.use_runs = self.settingsWidget.children[0].children[3].value
                name = [f'{params.algorithm.name.lower()}']
                for k,v in params.options.items():
                        if not type(k) == Option or len(v) == 0:
                                continue
                        o = k.name.lower()+ '-' + '-'.join([str(p[1]) for p in v])
                        o = o.replace('.','')
                        name += [o]

                count = len(self.line_checkboxes.children) + 1
                params.name = str(count) + '. ' + '_'.join(name)
                self.configurations[params.name] = params
                return params


        def plot_comparison(self, i):
                with self.out:
                        fig = plt.figure(num=f'{self.problemWidget.value}',clear=True)
                        g = gs.GridSpec(3,2)
                        ax = fig.add_subplot(g[0:2,:])
                        ax_bb = fig.add_subplot(g[2,0])
                        ax_sum = fig.add_subplot(g[2,1])
                        
                        legend_handles=[]
                        checked = [c.description for c in self.line_checkboxes.children if c.value]
                        if checked == []:
                                return
                        sol = 'best' if self.plot_options.children[0].value.startswith('best') else 'current'
                        lines = [o.description for o in self.plot_options.children[1:] if o.value]
                        selected_data = self.run_data.iteration_df[checked]

                        if sol == 'best':
                                selected_data = selected_data.cummax(axis=0) if Problem(self.problemWidget.value) in [Problem.MAXSAT,Problem.MISP] \
                                                else selected_data.cummin(axis=0)
                                
                        for i,c in enumerate(checked):

                                col = f'C{int(c.split(".")[0]) % 10}'
                                df = selected_data[c]

                                maxi = df.max(axis=1)
                                mini = df.min(axis=1)
                                if 'max' in lines:
                                        maxi.plot(color=col, ax=ax)
                                if 'min' in lines:
                                        mini.plot(color=col, ax=ax)
                                if 'polygon' in lines:
                                        ax.fill_between(df.index, maxi, mini, where=maxi > mini , facecolor=col, alpha=0.2, interpolate=True)
                                if 'mean' in lines:
                                        m = df.mean(axis=1)
                                        m.plot(color=col, ax=ax, linestyle='dotted')
                                if 'median' in lines:
                                        m = df.median(axis=1)
                                        m.plot(color=col, ax=ax,linestyle='dashed')
                                legend_handles += [Line2D([0],[0],color=col,label=c + f' (n={len(self.run_data.iteration_df[c].columns)},s={self.configurations[c].seed})')]
                        loc = ''
                        best = None

                        if Problem(self.problemWidget.value) in [Problem.MAXSAT,Problem.MISP]:
                                best = selected_data.cummax(axis=0).cummax(axis=1).iloc[:,-1:]
                                loc = 'lower right'
                        else:
                                best = selected_data.cummin(axis=0).cummin(axis=1).iloc[:,-1:]
                                loc = 'upper right'
                        if 'best' in lines:
                                best.plot(color='black',ax=ax)
                                legend_handles += [Line2D([0],[0],color='black',label='best')]
                        ax.legend(handles=legend_handles,loc=loc)

                        ax.axvline(x=self.iter_slider.value)

                        bb_data = selected_data
                        bb_data = bb_data.loc[self.iter_slider.value].reset_index(level=1, drop=True).reset_index()
                        bb_data = bb_data.rename(columns={self.iter_slider.value:f'iteration={self.iter_slider.value}'})
                        bb_data.boxplot(by='index',rot=25,ax=ax_bb)
                        fig.suptitle('')
                        ax_bb.set_xlabel('')
                        ax_bb.set_ylabel('objective value')

                        widgets.interaction.show_inline_matplotlib_plots()
                


        
        



        
        




    