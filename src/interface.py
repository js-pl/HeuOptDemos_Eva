"""
module which builds all necessary widgets for visualisation and runtime analysis (TODO) based on information of handler module

"""
import networkx as nx
import matplotlib.pyplot as plt
import statistics        
from IPython.display import display, display_html 
import ipywidgets as widgets
import src.handler as handler
from src.handler import Problem, Algorithm, Option
from pymhlib.demos.maxsat import MAXSATInstance, MAXSATSolution
from pymhlib.demos.misp import MISPInstance, MISPSolution
import src.plotting as p
from src.logdata import Log, LogData, save_visualisation, read_from_logfile, get_log_description
from IPython.display import clear_output
import os




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
                #set max,min,value of slider and controls to appropriate iteration number
                self.controls.children[0].max = self.controls.children[1].max = len(self.log_data.log_data) - 3
                self.controls.children[1].value = next_iter
                self.animate(None)



        def animate(self,event):
                with self.out:
                        p.get_animation(self.controls.children[1].value, self.log_data.log_data, self.plot_instance)
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
                params = dict()
                log_data = list()
                if self.mainSelection.value == 'load from log file':
                        log_data, instance = read_from_logfile(self.logfileWidget.value)
                        params.update({'prob':Problem[log_data[0].upper()], 'algo':Algorithm[log_data[1].upper()]})
                        
                else:
                        params = self.prepare_parameters()
                        # starts call to pymhlib in handler module
                        log_data, instance = handler.run_algorithm_visualisation(params)

                self.log_data = LogData(log_data)
                #print('\n'.join([str(l) for l in self.log_data.log_data]))

                # initialize graph from instance
                with self.out:
                        self.out.clear_output()
                        self.plot_instance = p.get_visualisation(params['prob'],params['algo'], instance)
                        widgets.interaction.show_inline_matplotlib_plots()

                self.controls.children[1].value = 0
                self.controls.children[0].max = self.controls.children[1].max = len(self.log_data.log_data) - 3
                self.controls.children[4].value = Log.StepInter.value
                # start drawing
                self.animate(None)
                self.controls.layout.visibility = 'visible'
                self.save_button.disabled = self.mainSelection.value == 'load from log file' 

        def prepare_parameters(self):
                # prepare current widget parameters for call to run algorithm
                params = {'prob':Problem(self.problemWidget.value),
                                'algo':Algorithm(self.algoWidget.value),
                                'inst':'-'.join([str(c.value) for c in self.instanceBox.children])}

                # store each option as list of tuples (<name>,<parameter>)
                # extend if further options are needed
                if Option.CH in self.optionsHandles:
                        params[Option.CH] = [(self.optionsHandles.get(Option.CH).value, 0)]
                if Option.LI in self.optionsHandles:
                        #TODO: make sure name splitting works if no 'k=' given (ok for now because k is always added)
                        params[Option.LI] = [(name.split(',')[0], int(name.split('=')[1])) for name in list(self.optionsHandles.get(Option.LI).options)]
                if Option.SH in self.optionsHandles:
                        params[Option.SH] = [(name.split(',')[0], int(name.split('=')[1])) for name in list(self.optionsHandles.get(Option.SH).options)]
                if Option.RGC in self.optionsHandles:
                        params[Option.RGC] = [(self.optionsHandles[Option.RGC].children[0].value,self.optionsHandles[Option.RGC].children[1].value)]

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
                optionsBox = widgets.VBox([self.problemWidget,self.instanceBox,self.algoWidget,self.optionsWidget])
                return widgets.VBox([optionsBox, self.save_button])

        def on_click_save(self,event):
                #TODO make sure params were not changed!!!!
                save_visualisation(self.prepare_parameters(), self.plot_instance)
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
                self.add_instance = widgets.Button(description='Add configuration')
                self.configurations = []
                self.selectedConfigs = widgets.Select(options=[])
                self.out = widgets.Output()

        def display_widgets(self):

                self.problemWidget.observe(self.on_change_problem, names='value')
                self.algoWidget.observe(self.on_change_algo, names='value')
                self.optionsWidget.children = self.get_options(Algorithm(self.algoWidget.value))
                self.add_instance.on_click(self.on_add_instance)
                self.run_button.on_click(self.run_visualisation)
                display(widgets.VBox([self.problemWidget,self.instanceWidget,self.algoWidget,self.optionsWidget,self.add_instance,self.selectedConfigs,self.run_button]))
                display(self.out)

        def on_add_instance(self, event):
                params = self.prepare_parameters()
                name = [f'{params.get("algo").name.lower()}']
                for k,v in params.items():
                        if not type(k) == Option or len(v) == 0:
                                continue
                        o = k.name.lower()+ '-' + '-'.join([str(p[1]) for p in v])
                        name += [o] 

                self.problemWidget.disabled = True
                self.instanceWidget.disabled = True
                # TODO enable widgets when all instances are deleted
                options = list(self.selectedConfigs.options)
                count = int(options[-1].split('.')[0]) + 1 if len(options) > 0 else 1
                options.append(str(count) + '. ' + '_'.join(name))
                params['name'] = options[-1]
                self.configurations.append(params)
                self.selectedConfigs.options = options

        def run_visualisation(self, event):
                self.out.clear_output()
                text = widgets.Label(value='running...')
                display(text)
                log_df = handler.run_algorithm_comparison(self.configurations)
                text.layout.display = 'None'
                log_df.sort_values('iteration',inplace=True)
                idx = log_df[log_df['iteration'] == 0].index
                log_df.drop(idx, inplace=True)
                plt.rcParams['axes.spines.left'] = True
                plt.rcParams['axes.spines.right'] = True
                plt.rcParams['axes.spines.top'] = True
                plt.rcParams['axes.spines.bottom'] = True
               
                with self.out:
                        
                        log_df.plot(kind='line', x='iteration', y=[c['name'] for c in self.configurations]); plt.ylabel('obj')
                        widgets.interaction.show_inline_matplotlib_plots()

                        


                



        
        



        
        




    