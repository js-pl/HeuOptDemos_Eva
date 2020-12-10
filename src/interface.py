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
from IPython.display import clear_output



def load_visualisation_settings():

        interface = InterfaceVisualisation()
        interface.display_widgets()




class InterfaceVisualisation():

        def __init__(self):

                self.problemWidget = widgets.Dropdown(
                                        options = handler.get_problems(),
                                        description = 'Problem'
                                        )
                self.algoWidget =  widgets.Dropdown(
                                options = handler.get_algorithms(Problem(self.problemWidget.value)),
                                description = 'Algorithm'
                                )
                self.instanceWidget = widgets.Dropdown(
                                options = handler.get_instances(Problem(self.problemWidget.value)),
                                description = 'Instance'
                                )
                self.optionsWidget = widgets.VBox() #container which holds all options

                self.optionsHandles = {} #used to store references to relevant children of optionsWidget Box

                self.log_data = []

                self.plot_instance = None

                self.controls = self.init_controls()

                self.controls.layout.visibility = 'hidden'
                self.run_button =  widgets.Button(description = 'Run')


        def init_controls(self):

                out = widgets.Output()

                def animate(event):
                        with out:
                                p.get_animation(self.controls.children[1].value, self.log_data, self.plot_instance)
        
                play = widgets.Play(interval=1000, value=0, min=0, max=100,
                        step=1, description="Press play")
                slider = widgets.IntSlider(value=0, min=0, max=100,
                        step=1, orientation='horizontal',)

                def click_prev(event):
                        slider.value = slider.value - 1
                
                def click_next(event):
                        slider.value = slider.value + 1

                prev_iter = widgets.Button(description='<',tooltip='previous', layout=widgets.Layout(width='50px'))
                next_iter = widgets.Button(description='>', tooltip='next', layout=widgets.Layout(width='50px'))
                prev_iter.on_click(click_prev)
                next_iter.on_click(click_next)
                widgets.jslink((play, 'value'), (slider, 'value'))
                slider.observe(animate, names = 'value')

                return widgets.HBox([play, slider, prev_iter, next_iter,out])

                
        def on_change_problem(self, change):

                self.algoWidget.options = handler.get_algorithms(Problem(change.new))
                self.optionsWidget.children = self.get_options(Algorithm(self.algoWidget.value))
                self.instanceWidget.options = handler.get_instances(Problem(change.new))
                self.optionsHandles = {}
                self.on_change_algo(None)


    
        def on_change_algo(self, change):

                self.optionsHandles = {} #reset references
                self.optionsWidget.children = self.get_options(Algorithm(self.algoWidget.value))


        def run_visualisation(self, event):
                #if Problem(self.problemWidget.value).name == 'MISP' and Algorithm(self.algoWidget.value).name == 'GRASP':
                 #       self.optionsWidget.children += (widgets.Label(value='This feature will be available soon!'),)
                 #       return
                
                # prepare current widget parameters for call to run algorithm
                # store each option as list of tuples (<name>,<parameter>)
                params = {'prob':Problem(self.problemWidget.value),
                                'algo':Algorithm(self.algoWidget.value),
                                'inst':self.instanceWidget.value}
                

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

                # starts call to pymhlib in handler module
                self.log_data, instance = handler.run_algorithm(params)

               
                # initialize graph from instance
                self.plot_instance = p.get_visualisation(params['prob'],params['algo'], instance)

                self.controls.children[1].value = 0
                self.controls.children[0].max = self.controls.children[1].max = len(self.log_data) - 3
                # start drawing
                p.get_animation(self.controls.children[1].value, self.log_data, self.plot_instance)
                self.controls.layout.visibility = 'visible'


        def display_widgets(self):

                self.problemWidget.observe(self.on_change_problem, names='value')
                self.algoWidget.observe(self.on_change_algo, names='value')
                self.optionsWidget.children = self.get_options(Algorithm(self.algoWidget.value))
                self.run_button.on_click(self.run_visualisation)
                display(widgets.VBox([self.problemWidget,self.instanceWidget,self.algoWidget,self.optionsWidget,self.run_button]))
                display(self.controls)

                
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
                alpha = widgets.FloatSlider(value=0.95, description='alpha:',layout=widgets.Layout(display='None'), orientation='horizontal', min=0, max=1)
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
                add = widgets.Button(description='>',layout=widgets.Layout(width='50px'), tooltip='add ' + phase.name)
                remove = widgets.Button(description='<',layout=widgets.Layout(width='50px'), tooltip='remove ' + phase.name)

                selected = widgets.Select(
                                options = [],
                                description = 'Selected'
                )

                self.optionsHandles[phase] = selected

                add.on_click(self.on_add_neighborhood)
                remove.on_click(self.on_remove_neighborhood)

                middle = widgets.Box([size, add, remove],layout=widgets.Layout(display='flex',flex_flow='column',align_items='flex-end'))
                
                return widgets.HBox([available,middle,selected])        

                
        def on_add_neighborhood(self,event):

                phase = event.tooltip.split(' ')[1]
                descr = dict(Option.__members__.items()).get(phase).value
                n_block = [c for c in self.optionsWidget.children if c.children[0].description == descr][0]

                size = n_block.children[1].children[0].value
                sel = n_block.children[0].value
                n_block.children[2].options += (f'{sel}, k={max(1,size)}',)


        def on_remove_neighborhood(self,event):

                phase = event.tooltip.split(' ')[1]
                descr = dict(Option.__members__.items()).get(phase).value
                n_block = [c for c in self.optionsWidget.children if c.children[0].description == descr][0]
                selected = n_block.children[2]
                
                if len(selected.options) == 0:
                        return

                to_remove = selected.value
                options = list(selected.options)
                options.remove(to_remove)
                selected.options = tuple(options)
                        



        
        



        
        




    