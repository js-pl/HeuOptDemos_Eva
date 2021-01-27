import enum
import os
from abc import ABC, abstractmethod

from .pymhlib.demos.maxsat import MAXSATInstance, MAXSATSolution
from .pymhlib.demos.misp import MISPInstance, MISPSolution
from .pymhlib.scheduler import Method
from .pymhlib import demos


demo_data_path = os.path.dirname(demos.__file__) + os.path.sep + 'data'

# extend enums as needed, they hold the string values which are used for representation in widgets
class Problem(enum.Enum):
    MAXSAT = 'MAX-SAT'
    MISP = 'MAX-Independent Set'

class Algorithm(enum.Enum):
    GVNS = 'GVNS'
    GRASP = 'GRASP'

class Option(enum.Enum):
    CH = 'Initial Solution'
    LI = 'Local Improvement'
    SH = 'Shaking'
    RGC = 'Randomized Greedy Construction'

class InitSolution(enum.Enum):
    random = 0
    greedy = 1

class Parameters():

    def __init__(self, name: str, callback=None, type: type=None, value=None):
        self.name = name
        self.callback = callback
        self.type = type(value) if value != None else type
        self.value = value

    def get_widget_info(self):
        return (self.name,self.type if self.value == None else self.value)

    def get_method(self, par=None):
        return



class ProblemDefinition(ABC):
    """A base class for problem definition to store and retrieve available algorithms and options and problem specific solution instances.

    Attributes
        - name: name of the problem, has to correspond to name of directory where logs/instances are stored
        - options: dict of dicts of available algorithms and their corresponding available options/methodes, which have to be stored as tuples of the form: 
                    (<name for widget> , <function callback with signature (solution: Solution, par: Any, result: Result) or None>  , <type of additional parameter, fixed default or None>)
    """
    def __init__(self, name: str, options: dict):
        self.name = name
        self.options = options

    def get_algorithms(self):
        return [k.value for k,_ in self.options.items()]

    def get_options(self, algo: Algorithm):
        options = {}
        for o,m in self.options[algo].items():
            options[o] = [(t[0],t[2]) for t in m]
        return options

    @abstractmethod
    def get_solution(self, instance_path: str):
        pass

    def get_instances(self,visualisation):
        instance_path = "instances" + os.path.sep
        if os.path.isdir(instance_path + self.name):
            return os.listdir(instance_path + self.name)
        return []
    

    def get_method(self, algo:Algorithm, opt: Option, name: str, par):
        m = [t for t in self.options[algo][opt] if t[0] == name][0]
        method = Method(f'{opt.name.lower()}{par if type(m[2]) == type else m[2]}', m[1], par if type(m[2]) == type else m[2])
        return method



class MAXSAT(ProblemDefinition):

    def __init__(self):

        options = {Algorithm.GVNS: {
                                Option.CH: [(InitSolution.random.name, MAXSATSolution.construct, InitSolution.random.value)
                                            ,(InitSolution.greedy.name, MAXSATSolution.construct_greedy, InitSolution.greedy.value)
                                            ],
                                Option.LI: [('k-flip neighborhood search', MAXSATSolution.local_improve, int)],
                                Option.SH: [('k random flip', MAXSATSolution.shaking, int)]
                                },
                    Algorithm.GRASP: {
                                Option.CH: [('random', MAXSATSolution.construct, 0)],   #not needed for grasp
                                Option.LI: [('k-flip neighborhood search', MAXSATSolution.local_improve, int)],
                                Option.RGC: [('k-best', MAXSATSolution.greedy_randomized_construction, int),('alpha', MAXSATSolution.greedy_randomized_construction, float)]
                                }
                    }

        super().__init__(Problem.MAXSAT.name.lower(), options)

    def get_solution(self, instance_path: str):
        instance = MAXSATInstance(instance_path)
        return MAXSATSolution(instance)

    def get_instances(self,visualisation):
        if visualisation:
            return super().get_instances(True)
        else: 
            instances = os.listdir(demo_data_path)
            return [i for i in instances if i[-3:] == 'cnf']


class MISP(ProblemDefinition):

    def __init__(self):

        options = {Algorithm.GVNS: {
                                Option.CH: [(InitSolution.random.name, MISPSolution.construct, InitSolution.random.value)
                                            ,(InitSolution.greedy.name, MISPSolution.construct_greedy, InitSolution.greedy.value)
                                            ],
                                Option.LI: [('two-exchange random fill neighborhood search', MISPSolution.local_improve, 2)],
                                Option.SH: [('remove k and random fill', MISPSolution.shaking, int)]
                                }
                                ,
                    Algorithm.GRASP: {
                                Option.CH: [('random', MISPSolution.construct, 0)],   #not needed for grasp
                                Option.LI: [('two-exchange random fill neighborhood search', MISPSolution.local_improve, 2)],
                                Option.RGC: [('k-best', MISPSolution.greedy_randomized_construction, int),('alpha', MISPSolution.greedy_randomized_construction, float)]
                              }
                    }

        super().__init__(Problem.MISP.name.lower(), options)

    def get_solution(self, instance_path):
        file_path = instance_path
        if instance_path.startswith('random'):
            file_path = "gnm" + instance_path[6:]
        instance = MISPInstance(file_path)
        return MISPSolution(instance)


    def get_instances(self, visualisation):
        if visualisation:
            inst = super().get_instances(True)
            return inst + ['random']
        else: 
            instances = os.listdir(demo_data_path)
            return [i for i in instances if i[-3:] == 'mis']
