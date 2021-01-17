import enum

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

