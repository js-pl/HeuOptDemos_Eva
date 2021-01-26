
# for testing
import sys
sys.path.append('..\\HeuOptDemos_Eva')
from pymhlib.demos.maxsat import MAXSATInstance, MAXSATSolution
########


from typing import List, Callable
import time
from pymhlib.scheduler import Method, Scheduler
from pymhlib.settings import parse_settings, get_settings_parser, settings
from pymhlib.solution import Solution


#parser = get_settings_parser()
#parser.add_argument("--mh_ts_ll", type=int, default=5,
                    #help='TS length of tabu list')
#parse_settings()

class TabuAttribute():
    # maybe use this class for variable lifespan of elements
    def __init__(self, attr,lifespan=settings.mh_ts_ll):
        self.attribute = attr
        self.lifespan = lifespan

    def remove(self):
        return self.lifespan == 0

    def update(self):
        self.lifespan -= 1
        return self



class TabuList():
    #min
    #max
    #iterations
    #list
    #generate new tabu list

    #append
    #delete
    #update list elements (-1) and remove if 0
    pass




class TS(Scheduler):

    def __init__(self, sol: Solution, meths_ch: List[Method], meths_rli: List[Method],
                get_tabu_attribute: Callable, own_settings: dict = None, consider_initial_sol=False):
        super().__init__(sol, meths_ch+meths_rli, own_settings, consider_initial_sol)

        self.tabu_list = [] #init tabulist with min max iteration
        self.meths_ch = meths_ch
        self.meths_rli = meths_rli # restricted neighborhood search with best improvement
        self.get_tabu_attribute = get_tabu_attribute # problem specific function which takes new and old solution and returns the tabu attribute


    def update_tabu_list(self, sol: Solution, sol_old: Solution):
        self.tabu_list = list(map(self.tabu_list, lambda x: x.update))
        tabu = TabuAttribute(self.get_tabu_attribute(sol, sol_old), self.own_settings.mh_ts_ll)
        self.tabu_list = [x for x in self.tabu_list if not x.remove]
        self.tabu_list.append(tabu)
    
    
    def ts(self, sol: Solution):

        while True:
            for m in self.next_method(self.meths_rli, repeat=True):
                sol_old = sol.copy()
                t_start = time.process_time()
                m.par = self.tabu_list #set tabu list as method parameter
                res = self.perform_method(m, sol, delayed_success=True)
                self.update_tabu_list(sol, sol_old)
                # sol.get_tabu_attribute(old_sol)
                self.delayed_success_update(m, sol.obj(), t_start, sol_old) # necessary?
                if res.terminate:
                    return


    def run(self) -> None:
        sol = self.incumbent.copy()
        assert self.incumbent_valid or self.meths_ch
        self.perform_sequentially(sol, self.meths_ch)
        self.ts(sol)









if __name__ == '__main__':
    from pymhlib.settings import get_settings_parser, settings
    parser = get_settings_parser()
    settings.mh_titer=50
    settings.mh_out='summary.log'
    def meth_rli(sol: MAXSATSolution, par, res):
        sol.x[0] = not sol.x[0]
        print('   restricted local search with tabulist', par)

    def get_tabu_attribute(sol, old_sol):
        print('   calculate changed elem an return it', sol, old_sol)
        elem = -1
        for i, e in enumerate(sol.x):
            if bool(e) != bool(old_sol.x[i]):
                elem = i+1 if e else -(i+1)
                print('       elem', elem)
                return elem
        return None
        
    m_rli = Method('rli1',meth_rli,None)

    inst = MAXSATInstance('maxsat-simple.cnf')
    sol = MAXSATSolution(inst)
    ts = TS(sol,[Method('ch',MAXSATSolution.construct,0)], [m_rli], get_tabu_attribute)
    ts.run()