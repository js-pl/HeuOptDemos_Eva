import sys
import sys
import os


# for testing
from .pymhlib.demos.maxsat import MAXSATInstance, MAXSATSolution
########


from typing import List, Callable, Any
import time
from .pymhlib.scheduler import Method, Scheduler
from .pymhlib.settings import get_settings_parser, settings
from .pymhlib.solution import Solution
import random as rd



class TabuAttribute():

    def __init__(self, attr,lifespan: int):
        self.attribute = attr
        self.lifespan = lifespan

    def remove(self):
        return self.lifespan == 0

    def update(self):
        self.lifespan -= 1
        return self



class TabuList():

    def __init__(self,min_ll, max_ll, change_ll_iter):
        self.min_ll = min_ll
        self.max_ll = max_ll
        self.change_ll_iter = change_ll_iter
        self.tabu_list = []
        self.current_ll = max_ll # what should be initial list length? random?

    #generate new tabu list

    def append(self,elem: Any, current_iter: int):
        if current_iter % self.change_ll_iter == 0:
            self.current_ll = rd.choice(range(self.min_ll,self.max_ll+1))

        self.tabu_list.append(TabuAttribute(elem,self.current_ll))
        

    #delete

    def update_list(self):
        self.tabu_list = list(map(lambda x: x.update(),self.tabu_list))
        self.tabu_list = [x for x in self.tabu_list if not x.remove()]
    
    



class TS(Scheduler):

    def __init__(self, sol: Solution, meths_ch: List[Method], meths_rli: List[Method],
                #get_tabu_attribute: Callable,
                min_ll: int=5,max_ll: int=5,change_ll_iter: int=1,
                own_settings: dict = None, consider_initial_sol=False):
        super().__init__(sol, meths_ch+meths_rli, own_settings, consider_initial_sol)

        self.tabu_list = TabuList(min_ll, max_ll, change_ll_iter)
        self.meths_ch = meths_ch
        self.meths_rli = meths_rli # restricted neighborhood search with best improvement
        #self.get_tabu_attribute = get_tabu_attribute # problem specific function which takes new and old solution and returns the tabu attribute


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