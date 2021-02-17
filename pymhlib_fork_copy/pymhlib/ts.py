
from pymhlib.scheduler import Method, Scheduler
from pymhlib.solution import Solution
from pymhlib.demos.maxsat import MAXSATInstance, MAXSATSolution
from pymhlib.ts_helper import TabuList

from typing import List, Callable, Any
import time
import random as rd
import logging



class TS(Scheduler):


    def __init__(self, sol: Solution, meths_ch: List[Method], meths_rli: List[Method],
                min_ll: int=5, max_ll: int=5, change_ll_iter: int=1,
                own_settings: dict = None, consider_initial_sol=False):
        super().__init__(sol, meths_ch+meths_rli, own_settings, consider_initial_sol)

        self.tabu_list = TabuList(min_ll, max_ll, change_ll_iter)
        self.meths_ch = meths_ch
        self.meths_rli = meths_rli 

    def update_tabu_list(self, sol: Solution, sol_old: Solution):
        ll = self.tabu_list.generate_list_length(self.iteration) # generate list length for current iteration

        self.step_logger.info(f'LL: {ll}')
        
        if self.incumbent_iteration == self.iteration and self.incumbent.is_tabu(self.tabu_list):
            # a new best solution was found, but it was tabu (aspiration criterion)
            # get the violated tabu attribute and delete it from the list
            tabu_violated = sol_old.get_tabu_attribute(self.incumbent)
            for t in tabu_violated:
                self.tabu_list.delete_attribute({t})

            self.step_logger.info(f'TA_DEL: {tabu_violated}')

        self.tabu_list.update_list() # updates lifespan of each tabu attribute and deletes expired attributes
        self.tabu_list.add_attribute(sol.get_tabu_attribute(sol_old), self.tabu_list.current_ll)

    
    
    def ts(self, sol: Solution):

        while True:
            # use of multiple different methods for restricted neighborhood search is possible,
            # but usually only one is used
            for m in self.next_method(self.meths_rli, repeat=True):
                sol_old = sol.copy()

                def ts_iteration(sol: Solution, _par, result):

                    for ta in self.tabu_list.tabu_list:
                        self.step_logger.info(f'TA: {ta}')

                    m.func(sol, m.par, None, self.tabu_list, self.incumbent)

                ts_method = Method(m.name, ts_iteration, m.par)

                t_start = time.process_time()
                res = self.perform_method(ts_method, sol, delayed_success=True)
                self.update_tabu_list(sol, sol_old)
                self.delayed_success_update(m, sol.obj(), t_start, sol_old)

                for ta in self.tabu_list.tabu_list:
                    self.step_logger.info(f'TA: {ta}')

                if res.terminate:
                    return


    def run(self) -> None:
        sol = self.incumbent.copy()
        assert self.incumbent_valid or self.meths_ch
        self.perform_sequentially(sol, self.meths_ch)
        self.ts(sol)





