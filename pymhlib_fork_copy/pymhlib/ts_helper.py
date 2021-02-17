import random as rd
from typing import Any

class TabuAttribute():

    def __init__(self, attr: Any, lifespan: int):
        self.attribute = attr
        self.lifespan = lifespan

    def expired(self):
        return self.lifespan == 0

    def update(self):
        self.lifespan -= 1
        return self

    def __repr__(self) -> str:
        return str((self.attribute,self.lifespan))

    def __eq__(self, o) -> bool:
        if o == None or not isinstance(o, TabuAttribute):
            return False
        return (self.attribute == o.attribute) and (o.lifespan == self.lifespan)



class TabuList():

    def __init__(self,min_ll, max_ll, change_ll_iter):
        self.min_ll = min_ll
        self.max_ll = max_ll
        self.change_ll_iter = change_ll_iter if self.min_ll < self.max_ll else 0
        self.tabu_list = []
        self.current_ll = rd.choice(range(self.min_ll,self.max_ll+1))

    def __repr__(self) -> str:
        return str(self.tabu_list)

    def add_attribute(self, attribute: Any, lifespan: int):
        if not attribute:
            return
        self.tabu_list.append(TabuAttribute(attribute,lifespan))

    def generate_list_length(self, current_iter: int):
        if self.change_ll_iter == 0:
            return self.current_ll
        if current_iter % self.change_ll_iter == 0:
            self.current_ll = rd.choice(range(self.min_ll,self.max_ll+1))
        return self.current_ll
        
    def delete_attribute(self, to_delete: Any):
        attr_idx = [i for i, elem in enumerate(self.tabu_list) if elem.attribute == to_delete]
        attr_idx.reverse()
        for i in attr_idx:
            self.tabu_list.pop(i)

    def update_list(self):
        self.tabu_list = list(map(lambda x: x.update(),self.tabu_list))
        self.tabu_list = [x for x in self.tabu_list if not x.expired()]
