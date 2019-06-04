
class Graph:
    def __init__(self):
        self.node_list = []
        self.operator_list = []
        self.relation = []

    def insert(self, from1, from2, operator, variable):
        self.relation.append([from1, from2, operator, variable])



class Relation:
    """
    Relation 用来记录从两个variable map 到一个新的variable的过程
    """
