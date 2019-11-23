class Searcher:
    def __init__(self, graph, search_method='Bayesian'):
        self.graph = graph
        self.search_method = search_method
        self.log = []

    def get_next_hps(self, last_hps, last_rst):
        pass