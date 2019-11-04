# for node class
class Node:
    def __init__(self, node_name="", script_name="", script_version=0, 
                 input_nodes=[], dependencies=[] output_nodes=[], hyperparams={}):
        self.node_name = node_name
        self.script_name = script_name
        self.script_version = script_version
        self.input_nodes = input_nodes
        self.input_nodes_num = len(input_nodes)
        self.dependencies = dependencies
        self.output_nodes = output_nodes
        self.hyper_parameter = hyperparams
    
    def toStr(self):
        template = "name: {}\nscript: {}\nversion: {}\ninput: {}\noutput: {}\nhparams: {}"
        in_node_names = [node.node_name for node in self.input_nodes]
        out_node_names = [node.node_name for node in self.output_nodes]
        string = template.format(self.node_name, self.script_name, self.script_version, 
                                 ",".join(in_node_names), ",".join(out_node_names), str(self.hyper_parameter))
        return string