# for node class
class Node:
    def __init__(self, node_name="", script_name="", script_version=0, 
                 input_nodes=[], output_nodes=[], hyperparams={}):
        self.NodeName = node_name
        self.ScriptName = script_name
        self.ScriptVersion = script_version
        self.InputNodes = input_nodes
        self.OutputNodes = output_nodes
        self.HyperParameter = hyperparams
    
    def toStr(self):
        template = "name: {}\nscript: {}\nversion: {}\ninput: {}\noutput: {}\nhparams: {}"
        in_node_names = [node.NodeName for node in self.InputNodes]
        out_node_names = [node.NodeName for node in self.OutputNodes]
        string = template.format(self.NodeName, self.ScriptName, self.ScriptVersion, 
                                 ",".join(in_node_names), ",".join(out_node_names), str(self.HyperParameter))
        return string