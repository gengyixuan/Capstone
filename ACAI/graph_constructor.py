import yaml
import io
import os

class ComputeNode:
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

# GraphConstructor:
# parse config.yaml
# for each file: 
#    load _version.yaml ({ filename: (version, time, {dependency path: time}) }) check & update version map, get curr version (compute hash for the script and all its dependent data files)
#    create node, input nodes  = [node names] 
#    add to {name: nodeptr}
# save new version maps to disk
# traverse {name: nodeptr} map, update input nodes to [node ptrs], output nodes to [node ptrs]
# return the new [nodeptr] list

# sample usage:
# gc = GraphConstructor("../sample/")
# graph = gc.load_graph()
# for node in graph:
#     print(node.toStr())

class GraphConstructor(object):
    def __init__(self, workspace):
        self.workspace = workspace
    
    def isExist(self, path):
        path = self.workspace + path
        return os.path.isfile(path) or os.path.isdir(path)

    def getmtime(self, path):
        path = self.workspace + path
        if self.isExist(path):
            return str(os.path.getmtime(path))
        else:
            return None
    
    def load_graph(self):
        # Read config YAML file
        with open(self.workspace+"config.yaml", 'r') as stream:
            config = yaml.safe_load(stream)

        # read version history from disk
        version_map = {}
        if isExist('_versions.yaml'):
            with open(workspace+"_versions.yaml", 'r') as stream:
                version_map = yaml.safe_load(stream)

        # create compute nodes
        compute_nodes = {}
        for module in config['modules']:
            node_name = module['node']
            script_name = module['script']

            # compute version of this node
            script_version = 0
            currtime = self.getmtime(script_name)
            currdepstime = {}
            dependencies = {} if 'dependencies' not in module else module['dependencies']
            for path in dependencies:
                currdepstime[path] = self.getmtime(path)
            if script_name not in version_map:
                version_map[script_name] = {'version': 0, 'script_lastmtime': currtime, 'deps': currdepstime}
            else:
                lastversion = version_map[script_name]['version']
                lasttime = version_map[script_name]['script_lastmtime']
                lastdeps = version_map[script_name]['deps']
                # check if everything is the same as last time
                all_same = True
                if lasttime != currtime:
                    all_same = False
                else:
                    for path in currdepstime:
                        if path not in lastdeps or lastdeps[path] != currdepstime[path]:
                            all_same = False
                            break
                if all_same:
                    script_version = lastversion
                else:
                    script_version = lastversion + 1
                    version_map[script_name] = {'version': script_version, 'script_lastmtime': currtime, 'deps': currdepstime} 
            
            # continue building the new node
            hyperparams = module['params']
            input_nodes = [] if 'input_nodes' not in module else module['input_nodes']
            newnode = ComputeNode(node_name=node_name,
                                  script_name=script_name, 
                                  script_version=script_version, 
                                  input_nodes=input_nodes,
                                  output_nodes=[],
                                  hyperparams=hyperparams)
            compute_nodes[node_name] = newnode

        # save new version map to disk
        with io.open(self.workspace+'_versions.yaml', 'w+', encoding='utf8') as outfile:
            yaml.dump(version_map, outfile, default_flow_style=False, allow_unicode=True)

        # second pass to connect in/out edges between nodes
        graph = []
        for node_name in compute_nodes:
            node = compute_nodes[node_name]
            input_node_names = node.InputNodes
            node.InputNodes = []
            input_nodes = [compute_nodes[name] for name in input_node_names]
            for in_node in input_nodes:
                node.InputNodes.append(in_node)
                in_node.OutputNodes.append(node)
            graph.append(node)
        return graph


        