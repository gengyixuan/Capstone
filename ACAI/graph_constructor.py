import yaml
import io
import os
from utils import *

import os
import sys
sys.path.append(os.path.dirname(os.path.realpath('__file__')) + '/../../')
from acaisdk.file import File
from acaisdk.project import Project
from acaisdk.fileset import FileSet
from acaisdk.job import Job
from acaisdk.meta import *
from acaisdk.utils import utils

utils.DEBUG = True  # print debug messages

# class ComputeNode:
#     def __init__(self, node_name="", script_path="", script_version=0,
#                  input_nodes=[], output_nodes=[], hyperparams={}):
#         self.NodeName = node_name
#         self.ScriptName = script_path
#         self.ScriptVersion = script_version
#         self.InputNodes = input_nodes
#         self.OutputNodes = output_nodes
#         self.HyperParameter = hyperparams
    
#     def toStr(self):
#         template = "name: {}\nscript: {}\nversion: {}\ninput: {}\noutput: {}\nhparams: {}"
#         in_node_names = [node.NodeName for node in self.InputNodes]
#         out_node_names = [node.NodeName for node in self.OutputNodes]
#         string = template.format(self.NodeName, self.ScriptName, self.ScriptVersion, 
#                                  ",".join(in_node_names), ",".join(out_node_names), str(self.HyperParameter))
#         return string

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
        path = os.path.join(self.workspace, path)
        return os.path.isfile(path) or os.path.isdir(path)

    def getmtime(self, path):
        path = os.path.join(self.workspace, path)
        if self.isExist(path):
            return str(os.path.getmtime(path))
        else:
            return None
    
    def load_graph(self):
        # Read config YAML file
        with open(self.workspace+"config.yaml", 'r') as stream:
            config = yaml.safe_load(stream)

        # read version history from disk
        history = {}
        if self.isExist('_history.yaml'):
            with open(self.workspace+"_history.yaml", 'r') as stream:
                history = yaml.safe_load(stream)

        # read credentials
        project_name = config['pipeline']['project_name']
        user_name = config['pipeline']['user_name']
        admin_token = config['pipeline']['admin_token']

        # create project / user if this is first time
        if not history:
            r = Project.create_project(project_name, admin_token, 'proj_admin')
            r = Project.create_user(project_name, r['project_admin_token'], user_name)
            history['credentials'] = {'ACAI_PROJECT': project_name,
                                      'ACAI_TOKEN': r['user_token']}
        
        # update environment var with credentials
        for key in history['credentials']:
            os.environ[key] = history['credentials'][key]

        # create version map under history if history was empty
        if 'versions' not in history:
            history['versions'] = {}
        version_map = history['versions']

        # create compute nodes
        compute_nodes = {}
        uploaded = {}
        for module in config['modules']:
            node_name = module['node']
            script_path = module['script']

            # compute version of this node
            script_version = 0
            currtime = self.getmtime(script_path)
            currdepstime = {}
            dependencies = {} if 'dependencies' not in module else module['dependencies']
            for path in dependencies:
                currdepstime[path] = self.getmtime(path)
            if script_path not in version_map:
                version_map[script_path] = {'version': 0, 'script_lastmtime': currtime, 'deps': currdepstime}
            else:
                lastversion = version_map[script_path]['version']
                lasttime = version_map[script_path]['script_lastmtime']
                lastdeps = version_map[script_path]['deps']
                # check if everything is the same as last time, upload the changed dependencies
                all_same = True
                if lasttime != currtime:
                    all_same = False
                for path in currdepstime:
                    if path not in lastdeps or lastdeps[path] != currdepstime[path]:
                        all_same = False
                        # upload curr version of path files to ACAI cloud if not yet uploaded
                        if path not in uploaded:
                            uploaded.add(path)
                            input_dir = os.path.join(self.workspace, path)
                            File.upload([(input_dir, path)], [])\
                                .as_new_file_set()
                        
                if all_same:
                    script_version = lastversion
                else:
                    script_version = lastversion + 1
                    version_map[script_path] = {'version': script_version, 'script_lastmtime': currtime, 'deps': currdepstime}
            
            # continue building the new node
            hyperparams = module['params']
            input_nodes = [] if 'input_nodes' not in module else module['input_nodes']
            newnode = Node(node_name=node_name,
                          script_path=script_path,
                          script_version=script_version, 
                          input_nodes=input_nodes,
                          output_nodes=[],
                          dependencies=module['dependencies'] if 'dependencies' in module else [],
                          hyperparams=hyperparams)
            compute_nodes[node_name] = newnode

        # save new version map to disk
        with io.open(self.workspace+'_history.yaml', 'w+', encoding='utf8') as outfile:
            yaml.dump(history, outfile, default_flow_style=False, allow_unicode=True)

        # second pass to connect in/out edges between nodes
        graph = []
        for node_name in compute_nodes:
            node = compute_nodes[node_name]
            input_node_names = node.input_nodes
            node.input_nodes = []
            input_nodes = [compute_nodes[name] for name in input_node_names]
            for in_node in input_nodes:
                node.input_nodes.append(in_node)
                in_node.output_nodes.append(node)
            graph.append(node)
        return graph


        