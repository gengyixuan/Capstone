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

    # return list of all file paths in dir 
    # path: relative path to workspace
    # returned path is relative path to workspace
    def list_all_file_paths(self, path):
        # if dir is a file itself, return [dir]
        path = os.path.join(self.workspace, path)
        if os.path.isfile(path):
            return [path]
        all_rel_paths = []
        for r, d, f in os.walk(path):
            for file in f:
                all_rel_paths.append(os.path.relpath(os.path.join(r, file), start=self.workspace))
        return all_rel_paths
    
    def is_exist(self, path):
        path = os.path.join(self.workspace, path)
        return os.path.isfile(path) or os.path.isdir(path)

    def get_modified_time(self, path):
        path = os.path.join(self.workspace, path)
        if self.is_exist(path):
            return str(os.path.getmtime(path))
        else:
            return None
    
    def load_graph(self):
        # Read config YAML file
        with open(self.workspace+"config.yaml", 'r') as stream:
            config = yaml.safe_load(stream)

        # read version history from disk
        history = {}
        if self.is_exist('_history.yaml'):
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

        # get from history: last snapshot of workspace: file modified times
        file_versions = {}
        if 'file_versions' in history:
            file_versions = history['file_versions'] 
            # {filepath: {"time": latest modified time, "version": version}} dict

        # get current file modified times
        # figure out set of modified files
        # update versions map
        curr_files = self.list_all_file_paths(self.workspace)
        modified = set()
        for curr_file in curr_files:
            curr_mod_time = self.get_modified_time(curr_file)
            if curr_file not in file_versions or curr_mod_time != file_versions[curr_file]['time']:
                # curr_file is modified
                modified.add(curr_file)
                # update versions map
                curr_version = 0 if curr_file not in file_versions else file_versions[curr_file]['version'] + 1
                file_versions[curr_file] = {"time": curr_mod_time, "version": curr_version}

        # figure out set of needed files for current run
        needed = set()
        for module in config['modules']:
            script_path = module['script']
            needed.add(script_path)
            dependencies = {} if 'dependencies' not in module else module['dependencies']
            for path in dependencies:
                for needed_file in self.list_all_file_paths(path):
                    needed.add(needed_file)


        # for a file that is both needed for current run and has been modified from last snapshot, upload it
        for needed_file in needed:
            if needed_file in modified:
                input_dir = os.path.join(self.workspace, needed_file)
                print("uploading: "+needed_file)
                File.upload([(input_dir, needed_file)], []).as_new_file_set(needed_file)

        # load script versions to prepare for subsequent stage of generating "compute nodes"
        # note that a script version is dependent on both script file version and dependent file versions
        script_versions = {}
        if 'script_versions' in history:
            script_versions = history['script_versions']
            # {node_name: {"version":version, "script_path":script_path}

        # create compute nodes
        compute_nodes = {}
        uploaded = set()
        for module in config['modules']:
            node_name = module['node']
            script_path = module['script']

            # compute script version:
            if node_name not in script_versions:
                script_versions[node_name] = {"version": 0, "script_path": script_path}
            else:
                print(script_versions)
                past_version = script_versions[node_name]["version"]
                past_script = script_versions[node_name]["script_path"]
                if script_path != past_script:
                    script_versions[node_name] = {"version": past_version + 1, "script_path": script_path}
                else:
                    # get all needed files (one script file + optional data files) for this node
                    needed_files = [script_path]
                    dependencies = {} if 'dependencies' not in module else module['dependencies']
                    for path in dependencies:
                        needed_files += self.list_all_file_paths(path)

                    # if any of the needed files is in modified set, increment script version by 1
                    for needed_file in needed_files:
                        if needed_file in modified:
                            script_versions[node_name]["version"] += 1
                            break

            # continue building the new node
            hyperparams = module['params']
            input_nodes = [] if 'input_nodes' not in module else module['input_nodes']
            newnode = Node(node_name=node_name,
                          script_path=script_path,
                          script_version=script_versions[node_name]["version"],
                          input_nodes=input_nodes,
                          output_nodes=[],
                          dependencies=[] if 'dependencies' not in module else module['dependencies'],
                          hyperparams=hyperparams)
            compute_nodes[node_name] = newnode

        # save new history dict to disk
        history['script_versions'] = script_versions
        history['file_versions'] = file_versions
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


        