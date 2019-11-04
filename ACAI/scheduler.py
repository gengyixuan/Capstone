import time
import sys
from threading import Thread
from zipfile import ZipFile

from log_manager import LogManager
from constants import *
from acaisdk import acaisdk


class Scheduler:
    # graph: list of Node
    def __init__(self, graph):
        self.nodeVersions = dict()
        self.graph = graph
        self.log_manager = LogManager()

    # Build script files for each user-provided function
    # by adding input and output processer
    def build_scripts(self, node):
        name = node.script_name
        fs = open("{}.py".format(name), "w")
        # Import necessary function & tool
        fs.write("from {} import {}\n".format(name, name))
        fs.write("import argparse\n")
        fs.write("import pickle as pkl\n")
        # Build argument parser
        fs.write("parser=argparse.ArgumentParser()\n")
        for hp in node.hyper_parameter:
            fs.write("parser.add_argument('--{}')\n".format(hp))
        fs.write("args=parser.parse_args()\n")
        # Get input data and hyper parameters
        fs.write("inputs=dict()\n")
        for in_node in node.input_nodes:
            fs.write("inputs[{}]=pkl.load(open('{}.pkl', 'rb'))".format(in_node.node_name, in_node.node_name))
        fs.write("hps=dict()\n")
        for hp in node.hyper_parameter:
            fs.write("hps[{}]=args.{},".format(hp, hp))
        # Call function
        fs.write("rst={}(inputs, hps)".format(name))
        # Save the result
        fs.write("pkl.dump(rst, open('{}/{}.pkl', 'wb'))".format(OUTPUT_PATH, name))
        # Compress the script and submit to ACAI system
        fs.close()
        with ZipFile("{}.zip".format(name), "w") as zipf:
            zipf.write("{}.py".format(name))
        acaisdk.file.File.upload([("{}.zip".format(name), "{}.zip".format(name))])

    def run_workflow(self):
        # Build graph for topological sort
        for node in self.graph:
            node.inputNodesNum = len(node.inputNodes)
            for pre in node.inputNodes:
                pre.outputNodes.append(node)
        # Execute nodes with zero indegree
        q = []
        for node in self.graph:
            if node.inputNodes == 0:
                q.append(node)
        # Count the number of executed nodes
        exec_count = 0
        # Keep looping until all nodes are executed
        while exec_count < len(self.graph):
            # Constantly check if new nodes are added to the queue
            while not q:
                time.sleep(5000)
            # Submit current node for execution in a new thread
            runNode = Thread(target=self.submit_node, args=(q.pop(0), q))
            runNode.start()

    # Submit all jobs for target node to ACAI System
    # node: target Node
    # q: Queue of Nodes
    def submit_node(self, node, q):
        # Go through hyper parameters and input node versions
        hp_list = self.grid_search(node.hyperParameters)
        input_nodes_versions = self.grid_search(node.inputNodes)
        jobs = []
        submitted = False
        for hp in hp_list:
            for input_nodes in input_nodes_versions:
                shouldRun = self.log_manager.experiment_run(node.NodeName, node.ScriptVersion, hp, input_nodes)
                if not shouldRun:
                    continue
                if not submitted:
                    self.build_scripts(node)
                runJob = Thread(target=self.submit_job, args=(node, hp, input_nodes, self.log_manager))
                runJob.start()
                jobs.append(runJob)
        # Waiting for all jobs to finish
        for j in jobs:
            j.join()
        # After this node is finished, check its descendants
        # for executable nodes (nodes with 0 indegree)
        for out in node.outputNodes:
            out.inputNodesNum -= 1
            if out.inputNodesNum == 0:
                q.append(out)

    # Submit a job to ACAI System
    # node: target Node
    # hp: hyper parameter setting for this job
    # input_nodes: input node versions for this job
    @staticmethod
    def submit_job(node, hp, input_nodes, log_manager):
        name = node.node_name
        command = "python {}.py ".format(name)
        for key in hp:
            command += "--{} {} ".format(key, hp[key])
        fileset_list = []
        for in_node in input_nodes:
            fileset_list.append("@{}:{}".format(in_node, input_nodes[in_node]))
        input_file_set = acaisdk.fileset.FileSet.create_file_set("{}_input".format(name), fileset_list)
        attr = {
            "v_cpu": "0.5",
            "memory": "320Mi",
            "gpu": "0",
            "command": command,
            "container_image": "pytorch/pytorch",
            'input_file_set': input_file_set['id'],
            'output_path': OUTPUT_PATH,
            'code': '{}.zip'.format(name),
            'description': 'a job for {}'.format(name),
            'name': name
        }
        output_version = acaisdk.job.Job().with_attributes(attr).register().run()
        log_manager.save_output_data(node.NodeName, node.ScriptVersion, hp, input_nodes, output_version)

    # Get all possible hyper parameter settings
    # candidates: dict (key: string, value: list)
    @staticmethod
    def grid_search(candidates):
        combo_list = [{}]
        for name in candidates:
            for cur_combo in combo_list:
                for value in candidates[name]:
                    new_hp = cur_combo.copy()
                    new_hp[name] = value
                    combo_list.append(new_hp)
                combo_list.remove(cur_combo)
        return combo_list
    
    # Reach goal
    # def bayesian_search(self):