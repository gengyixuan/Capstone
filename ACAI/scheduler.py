import time
import sys
from threading import Thread
from zipfile import ZipFile
from termcolor import colored

from log_manager import LogManager
from constants import *
import acaisdk


class Scheduler:
    # graph: list of Node
    def __init__(self, graph, local=False):
        self.node_versions = dict()
        self.graph = graph
        self.log_manager = LogManager()
        self.local = local
        for node in graph:
            self.node_versions[node.node_name] = []

    # Build script files for each user-provided function
    # by adding input and output processer
    def build_scripts(self, node):
        node_name = node.node_name
        script_path = node.script_path[:-3]
        script_name = script_path.split('/')[-1]
        fs = open("_{}.py".format(node_name), "w")
        # Import necessary function & tool
        fs.write("from {} import {}\n".format('.'.join(script_path.split('/')), script_name))
        fs.write("import argparse\n")
        fs.write("import pickle as pkl\n")
        fs.write("import os\n")
        # Build argument parser
        fs.write("parser=argparse.ArgumentParser()\n")
        for hp in node.hyper_parameter:
            fs.write("parser.add_argument('--{}', type={})\n".format(hp['name'],
                     'float' if hp['type'] == 'float' else 'str'))
        fs.write("args=parser.parse_args()\n")
        # Get input data and hyper parameters
        fs.write("inputs=dict()\n")
        for in_node in node.input_nodes:
            fs.write("inputs['{}']=pkl.load(open('{}_output/{}.pkl', 'rb'))\n"
                     .format(in_node.node_name, in_node.node_name, in_node.node_name))
        fs.write("hps=dict()\n")
        for hp in node.hyper_parameter:
            fs.write("hps['{}']=args.{}\n".format(hp['name'], hp['name']))
        # Call function
        fs.write("rst={}(inputs, hps)\n".format(script_name))
        # Save the result
        fs.write("os.mkdir('{}_output')\n".format(node_name))
        fs.write("pkl.dump(rst, open('{}_output/{}.pkl', 'wb'))\n".format(node_name, node_name))
        # Compress the script and submit to ACAI system
        fs.close()

        if self.local:
            return
        with ZipFile("_{}.zip".format(node_name), "w") as zipf:
            zipf.write("_{}.py".format(node_name))
        acaisdk.file.File.upload([("_{}.zip".format(node_name), "_{}.zip".format(node_name))])

    def run_workflow(self):
        print("Workflow start")
        # Build graph for topological sort
        # for node in self.graph:
        #     node.input_nodes_num = len(node.input_nodes)
        #     for pre in node.input_nodes:
        #         pre.output_nodes.append(node)

        # Execute nodes with zero indegree
        q = []
        for node in self.graph:
            if node.input_nodes_num == 0:
                q.append(node)
        print(colored("Found {} initial nodes".format(len(q)), 'blue'))
        # Count the number of executed nodes
        exec_count = 0
        # Keep looping until all nodes are executed
        while exec_count < len(self.graph):
            if not q:
                print(colored("No runnable nodes now. Waiting...", 'green'))
            # Constantly check if new nodes are added to the queue
            while not q:
                time.sleep(SLEEP_INTERVAL)
            # Submit current node for execution in a new thread
            runNode = Thread(target=self.submit_node, args=(q.pop(0), q))
            runNode.start()
            exec_count += 1

    # Submit all jobs for target node to ACAI System
    # node: target Node
    # q: Queue of Nodes
    def submit_node(self, node, q):
        print(colored("Node {} is ready to run. Starting...".format(node.node_name), 'blue'))
        # Go through hyper parameters and input node versions
        hp_list = self.grid_search_hp(node.hyper_parameter)
        input_nodes_versions = self.grid_search_nv(node.input_nodes)
        jobs = []
        submitted = False
        total = len(hp_list) * len(input_nodes_versions)
        if total == 0:
            print(colored("No jobs to explore for Node {}. "
                          "Something is wrong with the hyper-parameter"
                          " setting or previous nodes.".format(node.node_name), 'red'))
        cur = 1
        for hp in hp_list:
            for input_nodes in input_nodes_versions:
                shouldRun, version = self.log_manager.experiment_run(node.node_name, node.script_version, hp, input_nodes)
                if not shouldRun:
                    if version:
                        self.add_node_version(node, version)
                        print(colored("Skip the {}/{} job for node {}: Already run before".format(cur, total, node.node_name), 'blue'))
                    else:
                        print(colored("Skip the {}/{} job for node {}: Bad common ancestor".format(cur, total, node.node_name), 'blue'))
                    cur += 1
                    continue
                print(colored("Starting the {}/{} job for node {}".format(cur, total, node.node_name), 'blue'))
                if not submitted:
                    self.build_scripts(node)
                    submitted = True
                runJob = Thread(target=self.submit_job, args=(node, hp, input_nodes, self.log_manager))
                runJob.start()
                jobs.append(runJob)
                cur += 1
        # Waiting for all jobs to finish
        for j in jobs:
            j.join()
        print(colored("All jobs for node {} finished!".format(node.node_name), 'blue'))
        # After this node is finished, check its descendants
        # for executable nodes (nodes with 0 indegree)
        for out in node.output_nodes:
            out.input_nodes_num -= 1
            if out.input_nodes_num == 0:
                q.append(out)

    # Submit a job to ACAI System
    # node: target Node
    # hp: hyper parameter setting for this job
    # input_nodes: input node versions for this job
    def submit_job(self, node, hp, input_nodes, log_manager):
        name = node.node_name
        # Build command
        command = "python _{}.py ".format(name)
        for key in hp:
            command += "--{} {} ".format(key, hp[key])
        if self.local:
            fileset_list = []
            for in_node in input_nodes:
                fileset_list.append("{}:{}".format(in_node, input_nodes[in_node]))
            file_list = []
            for dependency in node.dependencies:
                file_list.append(dependency)
            file_list.append(node.script_path)
        else:
            fileset_list = []
            for in_node in input_nodes:
                fileset_list.append("@{}:{}".format(in_node, input_nodes[in_node]))
            for dependency in node.dependencies:
                fileset_list.append(dependency)
            fileset_list.append(node.script_path)
            input_file_set = acaisdk.fileset.FileSet.create_file_set("{}_input".format(name), fileset_list)
            attr = {
                "v_cpu": "0.5",
                "memory": "320Mi",
                "gpu": "0",
                "command": command,
                "container_image": "pytorch/pytorch",
                'input_file_set': input_file_set['id'],
                'output_path': "{}_output".format(name),
                'code': '_{}.zip'.format(name),
                'description': 'a job for {}'.format(name),
                'name': name,
                'output_file_set': name
            }
            job = acaisdk.job.Job()
            status = job.with_attributes(attr).register().run().wait()
            if status != acaisdk.job.JobStatus.FINISHED:
                print(colored("A job for node {} failed!".format(name), "red"))
                return
            output_version = job.output_file_set.split(':')[-1]
        self.add_node_version(node, output_version)
        log_manager.save_output_data(node.node_name, node.script_version, hp, input_nodes, output_version)

    def add_node_version(self, node, version):
        name = node.node_name
        self.node_versions[name].append(version)

    # Get all possible hyper parameter settings
    # hps: list of dict
    @staticmethod
    def grid_search_hp(hps):
        combo_list = [{}]
        for hp in hps:
            name = hp['name']
            new_combo_list = []
            for cur_combo in combo_list:
                if hp['type'] == 'float':
                    value = hp['start']
                    while value < hp['end']:
                        new_hp = cur_combo.copy()
                        new_hp[name] = value
                        new_combo_list.append(new_hp)
                        value += hp['step_size']
                elif hp['type'] == 'collection':
                    for value in hp['values']:
                        new_hp = cur_combo.copy()
                        new_hp[name] = value
                        new_combo_list.append(new_hp)
            combo_list = new_combo_list
        return combo_list

    # Get all input node version settings
    # nv: list of Node
    def grid_search_nv(self, nodes):
        combo_list = [{}]
        for node in nodes:
            name = node.node_name
            new_combo_list = []
            for cur_combo in combo_list:
                for value in self.node_versions[name]:
                    new_nv = cur_combo.copy()
                    new_nv[name] = value
                    new_combo_list.append(new_nv)
            combo_list = new_combo_list
        return combo_list

    # Reach goal
    # def bayesian_search(self):