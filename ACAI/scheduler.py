import time
import os
import sys
from threading import Thread
from zipfile import ZipFile
from termcolor import colored

from log_manager import LogManager
from constants import *
from searcher import Searcher
import acaisdk
import random


class Scheduler:
    # graph: list of Node
    def __init__(self, graph, workspace, search_method='grid', mock=None):
        self.node_versions = dict()
        self.graph = graph
        self.log_manager = LogManager()
        self.workspace = workspace
        self.mock = mock
        self.local = not not mock
        self.search_method = search_method
        self.searcher = None
        for node in graph:
            self.node_versions[node.node_name] = []

    # Build script files for each user-provided function
    # by adding input and output procedure
    def build_scripts(self, node):
        node_name = node.node_name
        script_path = node.script_path[:-3]
        script_name = script_path.split('/')[-1]
        local_generated_script_name = "{}_{}.py".format(self.workspace, node_name)
        cloud_generated_script_name = "_{}.py".format(node_name)
        fs = open(local_generated_script_name, "w")

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
        fs.write("if not os.path.exists('{}_output'):\n".format(node_name))
        fs.write("\tos.mkdir('{}_output')\n".format(node_name))
        fs.write("pkl.dump(rst, open('{}_output/{}.pkl', 'wb'))\n".format(node_name, node_name))
        # Compress the script and submit to ACAI system
        fs.close()

        if self.local:
            return
        # zip generated script and upload
        local_zip_filename = "{}_{}.zip".format(self.workspace, node_name)
        cloud_zip_filename = "_{}.zip".format(node_name)
        print(local_generated_script_name, cloud_generated_script_name, local_zip_filename, cloud_zip_filename)
        with ZipFile(local_zip_filename, "w") as zipf:
            zipf.write(local_generated_script_name, cloud_generated_script_name)
        acaisdk.file.File.upload([(local_zip_filename, cloud_zip_filename)])

    def run_workflow(self):
        print("Workflow start")
        if self.search_method == 'grid':
            self.run_workflow_grid()
        else:
            self.searcher = Searcher(self.graph, self.search_method)
            self.run_workflow_optim()
        # Delete temporary scripts
        for node in self.graph:
            temp_script = "{}_{}.py".format(self.workspace, node.node_name)
            if os.path.exists(temp_script):
                os.remove(temp_script)

    def run_workflow_grid(self):
        # Execute nodes with zero in-degree
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
            run_node = Thread(target=self.submit_node, args=(q.pop(0), q))
            run_node.start()
            exec_count += 1
        run_node.join()

    def run_workflow_optim(self):
        last_rst = None
        hps = None
        best_rst = None
        no_improve_count = 0
        max_no_improve_count = 10
        count = 1
        while no_improve_count < max_no_improve_count:
            print(colored("Starting round {} of hyper parameter search...".format(count), 'blue'))
            hps = self.searcher.get_next_hps(hps, last_rst)
            # Reset in-degree for each node
            for node in self.graph:
                node.input_nodes_num = len(node.input_nodes)
                for pre in node.input_nodes:
                    pre.output_nodes.append(node)
            # Execute nodes with zero indegree
            q = []
            for node in self.graph:
                if node.input_nodes_num == 0:
                    q.append(node)
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
                run_node = Thread(target=self.submit_node_optim, args=(q.pop(0), q, hps))
                run_node.start()
                exec_count += 1
            run_node.join()
            # TODO: get results
            last_rst = self.get_result()
            if not best_rst or self.compare(last_rst, best_rst):
                best_rst = last_rst
                no_improve_count = 0
            else:
                no_improve_count += 1

    # Submit all jobs for target node to ACAI System
    # node: target Node
    # q: Queue of Nodes
    def submit_node(self, node, q):
        print(colored("Node {} is ready to run. Starting...".format(node.node_name), 'blue'))
        # Go through hyper parameters and input node versions
        hp_list = self.grid_search_hp(node.hyper_parameter)
        input_nodes_versions = self.grid_search_nv(node.input_nodes)
        #jobs = []
        submitted = False
        total = len(hp_list) * len(input_nodes_versions)
        if total == 0:
            print(colored("No jobs to explore for Node {}. "
                          "Something is wrong with the hyper-parameter"
                          " setting or previous nodes.".format(node.node_name), 'red'))
        cur = 1
        jobs = []
        MAX_JOBS_PARALLEL = 24
        finish_count = 0
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
                # print(colored("Starting the {}/{} job for node {}".format(cur, total, node.node_name), 'blue'))
                if not submitted:
                    self.build_scripts(node)
                    submitted = True
                run_job = Thread(target=self.submit_job, args=(node, hp, input_nodes))
                run_job.start()
                jobs.append(run_job)
                cur += 1
                if len(jobs) >= MAX_JOBS_PARALLEL:
                    # wait for all jobs to finish before continuing
                    for j in jobs:
                        j.join()
                        finish_count += 1
                        print(colored("+++ Finished: "+str(finish_count), 'blue'))
                    jobs = []
        for j in jobs:
            j.join()
            finish_count += 1
            print(colored("+++ Finished: "+str(finish_count), 'blue'))
        
            # print(colored("Finished {}/{} job for node {}".format(finish_count, total, node.node_name), 'blue'))
        print(colored("All jobs for node {} finished!".format(node.node_name), 'blue'))
        # After this node is finished, check its descendants
        # for executable nodes (nodes with 0 indegree)
        for out in node.output_nodes:
            out.input_nodes_num -= 1
            if out.input_nodes_num == 0:
                q.append(out)

    # Submit one job specified by the hyper parameter setting to ACAI system
    # node: target Node
    # q: Queue of Nodes
    # hps: hyper parameter setting
    def submit_node_optim(self, node, q, hps):
        input_nodes_ver = {}
        for pre in node.input_nodes:
            input_nodes_ver[pre.node_name] = pre.last_ver
        self.submit_job(node, hps[node.node_name], input_nodes_ver)
        # After this node is finished, check its descendants
        # for executable nodes (nodes with 0 in-degree)
        for out in node.output_nodes:
            out.input_nodes_num -= 1
            if out.input_nodes_num == 0:
                q.append(out)

    # Submit a job to ACAI System
    # node: target Node
    # hp: hyper parameter setting for this job
    # input_nodes: input node versions for this job
    def submit_job(self, node, hp, input_nodes):
        name = node.node_name
        # Build command
        command = "python3 _{}.py ".format(name)
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
            output_fileset = self.mock.run_job("_{}.py".format(name), fileset_list, file_list, name, command)
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
            job_stat_wait_time = 0 #10+int(random.random()*60)
            status = job.with_attributes(attr).register().run().wait(first_sleep=job_stat_wait_time, subsequent_sleeps=job_stat_wait_time)
            if status != acaisdk.job.JobStatus.FINISHED:
                print(colored("A job for node {} failed!".format(name), "red"))
                return
            output_fileset = job.output_file_set
        output_version = output_fileset.split(':')[-1]
        self.add_node_version(node, output_version)
        self.log_manager.save_output_data(node.node_name, node.script_version, hp, input_nodes, output_version)

    def add_node_version(self, node, version):
        name = node.node_name
        node.last_ver = version
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
                    count = int((hp['end'] - hp['start']) / hp['step_size'] + 1)
                    value = hp['start']
                    while count > 0:
                        new_hp = cur_combo.copy()
                        new_hp[name] = value
                        new_combo_list.append(new_hp)
                        value += hp['step_size']
                        count -= 1
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