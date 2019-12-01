import time
import os
import pickle as pkl
import numpy as np
from threading import Thread
from zipfile import ZipFile
from termcolor import colored
from matplotlib import pyplot as plt

from log_manager import LogManager
from constants import *
from searcher import Searcher

from acaisdk.file import File
from acaisdk.fileset import FileSet
from acaisdk.job import Job, JobStatus


class Scheduler:
    # graph: list of Node
    def __init__(self, graph, workspace, optim_info, mock=None):
        self.node_versions = dict()
        self.graph = graph
        self.log_manager = LogManager()
        self.workspace = workspace
        self.mock = mock
        self.local = not not mock
        self.optim_info = optim_info
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

        if os.path.exists(local_generated_script_name):
            return
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
        File.upload([(local_zip_filename, cloud_zip_filename)])

    def run_workflow(self):
        print("Workflow start")
        if self.optim_info['search'] == 'grid':
            self.run_workflow_grid()
        else:
            self.searcher = Searcher(self.graph, self.optim_info['search'])
            self.run_workflow_optim()
        # Persist local mock logs
        if self.mock:
            self.mock.persist_to_disk()
        # Delete temporary scripts
        for node in self.graph:
            temp_script = "{}_{}.py".format(self.workspace, node.node_name)
            if os.path.exists(temp_script):
                os.remove(temp_script)
        # Retrieve results
        self.retrieve_result(self.optim_info['result_node'])

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
        best_rsts = []
        no_improve_count = 0
        max_no_improve_count = 10
        count = 1
        # Run workflow
        while no_improve_count < max_no_improve_count:
            print(colored("Starting round {} of hyper parameter search...".format(count), 'blue'))
            hps = self.searcher.get_next_hps(hps, last_rst)
            # Reset in-degree for each node
            for node in self.graph:
                node.input_nodes_num = len(node.input_nodes)
            # Execute nodes with zero indegree
            q = []
            for node in self.graph:
                if node.input_nodes_num == 0:
                    q.append(node)
            # Count the number of executed nodes
            exec_count = 0
            # Keep looping until all nodes are executed
            while exec_count < len(self.graph):
                # Constantly check if new nodes are added to the queue
                while not q:
                    time.sleep(SLEEP_INTERVAL)
                # Submit current node for execution in a new thread
                run_node = Thread(target=self.submit_node_optim, args=(q.pop(0), q, hps))
                run_node.start()
                exec_count += 1
            run_node.join()
            # Download latest result
            result_node = self.optim_info['result_node']
            result_node_name = result_node.node_name
            if self.local:
                result_path = "{}/{}:{}/{}_output/{}.pkl".format(
                    MOCK_PATH, result_node_name, result_node.last_ver, result_node_name, result_node_name)
            else:
                result_path = "tmp_{}.pkl".format(result_node_name)
                File.download({"{}_output/{}.pkl".format(result_node_name, result_node_name)
                               : result_path})
            # Get target metric value
            result = pkl.load(open(result_path, "rb"))
            last_rst = result[self.optim_info['metric']]
            assert isinstance(last_rst, (int, float))
            if self.optim_info['direction'] == 'min':
                last_rst = -last_rst
            if not self.local and os.path.exists(result_path):
                os.remove(result_path)
            # Update best result
            if not best_rst or last_rst > best_rst:
                best_rst = last_rst
                no_improve_count = 0
                print(colored("New best result! {}:{}".format(self.optim_info['metric'], best_rst), 'blue'))
            else:
                no_improve_count += 1
                print(colored("No improvement in {} continuous searches".format(no_improve_count), 'blue'))
            count += 1
            best_rsts.append(best_rst)
        # Draw performance curve
        output_path = self.workspace + "/" + OUTPUT_PATH
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        xind = np.arange(1, len(best_rsts) + 1)
        plt.plot(xind, best_rsts, 'b', marker='^')
        plt.title('Performance curve with search iterations')
        plt.xlabel('# of search iterations')
        plt.ylabel(self.optim_info['metric'])
        plt.savefig('{}/{}_curve.pdf'.format(output_path, self.optim_info['search']))

    # Submit all jobs for target node to ACAI System
    # node: target Node
    # q: Queue of Nodes
    def submit_node(self, node, q):
        print(colored("Start Node {} ...".format(node.node_name), 'blue'))
        # Go through hyper parameters and input node versions
        hp_list = self.grid_search_hp(node.hyper_parameter)
        input_nodes_versions = self.grid_search_nv(node.input_nodes)
        jobs = []
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
                should_run, version = self.log_manager.experiment_run(node.node_name, node.script_version, hp, input_nodes)
                if not should_run:
                    if version:
                        self.add_node_version(node, version)
                        print(colored("Skip the {}/{} job for node {}: Already run before".format(cur, total, node.node_name), 'blue'))
                    else:
                        print(colored("Skip the {}/{} job for node {}: Bad common ancestor".format(cur, total, node.node_name), 'blue'))
                    cur += 1
                    continue
                # print(colored("Starting the {}/{} job for node {}".format(cur, total, node.node_name), 'blue'))
                self.build_scripts(node)
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
        print(colored("Start Node {} ...".format(node.node_name), 'blue'))
        # Get the latest version from input nodes
        input_nodes_ver = {}
        for pre in node.input_nodes:
            input_nodes_ver[pre.node_name] = pre.last_ver
        # Check if it's been run before
        should_run, version = self.log_manager.experiment_run(
            node.node_name, node.script_version, hps[node.node_name], input_nodes_ver)
        # No need to run
        if not should_run:
            self.add_node_version(node, version)
        else:
            # Build scripts and run
            self.build_scripts(node)
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
            input_file_set = FileSet.create_file_set("{}_input".format(name), fileset_list)
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
            job = Job()
            job_stat_wait_time = 0 #10+int(random.random()*60)
            status = job.with_attributes(attr).register().run().wait(first_sleep=job_stat_wait_time, subsequent_sleeps=job_stat_wait_time)
            if status != JobStatus.FINISHED:
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

    # Retrieve results from all the jobs for target node
    # node: target node (must be a result node)
    def retrieve_result(self, node):
        # Make tmp dir for downloaded result
        tmp_dir = "result_tmp"
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        # Download results
        node_name = node.node_name
        results = {}
        for ver in self.node_versions[node_name]:
            if self.local:
                local_path = "{}/{}:{}/{}_output/{}.pkl".format(
                    MOCK_PATH, node_name, ver, node_name, node_name)
            else:
                remote_path = "{}_output/{}.pkl:{}".format(node_name, node_name, ver)
                local_path = "{}/{}.pkl".format(tmp_dir, node_name)
                File.download({remote_path: local_path})
            rst = pkl.load(open(local_path, "rb"))
            for metric in rst:
                if not isinstance(rst[metric], (int, float, list, dict)):
                    rst.pop(metric, None)
            results[ver] = rst
        self.log_manager.save_result(node_name, results, self.workspace)
        if os.path.exists(tmp_dir):
            import shutil
            shutil.rmtree(tmp_dir)
