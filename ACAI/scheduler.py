from log_manager import *
from threading import Thread
import time

class Node:
    def __init__(self):
        self.outputNodes = []
        self.inputNodes = []
        self.inputNodesNum = 0
        self.hyperParameters = {}

class Scheduler:
    # graph: list of Node
    def __init__(self, graph):
        self.nodeVersions = dict()
        self.graph = graph

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
        while exec_count < len(graph):
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
        for hp in hp_list:
            for input_nodes in input_nodes_versions:
                shouldRun = ExperimentRun(node.NodeName, node.ScriptVersion, hp, input_nodes)
                if not shouldRun:
                    continue
                runJob = Thread(target=submit_job, args=(node, hp, input_nodes))
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
    def submit_job(node, hp, input_nodes):
        # TODO: update this based on System API
        output_version = submit()
        SaveOutputData(node.NodeName, node.ScriptVersion, hp, input_nodes, output_version)

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