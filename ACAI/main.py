from graph_constructor import GraphConstructor
from scheduler import Scheduler
from cloud_mock import Mock
from constants import *
import argparse
import time
from termcolor import colored

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path of the configuration file")
    parser.add_argument("-l", "--local", default=False, action="store_true", help="Run workflow in local")
    args = parser.parse_args()

    # Parse configuration file path
    conf_path = args.config
    name_index = conf_path.rfind('/')
    conf_name = conf_path[name_index+1:]
    workspace = conf_path[:name_index+1]

    print("workspace: "+workspace)

    # Begin workflow
    gc = GraphConstructor(workspace, args.local)
    graph = gc.load_graph()
    mock = None
    if args.local:
        mock = Mock(workspace, MOCK_PATH)
    sc = Scheduler(graph, workspace, 'grid', mock)
    sc.run_workflow()
    end = time.time()

    # Get running time
    end = time.time()
    print(colored("Total time: {}s".format(end - start), "green"))