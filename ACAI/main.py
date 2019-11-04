from graph_constructor import GraphConstructor
from scheduler import Scheduler
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path of the configuration file")
    args = parser.parse_args()

    # Parse configuration file path
    conf_path = args.config
    name_index = conf_path.rfind('/')
    conf_name = conf_path[name_index+1:]
    workspace = conf_path[:name_index+1]

    # Begin workflow
    gc = GraphConstructor(workspace)
    graph = gc.load_graph()
    sc = Scheduler(graph)
    sc.run_workflow()
