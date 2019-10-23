import copy

# NodeName: string
# ScriptVersion: int, start from 0
# HyperParameter: dict, key: parameter, val: value

class Node:
    def __init__(self, NodeName, HyperParameter):
        self.NodeName = NodeName
        self.ScriptVersion = 0
        self.InputNodes = []
        # dict: key: ScriptVersion, val: HyperParameter
        self.HyperParameter = {ScriptVersion: HyperParameter}
        # dict, key: ScriptVersion, val: list of HyperParameter run before
        self.run = {0: []}  
    
    # AddNodeList: list of Node variable as the input nodes
    # This func should only be called during graph init
    def AddInputNode(self, AddNodeList):
        for AddNode in AddNodeList:
            self.InputNodes.append(AddNode)

    def UpdateScriptVersion(self):
        self.ScriptVersion += 1
        self.run = False
        return self.ScriptVersion

    def GetScriptVersion(self):
        return self.ScriptVersion

    def GetInputNodes(self):
        return self.InputNodes
    
    def GetHyperParameter(self):
        return self.HyperParameter

    # NewHyperParameter: key: parameter to be updated, val: new value
    def UpdateHyperParameter(self, ScriptVersion, NewHyperParameter):
        if ScriptVersion not in self.HyperParameter:
            self.HyperParameter[ScriptVersion] = NewHyperParameter
        else:
            for key in NewHyperParameter:
                if key in self.HyperParameter[ScriptVersion]:
                    self.HyperParameter[ScriptVersion][key] = NewHyperParameter[key]

    # run this node under current configuration
    def RunNode(self, ScriptVersion, HyperParameter):
        self.run[ScriptVersion].append(copy.deepcopy(HyperParameter))


# visit computational graph
# NodeName: Node
# CurPath: set()
# Paths: list of set
def dfs_visit(NodeName, CurPath, Paths):
    if len(NodeName.InputNodes) == 0:
        Paths.append(copy.deepcopy(CurPath))
        return
    
    for InputNode in NodeName.InputNodes:
        NewPath = copy.deepcopy(CurPath)
        NewPath.add(InputNode)
        dfs_visit(InputNode, NewPath, Paths)

    return

# Description: give the result based on the following two conditions:

# 1. Check if the same experiment has been run before 
# (matching all ScriptName, ScriptVersion, [(InputData, InputDataVersion)] and HyperParameter). 
# If yes, return false

# 2. Check all the ancestors of all the input data. Check if there is any parameter mismatch. 
# If yes, return false.
def ExperimentRun(NodeName, ScriptVersion, HyperParameter, Inputs):
    if ScriptVersion in NodeName.run:
        all_match = True

        for one_hp in NodeName.run[ScriptVersion]:
            for key in HyperParameter:
                if key not in one_hp or one_hp[key] != HyperParameter[key]:
                    all_match = False
                if not all_match:
                    break
            if not all_match:
                break
            
        if all_match:
            return False

    Paths = []
    CurPath = set()
    dfs_visit(NodeName, CurPath, Paths)

    for i in range(0, len(Paths)):
        for j in range(i+1, len(Paths)):
            path1 = Paths[i];
            path2 = Paths[j]

            common_nodes = []

            for one_node in path1:
                if one_node in path2:
                    common_nodes.append(one_node)

            



    
    