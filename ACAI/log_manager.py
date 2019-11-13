import copy
import pickle
from utils import Node
from constants import *

class LogManager:
    def __init__(self, reverse_log_path=REVERSE_LOG_PATH, fileset_hp_path=FILESET_HP_PATH, log_path=LOG_PATH):
        self.separator = "\t#\t"
        
        # Key: <str> NodeName + "\t#\t" + FileSetVersion
        # val: [(InputNodeName, FileSetVersion)]
        self.reverse_log = {}
        try:
            with open(reverse_log_path, 'rb') as input_file:
                self.reverse_log = pickle.load(input_file)
        except:
            pass

        # key: <str> NodeName + "\t#\t" + FileSetVersion
        # val: <dict> hyper_parameter
        self.fileset_hp = {}
        try:
            with open(fileset_hp_path, 'rb') as input_file:
                self.fileset_hp = pickle.load(input_file)
        except:
            pass

        # key: <str> Node_name + "\t#\t" + ScriptVersion + "\t#\t" + Inputs
        # value: list of (hyper_parameter(dict), OutputFileSetVersion)
        self.log = {}
        try:
            with open(log_path, 'rb') as input_file:
                self.log = pickle.load(input_file)
        except:
            pass

        self.reverse_log_path = reverse_log_path
        self.fileset_hp_path = fileset_hp_path
        self.log_path = log_path

        
    def convert_inputs_to_str(self, inputs):
        convert_str = ""
        for input_node_name in sorted(inputs):
            fileset_version = inputs[input_node_name]
            convert_str += str(input_node_name) + self.separator + str(fileset_version) + self.separator
        convert_str = convert_str[0: len(convert_str)-len(self.separator)]
        return convert_str


    def generate_node_key(self, node_name, script_version, inputs):
        return str(node_name) + self.separator + str(script_version) + self.separator + self.convert_inputs_to_str(inputs)


    def dfs_visit(self, node_name, fileset_version, cur_path, paths):
        tmp_key = node_name + self.separator + str(fileset_version)
        
        if tmp_key not in self.reverse_log:
            paths.append(copy.deepcopy(cur_path))
            return
        
        inputs = self.reverse_log[tmp_key]

        for one_tuple in inputs:
            input_node = one_tuple[0]
            input_fs_version = one_tuple[1]

            new_path = copy.deepcopy(cur_path)
            new_path[input_node] = input_fs_version
            # new_path.add(input_node)
            self.dfs_visit(input_node, input_fs_version, new_path, paths)


    def tracking_ancestors(self, inputs):
        all_paths = []
        
        for one_tuple in inputs:
            input_node = one_tuple[0]
            input_fs_version = one_tuple[1]

            cur_path = {input_node: input_fs_version}
            self.dfs_visit(input_node, input_fs_version, cur_path, all_paths)

        return all_paths


    # return true if valid
    # return false if there is invalid ancestors
    def check_ancestors_hp(self, paths):
        for i in range(0, len(paths)):
            for j in range(i+1, len(paths)):
                path1 = paths[i]
                path2 = paths[j]

                for one_node in path1:
                    if one_node in path2:
                        fs_version1 = path1[one_node]
                        fs_version2 = path2[one_node]

                        if fs_version1 == fs_version2:
                            continue

                        # if different file_version
                        # check if hp has the same value
                        hp1 = self.fileset_hp[one_node + self.separator + str(fs_version1)]
                        hp2 = self.fileset_hp[one_node + self.separator + str(fs_version2)]

                        if len(hp1) != len(hp2):
                            return False

                        for one_hp in hp1:
                            if hp1[one_hp] != hp2[one_hp]:
                                return False
        
        return True


    def experiment_run(self, node_name, script_version, hyper_parameter, inputs):
        check_log = self.generate_node_key(node_name, script_version, inputs)
        
        if check_log in self.log:
            for hp_dict, out_fileset_V in self.log[check_log]:
                # check if hp_dict all match with hyper_parameters
                all_match = True

                for one_hp in hyper_parameter:
                    if one_hp not in hp_dict:
                        all_match = False
                        break

                    if hp_dict[one_hp] != hyper_parameter[one_hp]:
                        all_match = False
                        break
                
                if all_match:
                    return False, out_fileset_V

        all_paths = self.tracking_ancestors(inputs)
        return self.check_ancestors_hp(all_paths), None


    def save_output_data(self, node_name, script_version, hyper_parameter, inputs, output_fileset_version):
        log_key = self.generate_node_key(node_name, script_version, inputs)
        self.log[log_key].append( (hyper_parameter, output_fileset_version) )
        
        reverse_log_key = node_name + self.separator + str(output_fileset_version)
        self.reverse_log[reverse_log_key] = inputs
        self.fileset_hp[reverse_log_key] = hyper_parameter

        with open(self.reverse_log_path, 'wb') as outfile:
            pickle.dump(self.reverse_log, outfile, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.fileset_hp_path, 'wb') as outfile:
            pickle.dump(self.fileset_hp, outfile, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.log_path, 'wb') as outfile:
            pickle.dump(self.log, outfile, protocol=pickle.HIGHEST_PROTOCOL)


# for testing
if __name__ == "__main__":
    lm = LogManager(reverse_log_path, fileset_hp_path, log_path)
    
    inputs = [("b.py", 1), ("a.py", 2), ("c.py", 3)]
    # # print(lm.convert_inputs_to_str(inputs))

    hp_test = {'x1': 1.2, 'x2': 0.5}
    # lm.save_output_data("e.py", 2, hp_test, inputs, 2)

    # for key in lm.log:
    #     print(key)
    #     print(lm.log[key])

    lm.reverse_log["b" + lm.separator + "1"] = [("a", 1)]
    lm.reverse_log["d" + lm.separator + "1"] = [("b", 1), ("c", 3)]
    
    lm.reverse_log["g" + lm.separator + "2"] = [("e", 2), ("f", 1)]
    lm.reverse_log["e" + lm.separator + "2"] = [("h", 1)]
    lm.reverse_log["f" + lm.separator + "1"] = [("h", 1)]

    lm.reverse_log["g" + lm.separator + "1"] = [("e", 1), ("f", 2)]
    lm.reverse_log["e" + lm.separator + "1"] = [("h", 1)]
    lm.reverse_log["f" + lm.separator + "2"] = [("h", 2)]

    inputs = [("d", 1), ("g", 1)]

    print(lm.tracking_ancestors(inputs))

    lm.save_output_data("a.py", 1, hp_test, inputs, 1)