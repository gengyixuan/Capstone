import sys
import gflags
from boom.modules import *
from extra_modules.Reader import Reader
if __name__ == '__main__':
    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)
    Reader(1, 'reader', '2019-11-12_22h59m24s', '127.0.0.1', {'mode': 'local', 'name': 'text_classifier', 'rabbitmq_host': '127.0.0.1', 'clean_up': False, 'use_mongodb': False, 'mongodb_host': '127.0.0.1'},{'name': 'reader', 'type': 'Reader', 'input_file': 'data.json', 'output_module': 'append', 'instances': 1, 'processes': 1, 'params': [{'name': 'mult', 'type': 'float', 'start': 0.2, 'end': 0.6, 'step_size': 0.2}, {'name': 'add', 'type': 'float', 'start': 1.0, 'end': 2.0, 'step_size': 0.5}]}).run()
