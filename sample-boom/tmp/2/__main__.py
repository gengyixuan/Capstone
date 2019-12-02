import sys
import gflags
from boom.modules import *
from extra_modules.Append import Append
if __name__ == '__main__':
    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)
    Append(2, 'append', '2019-11-12_22h59m24s', '127.0.0.1', {'mode': 'local', 'name': 'text_classifier', 'rabbitmq_host': '127.0.0.1', 'clean_up': False, 'use_mongodb': False, 'mongodb_host': '127.0.0.1'},{'name': 'append', 'type': 'Append', 'output_module': 'writer', 'instances': 1, 'params': [{'name': 'num', 'type': 'collection', 'values': [1000, 2000]}]}).run()
