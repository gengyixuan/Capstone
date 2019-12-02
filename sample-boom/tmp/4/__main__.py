import sys
import gflags
from boom.modules import Logger
if __name__ == '__main__':
    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)
    Logger(4, 'logger', '2019-11-12_22h59m24s', '127.0.0.1', {'mode': 'local', 'name': 'text_classifier', 'rabbitmq_host': '127.0.0.1', 'clean_up': False, 'use_mongodb': False, 'mongodb_host': '127.0.0.1'},{'name': 'logger', 'script': 'src/writer.py', 'type': 'Logger', 'output_file': './results-classifier.csv', 'instances': 1, 'params': [{'name': 'subtract', 'type': 'float', 'start': -1.0, 'end': 0.0, 'step_size': 2.0}, {'name': 'classifier', 'type': 'collection', 'values': ['SVM', 'NaiveBayes', 'MLP', 'DT', 'KNN', 'RandomForest', 'Adaboost', 'GradientBoost']}]}).run()
