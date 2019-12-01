import sys
import gflags
import boom
if __name__ == '__main__':
    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)
    p = boom.Pipeline('{"pipeline": {"mode": "local", "name": "text_classifier", "rabbitmq_host": "127.0.0.1", "clean_up": false, "use_mongodb": false, "mongodb_host": "127.0.0.1"}, "modules": [{"name": "reader", "type": "Reader", "input_file": "data.json", "output_module": "append", "instances": 1, "processes": 1, "params": [{"name": "mult", "type": "float", "start": 0.2, "end": 0.6, "step_size": 0.2}, {"name": "add", "type": "float", "start": 1.0, "end": 2.0, "step_size": 0.5}]}, {"name": "append", "type": "Append", "output_module": "writer", "instances": 1, "params": [{"name": "num", "type": "collection", "values": [1000, 2000]}]}, {"name": "writer", "script": "src/writer.py", "type": "Writer", "output_file": "./results-classifier.csv", "instances": 1, "params": [{"name": "subtract", "type": "float", "start": -1.0, "end": 0.0, "step_size": 2.0}, {"name": "classifier", "type": "collection", "values": ["SVM", "NaiveBayes", "MLP", "DT", "KNN", "RandomForest", "Adaboost", "GradientBoost"]}]}]}', '2019-11-12_22h59m24s')
    p.run()