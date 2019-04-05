import os
import csv

import glog as log
from boom.modules import Module
from boom.log import set_logger
import json


class ResultCollector(Module):
    "The CSVWriter class saves results to a csv file."

    def __init__(self, module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs):
        super(ResultCollector, self).__init__(module_id, name, exp_name, rabbitmq_host,
                                        pipeline_conf, module_conf, **kwargs)

        self.results = []

    def process(self, job, data):

        self.results.append(data)

    
    def save_job_data(self, job, data):
        path = job.output_base + '/' + self.output_file
        log.info('Save csv to ' + path)
        if not os.path.exists(job.output_base):
            os.mkdir(job.output_base)
        
        with open(path, "w+") as f:
            for result in self.results:
                f.write(json.dumps(result) + "\n")
        
        return path


if __name__ == '__main__':
    pass
