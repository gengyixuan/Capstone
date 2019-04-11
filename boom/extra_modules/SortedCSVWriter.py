import os
import csv

import glog as log

from boom.modules import Module


class SortedCSVWriter(Module):
    "The CSVWriter class saves results to a csv file."

    def __init__(self, module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs):
        super(SortedCSVWriter, self).__init__(module_id, name, exp_name, rabbitmq_host,
                                        pipeline_conf, module_conf, **kwargs)

        self.content = []
        self.header = []
        self.key_col_vals = []

    def process(self, job, data):
        # Create header when first called.
        key_col = job.params['key_column']
        if self.header == []:
            self.header = [k for k in data]
        self.content.append([job.output_path] + [data[k] for k in self.header])
        self.key_col_vals.append(data[key_col])
        return data

    ## Save csv file, overriding the default saving function.
    #  @param job The job to be saved.
    #  @param data The data to be saved.
    #  @return the path to data.
    def save_job_data(self, job, data):
        # Get sorted order first
        indices = [i[0] for i in sorted(enumerate(self.key_col_vals), key=lambda x:x[1], reverse=True)]
        path = job.output_base + '/' + self.output_file
        log.info('Save csv to ' + path)

        if not os.path.exists(job.output_base):
            os.mkdir(job.output_base)

        with open(path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['configuration'] + self.header)
            for ind in indices:
                writer.writerow(self.content[ind])
        return path


if __name__ == '__main__':
    pass
