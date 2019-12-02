import glog as log
from boom.modules import Module
from boom.log import set_logger


class Reader(Module):

    def __init__(self, module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs):
        super(Reader, self).__init__(module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs)

        # Initialize logger.
        set_logger(rabbitmq_host, exp_name)

    def process(self, job, data):

        log.debug(job)
        hps = job.params
        inputs1 = []
        with open(data['data11']) as f:
            for line in f:
                if line[:-1]:
                    inputs1.append(float(line[:-1]))

        inputs2 = []
        with open(data['data12']) as f:
            for line in f:
                if line[:-1]:
                    inputs2.append(float(line[:-1]))

        inputs3 = []
        with open(data['data21']) as f:
            for line in f:
                if line[:-1]:
                    inputs3.append(float(line[:-1]))

        inputs4 = []
        with open(data['data22']) as f:
            for line in f:
                if line[:-1]:
                    inputs4.append(float(line[:-1]))

        ret = []
        for x in inputs1 + inputs2 + inputs3 + inputs4:
            x = x * hps['mult'] + hps['add']
            ret.append(x)

        log.debug(ret)
        return ret
