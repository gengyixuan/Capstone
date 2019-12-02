import glog as log
from boom.modules import Module
from boom.log import set_logger


class Append(Module):

    def __init__(self, module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs):
        super(Append, self).__init__(module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs)

        # Initialize logger.
        set_logger(rabbitmq_host, exp_name)

    def process(self, job, data):

        log.debug(job)
        hps = job.params
        log.debug(data)

        return [(x,hps['num']) for x in data]
