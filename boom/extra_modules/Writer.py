import glog as log
from boom.modules import Module
from boom.log import set_logger
from classifier import model_training


class Writer(Module):

    def __init__(self, module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs):
        super(Writer, self).__init__(module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs)

        # Initialize logger.
        set_logger(rabbitmq_host, exp_name)

    def process(self, job, data):

        log.debug(job)
        hps = job.params
        log.debug(data)
        ret = []
        for line in data:
            x, ap = line
            ret.append((x - hps['subtract'], ap, hps['classifier']))
        return ret
