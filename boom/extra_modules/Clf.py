import glog as log
from boom.modules import Module
from boom.log import set_logger
from classifier import model_training


class Clf(Module):

    def __init__(self, module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs):
        super(Clf, self).__init__(module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs)

        # Initialize logger.
        set_logger(rabbitmq_host, exp_name)

    def process(self, job, data):

        log.debug(job)
        verbose_mode = True
        X = data['X']
        Y = data['Y']
        test_ratio = job.params['test_ratio']
        classifier = job.params['classifier']
        report = model_training(X, Y, test_ratio, verbose_mode, classifier)
        log.debug(data)

        return report
