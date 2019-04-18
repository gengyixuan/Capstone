import glog as log
from boom.modules import Module
from boom.log import set_logger
from extra_modules.data_model_list import data_model_list


class DataModel(Module):

    def __init__(self, module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs):
        super(DataModel, self).__init__(module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs)

        # Initialize logger.
        set_logger(rabbitmq_host, exp_name)

    def process(self, job, data):

        log.debug(job)
        verbose_mode = True
        raw_text_path = data['path']
        target_data_model = data_model_list[job.params['data_model']]
        X, Y = target_data_model(raw_text_path, verbose_mode)
        data['X'] = X.tolist()
        data['Y'] = Y.tolist()
        log.debug(data)

        return data
