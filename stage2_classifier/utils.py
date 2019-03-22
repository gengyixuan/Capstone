from pprint import pprint


def save_metrics(rst, metrics_path, metric_list, data_model_list, classifier_list):
    pprint(rst)
    for metric in metric_list:
        metrics_fullpath = "%s/%s.csv" % (metrics_path, metric)
        with open(metrics_fullpath, "w") as f_metrics:
            f_metrics.write("Data Model / Classifier")
            for classifier in classifier_list:
                f_metrics.write(", %s" % classifier)
            f_metrics.write("\n")

            for data_model in data_model_list:
                f_metrics.write(data_model)
                cur_rst = rst[data_model]
                for classifier in classifier_list:
                    f_metrics.write(", %.4f" % cur_rst[classifier][metric])
                f_metrics.write("\n")


def report_parser(report):
    line_list = report.split('\n')
    line = line_list[len(line_list)-2]
    res_list = line.split()
    precision = float(res_list[2])
    recall = float(res_list[3])
    f1 = float(res_list[4])
    return {"Precision": precision,
            "Recall": recall,
            "F1": f1}
