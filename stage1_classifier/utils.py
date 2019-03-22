
def print_metrics(rst, metrics_path):
    with open(metrics_path, "w+") as f_metrics:
        f_metrics.write("Model, Accuracy, Recall, F1\n")
        for model in rst:
            f_metrics.write("%s, %.4f, %.4f, %.4f\n" %
                            (model,
                             rst[model][0],
                             rst[model][1],
                             rst[model][2]))
