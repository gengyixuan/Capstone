from sklearn import datasets
import numpy as np


def ranking(data_path, verbose_mode):
    data = datasets.load_svmlight_file(data_path)
    return data[0].toarray(), np.array(data[1])


if __name__ == "__main__":
    ranking("../pre_processed_data/ranking1000.txt", True)