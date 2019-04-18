import numpy as np
import os


def data_model_word_embedding(train_p, verbose_mode):
    shuffle = False
    dimension = 300
    print(os.getcwd())
    word2vec_path = "word_embeddings/wordvec_d%d.txt" % dimension

    # Read pre-trained list
    with open(word2vec_path, "r") as lines:
        w2v = {line.split()[0]: np.array(list(map(float, line.split()[1:])))
               for line in lines}

    # Read from raw text
    label = []
    data = []
    with open(train_p, "r") as lines:
        for line in lines:
            segs = line.split("|||")
            label.append(segs[0])
            data.append(segs[1].split())
    data = np.array(data)
    label = np.array(label)

    # shuffle dataset
    if shuffle:
        data_label = np.c_[data.reshape(len(data), -1), label.reshape(len(label), -1)]
        np.random.shuffle(data_label)
        data = data_label[:, :data.size//len(data)].reshape(data.shape)
        label = data_label[:, data.size//len(data):].reshape(label.shape)

    # Generate vectors
    data = np.array([
        np.mean([w2v[w] for w in words if w in w2v]
                or [np.zeros(dimension)], axis=0)
        for words in data
    ])
    return data, label


def main():
    train_p = "../topicclass/topicclass_train_small.txt"
    data_model_word_embedding(train_p, True)


if __name__ == "__main__":
    main()
