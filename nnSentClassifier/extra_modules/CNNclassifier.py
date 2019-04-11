import glog as log
from boom.modules import Module
from boom.log import set_logger

from collections import defaultdict
import time
import random
import torch
import pickle


class CNNclass(torch.nn.Module):
    def __init__(self, nwords, emb_size, num_filters, window_size, ntags, isRegress=False):
        super(CNNclass, self).__init__()

        """ layers """
        self.embedding = torch.nn.Embedding(nwords, emb_size)
        # uniform initialization
        torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        # Conv 1d
        self.conv_1d = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=window_size,
                                       stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.relu = torch.nn.ReLU()
        self.classifier_layer = torch.nn.Linear(in_features=num_filters, out_features=ntags, bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.classifier_layer.weight)

        self.regressor_layer = torch.nn.Linear(in_features=num_filters, out_features=1, bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.regressor_layer.weight)

        self.isRegress = isRegress

    def forward(self, words):
        emb = self.embedding(words)                 # nwords x emb_size
        emb = emb.unsqueeze(0).permute(0, 2, 1)     # 1 x emb_size x nwords
        h = self.conv_1d(emb)                       # 1 x num_filters x nwords
        # Do max pooling
        h = h.max(dim=2)[0]                         # 1 x num_filters
        h = self.relu(h)
        if self.isRegress:
            out = self.regressor_layer(h)
            return out
        else:
            out = self.classifier_layer(h)              # size(out) = 1 x ntags
            return out



class CNNclassifier(Module):

    def __init__(self, module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs):
        super(CNNclassifier, self).__init__(module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs)

        # Initialize logger.
        set_logger(rabbitmq_host, exp_name)

    def process(self, job, data):

        log.debug(job)
        
        train = data['train']
        w2i = data['w2i']
        t2i = data['t2i']
        dev = data['dev']
        nwords = data['nwords']
        ntags = data['ntags']

        # Define the model
        EMB_SIZE = job.params['EMB_SIZE']#64
        WIN_SIZE = job.params['WIN_SIZE']#3
        IS_REGRESS = False if job.params['IS_REGRESS'] == 0 else True
        FILTER_SIZE = 64

        # initialize the model
        model = CNNclass(nwords, EMB_SIZE, FILTER_SIZE, WIN_SIZE, ntags, IS_REGRESS)
        criterion = torch.nn.CrossEntropyLoss()
        if IS_REGRESS:
            criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        use_cuda = torch.cuda.is_available()
        type = torch.cuda.LongTensor if use_cuda else torch.LongTensor
        typeFloat = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        if use_cuda:
            model.cuda()

        # training !!
        iter_results = []
        for ITER in range(0,3):
            # Perform training
            random.shuffle(train)
            train_loss = 0.0
            train_correct = 0.0
            start = time.time()
            sampleID = 0
            for words, tag, lentag in train:
                sampleID += 1
                # Padding (can be done in the conv layer as well)
                if len(words) < WIN_SIZE:
                    words += [0] * (WIN_SIZE - len(words))
                words_tensor = torch.tensor(words).type(type)
                
                if IS_REGRESS:
                    lentag_tensor = torch.tensor([lentag]).type(typeFloat)
                    predict = model(words_tensor)
                    train_correct += (predict.item() - lentag) * (predict.item() - lentag)
                    
                    my_loss = criterion(predict, lentag_tensor)
                    train_loss += my_loss.item()
                    # Do back-prop
                    optimizer.zero_grad()
                    my_loss.backward()
                    optimizer.step()
                
                else:
                    tag_tensor = torch.tensor([tag]).type(type)
                    scores = model(words_tensor)
                    predict = scores[0].argmax().item()
                    if predict == tag:
                        train_correct += 1

                    my_loss = criterion(scores, tag_tensor)
                    train_loss += my_loss.item()
                    # Do back-prop
                    optimizer.zero_grad()
                    my_loss.backward()
                    optimizer.step()
            
            if ITER % 1 == 0:
                save_path = "basicCNNiter" + str(ITER) + ".stdict"
                torch.save(model.state_dict(), save_path)

            # Perform testing
            test_correct = 0.0
            for words, tag, lentag in dev:
                # Padding (can be done in the conv layer as well)
                if len(words) < WIN_SIZE:
                    words += [0] * (WIN_SIZE - len(words))
                words_tensor = torch.tensor(words).type(type)
                
                if IS_REGRESS:
                    predict = model(words_tensor)
                    test_correct += (predict.item() - lentag) * (predict.item() - lentag)
                else:
                    scores = model(words_tensor)[0]
                    predict = scores.argmax().item()
                    if predict == tag:
                        test_correct += 1

            # logging test accuracy
            test_acc = test_correct / len(dev)
            result_dict = {"iter": ITER, "test acc": test_acc}
            if IS_REGRESS:
                result_dict = {"iter": ITER, "test err": test_acc}
            iter_results.append(result_dict)
            log.debug(result_dict)

        return {"network": "CNN classifier", "config": job.params, "results": iter_results}