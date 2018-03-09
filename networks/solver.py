from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable
from networks.net_api.losses import CombinedLoss


class Solver(object):
    default_adam_args = {"lr": 1e-2,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0001}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=CombinedLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=5):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = 10
        # iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        curr_iter = 0
        for epoch in range(num_epochs):
            for i_batch, sample_batched in enumerate(train_loader):
                X = Variable(sample_batched[0])
                y = Variable(sample_batched[1])
                yb = Variable(sample_batched[2])
                w = Variable(sample_batched[3])

                if model.is_cuda:
                    X, y, yb, w = X.cuda(), y.cuda(), yb.cuda(), w.cuda()

                for iter in range(iter_per_epoch):
                    curr_iter += iter
                    optim.zero_grad()
                    output = model(X)
                    loss = self.loss_func(output, y, yb, w)
                    loss.backward()
                    optim.step()
                    if iter % log_nth == 0:
                        self.train_loss_history.append(loss.data[0])
                        print('[Iteration : ' + str(iter) + '/' + str(iter_per_epoch * num_epochs) + '] : ' + str(
                            loss.data[0]))

                # batch_output = torch.max(model(X), dim= 1)
                # train_accuracy = self.accuracy(batch_output[1], y)
                # self.train_acc_history.append(train_accuracy)
                #
                # val_output = torch.max(model(Variable(torch.from_numpy(val_loader.dataset.X))), dim= 1)
                # val_accuracy = self.accuracy(val_output[1], Variable(torch.from_numpy(val_loader.dataset.y)))
                # self.val_acc_history.append(val_accuracy)
            print('[Epoch : ' + str(epoch) + '/' + str(num_epochs) + '] : ' + str(loss.data[0]))
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        # print('FINISH.')

    def accuracy(self, y_pred, y_true):
        # print(y_pred)
        # print(y_true)
        return np.sum(y_pred.data[0] == y_true.data[0]) / len(y_pred)
