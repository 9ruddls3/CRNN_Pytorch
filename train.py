import argparse
import time

from model import Model
from torch.nn import CTCLoss
from torch.autograd import Variable

import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch


# use_VGG_extractor, dtype, loader_train, batch_size

my_model= Model(use_VGG_extractor=use_VGG_extractor).type(dtype)

loss_function = CTCLoss().type(dtype)

if use_VGG_extractor:
    opt_parameters=list(my_model.RNN.parameters())+list(my_model.toTraget.parameters())
    optimizer = optim.Adam(iter(opt_parameters), lr=learning_rate)
else:
    optimizer = optim.Adam(my_model.parameters(), lr=learning_rate)

scheduler = lrs.StepLR(optimizer, step_size=20, gamma=0.8)

def model_train(max_epoch, print_every):
    iter_each_epoch = num_train // batch_size
    loss_his_train = []

    for epoch in range(max_epoch):
        scheduler.step()
        my_model.train()
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
              'start epoch %d/%d:' % (epoch + 1, max_epoch), 'learning_rate =', scheduler.get_lr()[0],
              'sequence_len =', my_model.RNN.sql)
        tot_loss = 0

        it = 0
        for images, labels in loader_train:

            X_var = Variable(images.type(dtype))
            out_size = Variable(torch.IntTensor([sequence_len] * batch_size))
            y_size = Variable(torch.IntTensor([len(l) for l in labels]))
            conc_label = ''.join(labels)
            y = [chrToindex[c] for c in conc_label]
            y_var = Variable(torch.IntTensor(y))

            my_model.zero_grad()
            my_model.RNN.init_hidden(batch_size)

            scores = my_model(X_var)
            loss = loss_function(scores, y_var, out_size, y_size) / batch_size
            loss.backward()
            optimizer.step()

            tot_loss += loss.data

            if it == 0 or (it + 1) % print_every == 0 or it == iter_each_epoch - 1:
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                      'iter %d loss = %f' % (it + 1, loss.data))
            it += 1

        tot_loss /= iter_each_epoch
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
              'epoch %d/%d average_loss = %f\n' % (epoch + 1, max_epoch, tot_loss))
        loss_his_train.append(tot_loss)
    return loss_his_train

if __name__ == "__main__":
  my_model.apply(reset)
  my_model.train()
  my_model.RNN.init_hidden(batch_size)
  loss_his_train=model_train(max_epoch=500,print_every=25)