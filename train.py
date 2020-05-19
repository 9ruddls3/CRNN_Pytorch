import os
import argparse
import time

from parms import batch_size, use_VGG_extractor, sequence_len, chrToindex, learning_rate

from torch.nn import CTCLoss
from torch.autograd import Variable

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from model.CRNN import CRNN_model

def reset(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):

            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal(m.weight, gain=1)
            m.bias.data.zero_()
        elif hasattr(m, 'reset_parameters'):
            m.reset_parameters()

def model_train(train_model,dataloader, max_epoch, print_every, b_size=batch_size):
    iter_each_epoch = len(dataloader.dataset) // b_size
    loss_his_train = []

    for epoch in range(max_epoch):
        scheduler.step()
        train_model.train()
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
              'start epoch %d/%d:' % (epoch + 1, max_epoch), 'learning_rate =', scheduler.get_lr()[0],
              'sequence_len =', train_model.RNN.sql)
        tot_loss = 0

        it = 0
        for images, labels in dataloader:
            X_var = Variable(images.type(dtype))
            out_size = Variable(torch.IntTensor([sequence_len] * batch_size))
            y_size = Variable(torch.IntTensor([len(l) for l in labels]))
            conc_label = ''.join(labels)
            y = [chrToindex[c] for c in conc_label]
            y_var = Variable(torch.IntTensor(y))

            train_model.zero_grad()
            train_model.RNN.init_hidden(b_size)

            scores = train_model(X_var)
            loss = loss_function(scores, y_var, out_size, y_size) / b_size
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--Dataloader', type=str, required=True, help='path to folder which contains the images')
    parser.add_argument('--Model_path', type=str, required=True, help='path to save trained Model weight file')
    
    args = parser.parse_args()
    Dataloader_path = os.path.abspath(args.Dataloader)
    Model_path = os.path.abspath(args.Model_path)
    
    dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    dataLoader = torch.load(Dataloader_path)
    
    model= CRNN_model(use_VGG_extractor=use_VGG_extractor).type(dtype)
    
    loss_function = CTCLoss().type(dtype)

    if use_VGG_extractor:
        opt_parameters=list(model.RNN.parameters())+list(model.toTraget.parameters())
        optimizer = optim.Adam(iter(opt_parameters), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = lrs.StepLR(optimizer, step_size=20, gamma=0.8)

    model.apply(reset)
    model.train()
    model.RNN.init_hidden(batch_size)
    loss_his_train= model_train(model, dataloader=dataLoader, max_epoch=500, print_every=25)
    torch.save(model.state_dict(), Model_path+'/pytorch_CRNN_model.pth')
    sys.exit()
