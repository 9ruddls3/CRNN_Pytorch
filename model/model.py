class CNN_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CNN_block, self).__init__()
        self.conv_1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_channel)
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channel)
        self.batchnorm3 = nn.BatchNorm2d(num_features=out_channel)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv_1(x)))
        x = self.relu(self.batchnorm2(self.conv_2(x)))
        x = self.relu(self.batchnorm3(self.conv_3(x)))
        x = self.maxpool(x)
        return x


class ToRNN(nn.Module):
    def forward(self, x):
        x = x.permute(3, 0, 1, 2)
        W, N, C, H = x.size()
        x.contiguous()
        return x.view(W, N, -1)


class BiDireRNN(nn.Module):
    def __init__(self):
        super(BiDireRNN, self).__init__()
        self.hidden_dim = global RNN_hidden_dim
        self.num_layers = RNN_layer
        self.sql = sequence_len
        self.bsize = batch_size
        self.dropout = RNN_dropout
        self.rnn_type = RNN_type
        self.rnn = self.rnn_layer()
        self.hidden = None
        self.init_hidden(batch_size)

    def rnn_layer(self):
        if self.rnn_type == 'RNN':
            return nn.RNN(RNN_input_dim, self.hidden_dim, self.num_layers, dropout=self.dropout, bidirectional=True)
        elif self.rnn_type == 'LSTM':
            return nn.LSTM(RNN_input_dim, self.hidden_dim, self.num_layers, dropout=self.dropout, bidirectional=True)
        elif self.rnn_type == 'GRU':
            return nn.GRU(RNN_input_dim, self.hidden_dim, self.num_layers, dropout=self.dropout, bidirectional=True)
        else:
            raise AssertionError('unknown RNN type:', self.rnn_type)

    def init_hidden(self, bsize):
        if self.rnn_type == 'LSTM':
            self.hidden = (Variable(torch.zeros(self.num_layers * 2, bsize, self.hidden_dim).type(dtype)),
                           Variable(torch.zeros(self.num_layers * 2, bsize, self.hidden_dim).type(dtype)))
        else:
            self.hidden = Variable(torch.zeros(self.num_layers * 2, bsize, self.hidden_dim).type(dtype))

    def forward(self, x):
        rnn_out, self.hidden = self.rnn(x, self.hidden)
        return rnn_out


class Model(nn.Module):
    def __init__(self, use_VGG_extractor=False):
        super(Model, self).__init__()
        if use_VGG_extractor:
            self.feature_extractor = nn.Sequential(*([vgg16.features[i] for i in range(17)]))
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        else:
            self.feature_extractor = nn.Sequential(*([CNN_block(3, 64), CNN_block(64, 128), CNN_block(128, 256)]))
        self.toRNN = ToRNN()
        self.RNN = BiDireRNN()
        self.toTraget = nn.Linear(RNN_hidden_dim * 2, vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        feature = self.feature_extractor(x)
        RNN_input = self.toRNN(feature)
        RNN_out = self.RNN(RNN_input)
        tag_scores = self.toTraget(RNN_out)
        return tag_scores
