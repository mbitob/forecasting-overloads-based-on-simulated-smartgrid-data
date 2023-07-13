import torch.nn as nn
import numpy as np

class simpleLSTM(nn.Module):
    def __init__(self, n_input_features = 4, n_hidden = 50, num_layers = 4, n_outputs = 1, batch_first = False, bidirectional = False):
        super().__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        #n_input_features = 4
        #n_hidden = 50
        # LSTM Layer
        #self.cnn = nn.Conv1d(in_channels = n_features, out_channels=n_features, kernel_size=18, padding='same')
        self.rnn = nn.LSTM( n_input_features,
                            n_hidden,
                            num_layers,
                            batch_first=False, 
                            bidirectional = bidirectional)
        #self.rnn = CfC(n_features, n_hidden, batch_first=False)
        if self.bidirectional:
            self.linear = nn.Linear(n_hidden*2, n_outputs)
        else:
            self.linear = nn.Linear(n_hidden, n_outputs)


    def forward(self, x, hx=None, cx=None):
        x, hx = self.rnn(x, hx)  # hx is the hidden state of the RNN
        #x = x[:, -1, :] # Batch_first = True
        #if self.bidirectional:
            #print(x.shape)
            #x = x.view(200, -1, num_directions, hidden_size
        x = x[-1, :, :] # Batch_first = False

        x = self.linear(x)
        #print(x.shape)
        return x, hx