import torch
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
            self.linear = nn.Sequential(nn.Linear(n_hidden, 20), nn.ReLU(), nn.Linear(20, n_outputs)) # TODO: Should be same as simpleLSTM_quantiles


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
    

class simpleLSTM_quantiles(nn.Module):
    def __init__(self, n_input_features = 4, n_hidden = 50, num_layers = 4, n_outputs = 1, batch_first = False, quantiles = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]):
        super().__init__()
        self.n_outputs = n_outputs
        self.batch_first = batch_first
        self.rnn = nn.LSTM( n_input_features,
                            n_hidden,
                            num_layers,
                            batch_first=False, 
                            )
        self.quantiles = quantiles
        #self.quantile_heads = nn.ModuleList([nn.Linear(n_hidden, n_outputs) for i in range(len(quantiles))])
        self.quantile_heads = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, 20), nn.ReLU(), nn.Linear(20, n_outputs)) for i in range(len(quantiles))]) # TODO: Should be same as simpleLSTM
        
    def forward(self, x, hx=None, cx=None):
        # input x should be the median in the autoregressive fashion
        x, hx = self.rnn(x, hx)  # hx is the hidden state of the RNN
        x = x[-1, :, :] # Batch_first = False
        #print('LSTM_out', x.shape)
        quantiles_out = []
        for i, quantile_head in enumerate(self.quantile_heads):
            #print('quantile_head', quantile_head(x).shape)
            quantile_pred = quantile_head(x).view(-1,self.n_outputs,1) #(B,1,1)
            if i == 0:
                quantiles_out = quantile_pred
            else:
                quantiles_out = torch.concat((quantiles_out, quantile_pred) , dim=-1)
            #print('pred shape', quantile_pred.shape)
        #print('qunatile_out.shape', quantiles_out.shape)
        return quantiles_out, hx
    

#class simpleCfC_small(nn.Module):
#    def __init__(self, n_input_features = 4, n_hidden = 50, num_layers = 4, n_outputs = 1, batch_first = False,):
#        super().__init__()
#        self.batch_first = batch_first     
# 
#        self.rnn = CfC(n_input_features, n_hidden, batch_first=batch_first) #nn.Sequential(*cfc_cells)
#
#        self.linear = nn.Linear(n_hidden, n_outputs)
#
#
#    def forward(self, x, hx=None):
#        x, hx = self.rnn(x, hx)  # hx is the hidden state of the RNN
#        x = x[-1, :, :] # Batch_first = False
#
#        x = self.linear(x)
#        #print(x.shape)
#        return x, hx
    

#class simpleCfC(nn.Module):
#    def __init__(self, n_input_features = 4, n_hidden = 50, num_layers = 4, n_outputs = 1, batch_first = False, bidirectional = False):
#        super().__init__()
#        self.batch_first = batch_first     
# 
#        cfc_cells = []
#        for i in range(num_layers):
#            if i == 0:
#                cfc_cells.append(CfC(n_input_features, n_hidden, batch_first=batch_first))
#            else: 
#                cfc_cells.append(CfC(n_hidden, n_hidden, batch_first=batch_first))
#
#        self.rnn1 = CfC(n_input_features, n_hidden, batch_first=batch_first) #nn.Sequential(*cfc_cells)
#    
#        self.rnn2 = CfC(n_hidden, n_hidden, batch_first=batch_first)
#        self.rnn3 = CfC(n_hidden, n_hidden, batch_first=batch_first)
#        self.rnn4 = CfC(n_hidden, n_hidden, batch_first=batch_first)
#
#        self.linear = nn.Linear(n_hidden, n_outputs)
#
#
#    def forward(self, x, hx=None):
#        x, hx = self.rnn1(x, hx)  # hx is the hidden state of the RNN
#        x, hx = self.rnn2(x, hx)
#        x, hx = self.rnn3(x, hx)
#        x, hx = self.rnn4(x, hx)
#
#        #x = x[:, -1, :] # Batch_first = True
#        #if self.bidirectional:
#            #print(x.shape)
#            #x = x.view(200, -1, num_directions, hidden_size
#        x = x[-1, :, :] # Batch_first = False
#
#        x = self.linear(x)
#        #print(x.shape)
#        return x, hx