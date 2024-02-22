import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


# Function that creates sequences and targets 
def generate_sequences(df: pd.Series, tw: int, pw: int):
  '''
  df: Pandas DataFrame of the univariate time-series
  tw: Training Window - Integer defining how many steps to look back
  pw: Prediction Window - Integer defining how many steps forward to predict

  returns: dictionary of sequences and targets for all sequences
  '''
  data = dict() # Store results into a dictionary
  L = len(df)
  for i in range(L-tw-1):

    # Get current sequence  
    #sequence = df[i:i+tw].values[:,0:-1]  # [power, altitude, azimuth, irradiance]
    power_not_shifted = df[i:i+tw].values[:,0:1]
    #print(power_not_shifted.shape)
    sequence_shift = df[i+1:i+tw+1].values[:,1:-1]
    #print(sequence_shift.shape)
    sequence = np.concatenate((power_not_shifted, sequence_shift), axis = 1)
    #print(sequence.shape)
    # Get values right after the current sequence
    target = df[i+tw:i+tw+pw].values[:,0] # [power]
    
    data[i] = {'sequence': sequence, 'target': target}
  return data


class SequenceDataset(Dataset):

  def __init__(self, df, positional_encoding = True):
    self.data = df
    self.positional_encoding = positional_encoding

  def __getitem__(self, idx):
    sample = self.data[idx]

    if self.positional_encoding == 'all':
       sample_sequence = sample['sequence']
    elif self.positional_encoding == 'sun':
       sample_sequence = sample['sequence'][:, 0:3] # With sun positional encoding and active power
    else:
       sample_sequence = sample['sequence'][:, 0:1] # Without positional encoding only active power
       
    #print(sample_sequence.shape)
    #print(sample['target'].shape)

    return torch.Tensor(sample_sequence), torch.Tensor(sample['target'])#.squeeze()
  
  def __len__(self):
    return len(self.data)
  

def train_one_epoch(model, criterion, optimizer, trainloader):
    running_loss = 0.0
    pbar = tqdm(total=len(trainloader))
    model.train()
    device = next(model.parameters()).device  # get device the model is located on
    for i, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)  # move data to same device as the model
        labels = labels.to(device)
        inputs = torch.swapaxes(inputs, 0, 1)
        #print('inputs', inputs.shape)
        #print('labels', labels.shape)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs, hx = model(inputs)
        #print('outputs', outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        pbar.set_description(f"loss={running_loss / (i + 1):0.4g}")
        pbar.update(1)
    pbar.close()

    train_loss = running_loss / (i + 1)

    
    return train_loss
    


def evaluate(model, criterion, valloader):
    losses = []
    
    device = next(model.parameters()).device  # get device the model is located on
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs = inputs.to(device)  # move data to same device as the model
            labels = labels.to(device)
            inputs = torch.swapaxes(inputs, 0, 1)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
    return np.mean(losses)


def run_closed_loop(model, whole_sequence, lookback = 100, future_prediction=4, use_positional_encoding = 'all'):
    device = torch.device("cpu")
    #device = next(model.parameters()).device
    model.to(device)
    model.eval()
    #print(device)
    hx = None  # Hidden state of the RNN
    #input_sequence = whole_sequence[0:lookback]

    #print('whole_sequence', whole_sequence.shape)
    if use_positional_encoding == 'all' or use_positional_encoding == 'all_original':
      input_numpy = np.array([whole_sequence[0:lookback, 0], whole_sequence[1:lookback+1, 1], whole_sequence[1:lookback+1, 2], whole_sequence[1:lookback+1, 3]])
      input = torch.Tensor(input_numpy.T)
      input = input.view(-1,1,4)
      #print(input.shape)
    elif use_positional_encoding == 'sun':
      input_numpy = np.array([whole_sequence[0:lookback, 0], whole_sequence[1:lookback+1, 1], whole_sequence[1:lookback+1, 2]])
      #print(input_numpy.shape)
      input = torch.Tensor(input_numpy.T)
      #print(input.shape)
      input = input.view(-1,1,3)
    else:
      input_numpy = np.array([whole_sequence[0:lookback, 0]])
      input = torch.Tensor(input_numpy.T)
      input = input.view(-1,1,1)
    input.to(device)
    predictions = []
    with torch.no_grad():
        
        for i in range(input.shape[0]):
            if use_positional_encoding == 'all' or use_positional_encoding == 'all_original':
              input_tmp = input[i,:,:].view(-1,1,4)
            elif use_positional_encoding == 'sun':
              input_tmp = input[i,:,:].view(-1,1,3)
            else:
              input_tmp = input[i,:,:].view(-1,1,1)

            pred, hx = model(input_tmp, hx)

        for i in range(future_prediction-1):
            if use_positional_encoding == 'all':
              input = torch.Tensor([pred[0,0], whole_sequence[lookback + i+1, 1], whole_sequence[lookback + i+1, 2],  whole_sequence[lookback + i+1, 4]]).view(1,1,4) 
            elif use_positional_encoding == 'all_original':
              input = torch.Tensor([pred[0,0], whole_sequence[lookback + i+1, 1], whole_sequence[lookback + i+1, 2],  whole_sequence[lookback + i+1, 3]]).view(1,1,4) 
            elif use_positional_encoding == 'sun':
              input = torch.Tensor([pred[0,0], whole_sequence[lookback + i+1, 1], whole_sequence[lookback + i+1, 2]]).view(1,1,3) 
            else:
              input = torch.Tensor([pred[0,0]]).view(1,1,1) 
            pred, hx = model(input, hx)
            predictions.append(pred.detach().numpy()[0,0])
    return np.array(predictions)


def run_closed_loop_quantile(model, whole_sequence, lookback = 100, future_prediction=4, use_positional_encoding = 'all'):
    device = torch.device("cpu")
    #device = next(model.parameters()).device
    model.to(device)
    model.eval()
    #print(device)
    hx = None  # Hidden state of the RNN
    #input_sequence = whole_sequence[0:lookback]

    #print('whole_sequence', whole_sequence.shape)
    if use_positional_encoding == 'all' or use_positional_encoding == 'all_original':
      input_numpy = np.array([whole_sequence[0:lookback, 0], whole_sequence[1:lookback+1, 1], whole_sequence[1:lookback+1, 2], whole_sequence[1:lookback+1, 3]])
      input = torch.Tensor(input_numpy.T)
      input = input.view(-1,1,4)
      #print(input.shape)
    elif use_positional_encoding == 'sun':
      input_numpy = np.array([whole_sequence[0:lookback, 0], whole_sequence[1:lookback+1, 1], whole_sequence[1:lookback+1, 2]])
      #print(input_numpy.shape)
      input = torch.Tensor(input_numpy.T)
      #print(input.shape)
      input = input.view(-1,1,3)
    else:
      input_numpy = np.array([whole_sequence[0:lookback, 0]])
      input = torch.Tensor(input_numpy.T)
      input = input.view(-1,1,1)
    input.to(device)
    predictions = []
    predictions_quantiles = []
    with torch.no_grad():
        
        for i in range(input.shape[0]):
            if use_positional_encoding == 'all' or use_positional_encoding == 'all_original':
              input_tmp = input[i,:,:].view(-1,1,4)
            elif use_positional_encoding == 'sun':
              input_tmp = input[i,:,:].view(-1,1,3)
            else:
              input_tmp = input[i,:,:].view(-1,1,1)

            pred_quantiles, hx = model(input_tmp, hx)
            #print(pred_quantiles.shape)
            pred, _ = torch.median(pred_quantiles, dim=-1)
            #pred = pred_quantiles[0:1, 0:1, 5]
            #print(pred)

        for i in range(future_prediction-1):
            if use_positional_encoding == 'all':
              input = torch.Tensor([pred[0,0], whole_sequence[lookback + i+1, 1], whole_sequence[lookback + i+1, 2],  whole_sequence[lookback + i+1, 4]]).view(1,1,4) 
            elif use_positional_encoding == 'all_original':
              input = torch.Tensor([pred[0,0], whole_sequence[lookback + i+1, 1], whole_sequence[lookback + i+1, 2],  whole_sequence[lookback + i+1, 3]]).view(1,1,4) 
            elif use_positional_encoding == 'sun':
              input = torch.Tensor([pred[0,0], whole_sequence[lookback + i+1, 1], whole_sequence[lookback + i+1, 2]]).view(1,1,3) 
            else:
              input = torch.Tensor([pred[0,0]]).view(1,1,1) 
            pred_qunatiles, hx = model(input, hx)
            pred, _ = torch.median(pred_qunatiles, dim=-1)
            #pred = pred_quantiles[0:1, 0:1, 5]
            predictions.append(pred.detach().numpy()[0,0])
            predictions_quantiles.append(pred_qunatiles.detach().numpy()[0,0,:])
    return np.array(predictions), np.array(predictions_quantiles)


def search_csv_files(folder_path, term):
    results = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv') and term in file:
                file_path = os.path.join(root, file)
                results.append(file_path)
    return results

def perf_measure(y_actual, y_pred):
    tp = np.sum((y_actual==1) & (y_pred==1))
    tn = np.sum((y_actual==0) & (y_pred==0))
    fp = np.sum((y_actual==0) & (y_pred==1))
    fn = np.sum((y_actual==1) & (y_pred==0))
    
    return(tp, tn, fp, fn)


import warnings

import torch
import torch.nn.functional as F
from torch import nn


class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.5):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, input, target):
        if not (target.size() == input.size()):
            warnings.warn(
                "Using a target size ({}) that is different to the input size ({}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.".format(
                    target.size(), input.size()
                ),
                stacklevel=2,
            )
        error = (target - input).float()
        return torch.max((self.quantile - 1) * error, self.quantile * error).mean()


class QuantileLossMulti(nn.Module):
    def __init__(
        self,
        quantiles=[round(0.1 * i, 1) for i in range(1, 10)],
        quantile_weights=[1.0 / 9 for i in range(1, 10)],
        use_qweights=False,
    ):
        super(QuantileLossMulti, self).__init__()
        self.quantiles = quantiles
        self.quantile_losses = nn.ModuleList(
            [QuantileLoss(quantile) for quantile in quantiles]
        )
        self.quantile_weights = quantile_weights
        self.use_qweights = use_qweights

    def forward(self, outputs, input):
        losses = []
        for i in range(len(self.quantiles)):
            quantile_loss = self.quantile_losses[i](outputs[:, :, i], input)
            losses.append(quantile_loss)

        # Return combined quantile loss and predictions
        # total_loss = torch.mean(torch.stack(losses))

        # --------- the total loss is a weighted sum of the quantile losses ------------
        # use weights proportional to the quantiles frequencies
        if self.use_qweights:
            freq = torch.tensor(
                [
                    round(
                        float(
                            sum(
                                1
                                * (
                                    input
                                    > torch.quantile(
                                        input, i * 0.1, dim=0, keepdim=True
                                    )
                                )
                            )
                            / len(input)
                        ),
                        1,
                    )
                    for i in range(1, input.shape[1] + 1)
                ]
            )
            weights = torch.exp(freq) / torch.sum(torch.exp(freq))

            total_loss = 0
            for i in range(len(self.quantiles)):
                total_loss += weights[i] * losses[i]

        # use predefined weights
        else:
            total_loss = 0
            for i in range(len(self.quantiles)):
                total_loss += self.quantile_weights[i] * losses[i]

        return total_loss