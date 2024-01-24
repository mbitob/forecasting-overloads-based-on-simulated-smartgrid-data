import argparse
import json
import os
import pandas as pd
import numpy as np 
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from time import time as get_time
from utils import train_one_epoch, evaluate, generate_sequences
from utils import SequenceDataset, QuantileLossMulti, search_csv_files

from models import simpleLSTM_quantiles

torch.manual_seed(1)

def main():
    device = args.device 
    BATCH_SIZE = args.batch_size 
    EPOCH = args.epochs 
    LR = args.lr 
    lookback = args.lookback #Timestamps tp lookback for prediction
    experiment_path = args.experiment_path
    data_path = args.data_path
    positional_encoding = args.positional_encoding 
    #train_val_test_split = args.train_val_test_split # Splits the data into train, valid, and test chunks
    model_save_path =  os.path.join(experiment_path, 'best_valid_model.pt')
    scaling_factors_path = os.path.join(experiment_path, 'scalings.json')
    quantiles = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]
    quantile_weights=[1 for i in range(len(quantiles))]

    os.makedirs(experiment_path, exist_ok=True)
  
    # Dataframe for storing important metrics over epochs
    start_time = get_time()
    df_metric = pd.DataFrame(columns= ['train_loss', 'valid_loss', 'epoch', 'learning_rate' , 'batch_size_train', 'batch_size_valid'])
    best_val_loss = 1e3 #Init
    best_train_loss = 1e3 # Init
        
    # Initialze new network
    if  positional_encoding == 'all':
        n_input = 4
    elif positional_encoding == 'sun':
        n_input = 3
    elif positional_encoding == 'none':
        n_input = 1
        
    #print(n_input)
    model = simpleLSTM_quantiles(n_input_features = n_input, n_hidden = 50, num_layers = 4, n_outputs = 1)
    #model = simpleCfC(n_input_features = n_input, n_hidden = 50, num_layers = 1, n_outputs = 1)
    model.to(device)

    # Define Optimizer and Hyperaparameter/LR scheduler
    lr_tmp = LR # variable for logging lr
    criterion =  QuantileLossMulti(quantiles , quantile_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=0)

    # Start dataset loading
    df_train = pd.read_csv(os.path.join(data_path, 'train.csv'), index_col=0)
    df_test_list = []
    df_valid_list = []

    valid_csv_files = search_csv_files(data_path, 'val')
    test_csv_files = search_csv_files(data_path, 'test')

    for valid_csv_path in valid_csv_files:
        df_val = pd.read_csv(valid_csv_path, index_col=0)
        df_valid_list.append(df_val)
    
    for test_csv_path in test_csv_files:
        df_test = pd.read_csv(test_csv_path, index_col=0)
        df_test_list.append(df_test)

    target_columns = list(df_train.columns)
    df_train_scaled = pd.DataFrame(columns = target_columns)
    df_test_scaled_list = [pd.DataFrame(columns = target_columns) for i in range(len(test_csv_files))]
    df_valid_scaled_list = [pd.DataFrame(columns = target_columns) for i in range(len(valid_csv_files))]
    
    #Min-Max normalization
    scaling_dictionary = {}
    for column in target_columns:
        max_val = df_train.loc[:, column].values.max()
        min_val = df_train.loc[:, column].values.min()
        #scaling_dictionary[column] = {'min' : min_val}
        scaling_dictionary[column] = {'max' : max_val, 'min': min_val}
        #print(df_train_val[column])
        df_train_scaled[column] = (df_train[column].values - min_val) / (max_val - min_val)
    df_train_scaled.index = df_train.index
        
    for i, (df_valid, df_valid_scaled) in enumerate(zip(df_valid_list, df_valid_scaled_list)):
            for column in target_columns:
                max_val = scaling_dictionary[column]['max']
                min_val = scaling_dictionary[column]['min']
                df_valid_scaled[column] = (df_valid[column].values - min_val) / (max_val - min_val)
            df_valid_scaled.index = df_valid.index
            df_valid_scaled_list[i] = df_valid_scaled

    for i, (df_test, df_test_scaled) in enumerate(zip(df_test_list, df_test_scaled_list)):
            for column in target_columns:
                max_val = scaling_dictionary[column]['max']
                min_val = scaling_dictionary[column]['min']
                df_test_scaled[column] = (df_test[column].values - min_val) / (max_val - min_val)
            df_test_scaled.index = df_test.index
            df_test_scaled_list[i] = df_test_scaled
            

    # Write Scaling factors:
    with open(scaling_factors_path, 'w') as file:
        json.dump(scaling_dictionary, file)
    #Write Test data for further evaluation:
    df_train_scaled.to_csv(os.path.join(experiment_path,'train_data_scaled.csv'))

    for i in range(len(df_test_scaled_list)):
        df_test_scaled_list[i].to_csv(os.path.join(experiment_path, f'test{i}_data_scaled.csv'))
    
    for i in range(len(df_valid_scaled_list)):
        df_valid_scaled_list[i].to_csv(os.path.join(experiment_path, f'val{i}_data_scaled.csv'))

    sequences_train = generate_sequences(df_train_scaled, lookback, 1)
    print(len(sequences_train))
    sequences_valid_list = []
    for i, df_valid_scaled in enumerate(df_valid_scaled_list):
            sequences_valid = generate_sequences(df_valid_scaled, lookback, 1)
            sequences_valid_list.append(sequences_valid)

    sequences_valid_all = {}
    k = 0
    for data in sequences_valid_list:
        for j in range(len(data)):
            sequences_valid_all[k] = data[j]
            k += 1

    train_ds = SequenceDataset(sequences_train, positional_encoding = positional_encoding)
    valid_ds = SequenceDataset(sequences_valid_all, positional_encoding = positional_encoding)

    trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    validloader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print("Start training")
    for epoch in range(EPOCH):  
        model.train()
        train_loss = train_one_epoch(model, criterion, optimizer, trainloader)
        model.eval()
        valid_loss = evaluate(model, criterion, validloader)
        scheduler.step()
    
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_train_loss = train_loss
            torch.save(model.state_dict(), model_save_path)
    
        df_new_row = { 
                        'train_loss' : train_loss,
                        'valid_loss' : valid_loss,
                        'epoch' : epoch + 1,
                        'learning_rate' : lr_tmp,
                        'batch_size_train' : BATCH_SIZE,
                        'batch_size_valid' : BATCH_SIZE
            }
        lr_tmp = optimizer.param_groups[0]['lr']
        df_metric.loc[epoch] = df_new_row
        df_metric.to_csv(experiment_path + f'/train_history.csv')

        print(f"Epoch {epoch+1}, train_loss={train_loss:0.4g}, val_loss={valid_loss:0.4g}")
    
    training_time = get_time() - start_time
    
    print('Trainig took: {:.2f}s for {} epochs'.format(training_time, epoch))
    print('Finish training, Best-Test-Loss: %0.5f, Best-Train-Loss:%0.5f '  % (best_val_loss, best_train_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--batch_size', type=int, default=64, help='total batchsz for train and test')
    parser.add_argument('--epochs', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lookback', type=int, default=100, help='look back of network')
    parser.add_argument('--positional_encoding', type = str, default='all', choices=['none', 'sun', 'all'], help='defines which data to use for forecasting')
    parser.add_argument('--data_path', type=str, default='./raw_data/final_splits')
    parser.add_argument('--experiment_path', type=str, default='./saved_runs/test_quantiles')
    args = parser.parse_args()

    print("device is --------------", args.device)

    main()