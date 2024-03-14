import argparse
import json
import os, sys
import pandas as pd
import torch 
from torch.utils.data import DataLoader
from time import time as get_time
from tools.utils import train_one_epoch_multioutput, evaluate_multioutput, generate_sequences
from tools.utils import SequenceDataset_Multiple, QuantileLossMulti, search_csv_files

from tools.models import simpleLSTM_quantiles

torch.manual_seed(1)

def train_lstm_quantile():
    
    device  	        = args.device 
    BATCH_SIZE          = args.batch_size 
    EPOCH               = args.epochs 
    LR                  = args.lr 
    lookback            = args.lookback    #Timestamps tp lookback for prediction
    experiment_path     = args.experiment_path
    positional_encoding = args.positional_encoding 
    #target              = args.target
    n_outputs           = 3 #NP, EC, LV
    use_irradiance_real = True # False -> forecasted irradiance, True -> real irradiance
    exp_name = 'EC_NP_LV_first_real_irradiance'
    n_hidden = 50
    start_time = get_time()

    # set up paths
    data_path           = os.path.join(experiment_path, 'analysis', 'lstm', "data")
    model_save_path     = os.path.join(experiment_path,'analysis', 'lstm', "model")
    os.makedirs(model_save_path, exist_ok=True)
    #scaling_factors_file = os.path.join(data_path, f'{target}_scalings.json')
    #scaling_data_path = os.path.join(data_path,f'LV_scalings.json')
    #with open(scaling_data_path, 'r') as file:
    #    scaling_dict = json.load(file)    
    #scalings_max_min = (scaling_dict['LV']['max'], scaling_dict['LV']['min'])
    # define quantiles
    quantiles = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]
    quantile_weights=[1 for i in range(len(quantiles))]

   
    # Dataframe for storing important metrics over epochs
    df_metric       = pd.DataFrame(columns= ['train_loss', 'valid_loss', 'epoch', 'learning_rate' , 'batch_size_train', 'batch_size_valid'])
    best_val_loss   = 1e3 #Init
    best_train_loss = 1e3 # Init
        
    # Initialze new network
    if  positional_encoding == 'all':
        n_input = 4
    elif positional_encoding == 'sun':
        n_input = 3
    elif positional_encoding == 'none':
        n_input = 1
        
    model = simpleLSTM_quantiles(n_input_features = n_input, n_hidden = n_hidden, num_layers = 4, n_outputs = n_outputs)
    model.to(device)

    # Define Optimizer and Hyperaparameter/LR scheduler
    lr_tmp = LR # variable for logging lr
    criterion =  QuantileLossMulti(quantiles, quantile_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=0)

    # Load dataset for NP, EC and create dataloaders
    # Start dataset loading

    all_sequences_train = []
    all_sequences_valid = []

    for target in ['NP', 'EC', 'LV']:
        df_train_scaled = pd.read_csv(os.path.join(data_path, f'{target}_scaled_by_LV_train.csv'), index_col=0)
        df_test_scaled_list = []
        df_valid_scaled_list = []

        valid_csv_files = search_csv_files(data_path, f'{target}_scaled_by_LV_val')
        test_csv_files = search_csv_files(data_path, f'{target}_scaled_by_LV_test')

        for valid_csv_path in valid_csv_files:
            df_val_scaled = pd.read_csv(valid_csv_path, index_col=0)
            df_valid_scaled_list.append(df_val_scaled)

        for test_csv_path in test_csv_files:
            df_test_scale = pd.read_csv(test_csv_path, index_col=0)
            df_test_scaled_list.append(df_test_scale)

        sequences_train = generate_sequences(df_train_scaled, lookback, 1, use_irradiance_real = use_irradiance_real)
        print(len(sequences_train))
        sequences_valid_list = []
        for i, df_valid_scaled in enumerate(df_valid_scaled_list):
                sequences_valid = generate_sequences(df_valid_scaled, lookback, 1,  use_irradiance_real = use_irradiance_real)
                sequences_valid_list.append(sequences_valid)

        sequences_valid_all = {}
        k = 0
        for data in sequences_valid_list:
            for j in range(len(data)):
                sequences_valid_all[k] = data[j]
                k += 1

        all_sequences_train.append(sequences_train)
        all_sequences_valid.append(sequences_valid_all)

    #print(len(all_sequences_train))
    #print(len(all_sequences_valid))

    train_ds = SequenceDataset_Multiple(all_sequences_train, positional_encoding = positional_encoding)
    valid_ds = SequenceDataset_Multiple(all_sequences_valid, positional_encoding = positional_encoding)

    #print(len(train_ds))
    #print(len(valid_ds))

    trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    validloader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print("Start training")
    for epoch in range(EPOCH):  
        model.train()
        train_loss = train_one_epoch_multioutput(model, criterion, optimizer, trainloader)
        model.eval()
        valid_loss = evaluate_multioutput(model, criterion, validloader)
        scheduler.step()
    
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_train_loss = train_loss
            torch.save(model.state_dict(), os.path.join(model_save_path,f'{exp_name}_best_valid_model.pt'))
    
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
        df_metric.to_csv(os.path.join(model_save_path,f'{exp_name}_train_history.csv'))

        print(f"Epoch {epoch+1}, train_loss={train_loss:0.4g}, val_loss={valid_loss:0.4g}")
    
    training_time = get_time() - start_time
    
    print('Training took: {:.2f}s for {} epochs'.format(training_time, epoch))
    print('Finish training, Best-Test-Loss: %0.5f, Best-Train-Loss:%0.5f '  % (best_val_loss, best_train_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=64, help='total batchsz for train and test')
    parser.add_argument('--epochs', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lookback', type=int, default=100, help='look back of network')
    parser.add_argument('--positional_encoding', type = str, default='all', choices=['none', 'sun', 'all'], help='defines which data to use for forecasting')
    parser.add_argument('--experiment_path', type=str, default='./experiments/Rural-LV1-101-2034/BaseScenario/0101-3112', help='path to experiment')
    
    args = parser.parse_args()

    print("device is --------------", args.device)

    train_lstm_quantile()