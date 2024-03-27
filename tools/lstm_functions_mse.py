import argparse
import json
import os, sys
import pandas as pd
import numpy as np 
from time import time as get_time
import math

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tools.utils import train_one_epoch_multioutput, evaluate_multioutput, generate_sequences,SequenceDataset_Multiple, search_csv_files, run_closed_loop
from tools.models import simpleLSTM

torch.manual_seed(1234)

def lstm_mse_train(args):
    
    device                  = args.device 
    BATCH_SIZE              = args.batch_size 
    EPOCH                   = args.epochs 
    LR                      = args.lr 
    lookback                = args.lookback #Timestamps tp lookback for prediction
    settlement              = args.settlement
    experiment              = args.experiment
    time_horizon            = args.time_horizon
    n_outputs               = args.n_outputs #NP, EC, LV
    n_inputs                = args.n_inputs
    use_irradiance_real     = args.use_irradiance_real # False -> forecasted irradiance, True -> real irradiance
    analysis_name           = args.analysis_name
    scale_NP_EC_based_on_LV = args.scale_NP_EC_based_on_LV 
    
    n_hidden    = 50
    start_time  = get_time()

    # ger postfix for files based on scaling
    if scale_NP_EC_based_on_LV:
        scaling_postfix = "_scaled_by_LV"
    else:
        scaling_postfix = "_scaled"

    # set up paths
    data_path           = os.path.join("./experiments", settlement, experiment, time_horizon, 'analysis', 'lstm', "train_test_data")
    model_save_path     = os.path.join("./experiments", settlement, experiment, time_horizon, 'analysis', 'lstm', analysis_name, "model")
    os.makedirs(model_save_path, exist_ok=True)

  
    # Dataframe for storing important metrics over epochs
    start_time = get_time()
    df_metric = pd.DataFrame(columns= ['train_loss', 'valid_loss', 'epoch', 'learning_rate' , 'batch_size_train', 'batch_size_valid'])
    best_val_loss = 1e3 #Init
    best_train_loss = 1e3 # Init
        
    model = simpleLSTM(n_input_features = n_inputs, n_hidden = n_hidden, num_layers = 4, n_outputs = n_outputs)
    model.to(device)

    # Define Optimizer and Hyperparameter/LR scheduler
    lr_tmp = LR # variable for logging lr
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=0)

    all_sequences_train = []
    all_sequences_valid = []

    if n_outputs == 3:
        used_targets = [ 'NP', 'EC', 'LV']
    else:
        used_targets = [args.target]

    for target in used_targets:

        df_train_scaled = pd.read_csv(os.path.join(data_path, f'{target}{scaling_postfix}_train.csv'), index_col=0)

        df_valid_scaled_list = []

        valid_csv_files = search_csv_files(data_path, f'{target}{scaling_postfix}_val')

        for valid_csv_path in valid_csv_files:
            df_val_scaled = pd.read_csv(valid_csv_path, index_col=0)
            df_valid_scaled_list.append(df_val_scaled)


        sequences_train = generate_sequences(df_train_scaled, lookback, 1, use_irradiance_real = use_irradiance_real)

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

    train_ds = SequenceDataset_Multiple(all_sequences_train, positional_encoding = "all", n_outputs=n_outputs) # TODO: remove encoding or not?
    valid_ds = SequenceDataset_Multiple(all_sequences_valid, positional_encoding = "all", n_outputs=n_outputs) # TODO: remove encoding or not?


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
            torch.save(model.state_dict(), os.path.join(model_save_path,f'best_valid_model.pt'))
    
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
        df_metric.to_csv(os.path.join(model_save_path,f'train_history.csv'))

        print(f"Epoch {epoch+1}, train_loss={train_loss:0.4g}, val_loss={valid_loss:0.4g}")
    
    training_time = get_time() - start_time
    
    print('Training took: {:.2f}s for {} epochs'.format(training_time, epoch))
    print('Finish training, Best-Test-Loss: %0.5f, Best-Train-Loss:%0.5f '  % (best_val_loss, best_train_loss))

def lstm_mse_forecast(args):

    settlement      = args.settlement
    experiment      = args.experiment
    time_horizon    = args.time_horizon
    n_outputs       = args.n_outputs
    n_input         = args.n_inputs
    analysis_name   = args.analysis_name
    model_name      = f'best_valid_model.pt'


    experiment_path = os.path.join('./experiments', settlement, experiment, time_horizon)
    lstm_data_path  = os.path.join(experiment_path,"analysis", "lstm", "train_test_data")
    lstm_model_path = os.path.join(experiment_path, "analysis", "lstm", analysis_name, "model")

    model_file = os.path.join(lstm_model_path, model_name)
       
    model = simpleLSTM(n_input_features=n_input, n_hidden=50, num_layers=4, n_outputs=n_outputs)
    model.load_state_dict(torch.load(model_file,map_location=torch.device("cpu")))
    run_inference = run_closed_loop

    if n_outputs == 3:
        used_targets = [ 'NP', 'EC', 'LV']
        multi_output = True
        additional_info = "_scaled_by_LV"
        scaling_data_path = os.path.join(lstm_data_path,f'LV_scalings_by_LV.json')
        scaling_target = "LV"
    else:
        multi_output = False
        additional_info = "_scaled"
        used_targets = [args.target]
        scaling_data_path = os.path.join(lstm_data_path,f'{args.target}_scalings.json')
        scaling_target = args.target

    #Read test data
    df_test_scaled_list_LV = []
    for i in range(6):
        test_data_path = os.path.join(lstm_data_path, f'LV{additional_info}_test{i}.csv')
        df_test_scaled = pd.read_csv(test_data_path, index_col = 0)
        # Convert timestamps to datetime objects 
        datetime_index = pd.to_datetime(df_test_scaled.index)
        df_test_scaled.index = datetime_index
        df_test_scaled_list_LV.append(df_test_scaled)

    df_test_scaled_list_NP = []
    for i in range(6):
        test_data_path = os.path.join(lstm_data_path, f'NP{additional_info}_test{i}.csv')
        df_test_scaled = pd.read_csv(test_data_path, index_col = 0)
        # Convert timestamps to datetime objects 
        datetime_index = pd.to_datetime(df_test_scaled.index)
        df_test_scaled.index = datetime_index
        df_test_scaled_list_NP.append(df_test_scaled)

    df_test_scaled_list_EC = []
    for i in range(6):
        test_data_path = os.path.join(lstm_data_path, f'EC{additional_info}_test{i}.csv')
        df_test_scaled = pd.read_csv(test_data_path, index_col = 0)
        # Convert timestamps to datetime objects 
        datetime_index = pd.to_datetime(df_test_scaled.index)
        df_test_scaled.index = datetime_index
        df_test_scaled_list_EC.append(df_test_scaled)

    # Read scalings
    
    with open(scaling_data_path, 'r') as file:
        scaling_dict = json.load(file)

    # prepare scalings to rescale original power profile:
    max_power = scaling_dict[scaling_target]['max']
    min_power = scaling_dict[scaling_target]['min']

    months = ['july', 'august', 'september', 'october', 'november', 'december']

    df_metrics_comparison = []

    figs = []
    for i in range(6):
        lookback = 96
        
        if multi_output:
            test_values = df_test_scaled_list_LV[i].values
        else:
            if scaling_target == 'NP':
                test_values = df_test_scaled_list_NP[i].values
            elif scaling_target == 'EC':
                test_values = df_test_scaled_list_EC[i].values
            else:
                test_values = df_test_scaled_list_LV[i].values

        groundtruth_LV = df_test_scaled_list_LV[i].values[lookback+1::, 0] # There is a + 1 shift in the inference loop #TODO ASK
        groundtruth_NP = df_test_scaled_list_NP[i].values[lookback+1::, 0]
        groundtruth_EC = df_test_scaled_list_EC[i].values[lookback+1::, 0]
    
        num_test_samples = test_values.shape[0]
        steps2predict = num_test_samples-lookback
        
        #groundtruth = test_values[lookback:-1, 0] 
        time = pd.Series(df_test_scaled_list_LV[i].index)[lookback:-1]
        predictions_with_irradiance_fc, predictions_with_irradiance_fc_all = run_inference(model, test_values, lookback=lookback, 
                                                                                                            future_prediction=steps2predict, 
                                                                                                            use_positional_encoding='all', 
                                                                                                            multi_output=multi_output, 
                                                                                                            )
    
        predictions_with_irradiance_original, predictions_with_irradiance_original_all = run_inference(model, test_values, lookback=lookback, 
                                                                                                                        future_prediction=steps2predict,    
                                                                                                                        use_positional_encoding='all_original', 
                                                                                                                        multi_output=multi_output,
                                                                                                                        )
    
        groundtruth_LV = groundtruth_LV * (max_power - min_power) + min_power #Rescale to original values
        groundtruth_NP = groundtruth_NP * (max_power - min_power) + min_power #Rescale to original values
        groundtruth_EC = groundtruth_EC * (max_power - min_power) + min_power #Rescale to original values
    
        predictions_with_irradiance_fc = predictions_with_irradiance_fc * (max_power - min_power) + min_power #Rescale to original values
        predictions_with_irradiance_original = predictions_with_irradiance_original * (max_power - min_power) + min_power #Rescale to original values
    
        predictions_with_irradiance_fc_all= predictions_with_irradiance_fc_all * (max_power - min_power) + min_power #Rescale to original values
        predictions_with_irradiance_original_all = predictions_with_irradiance_original_all * (max_power - min_power) + min_power #Rescale to original values
    
        #numpy to pandas
        predictions_with_irradiance_fc_all_df = pd.DataFrame(predictions_with_irradiance_fc_all, index = time, columns = used_targets)
        predictions_with_irradiance_original_all_df  = pd.DataFrame(predictions_with_irradiance_original_all, index = time, columns = used_targets)

        predictions_with_irradiance_fc_all_df.to_csv(os.path.join(lstm_model_path, f'predictions_with_forecast_irradiance_{months[i]}.csv'))
        predictions_with_irradiance_original_all_df.to_csv(os.path.join(lstm_model_path, f'predictions_with_real_irradiance_{months[i]}.csv'))


        # for NP, EC, LV:
        # Evaluate squared error, absolute error for forecasted and original irradiance and write to dataframe
        # Evaluate percentage of being in uncertainty bounds for forecasted and original irradiance and write to dataframe
        # columns of dataframe: MAE, RMSE, p_in_bound, mean_interval_width, month
        # Two dataframes for forecasted and original irradiance
            
        df_rows_month = pd.DataFrame(columns= ['MAE', 'RMSE', 'source', 'month', 'predict_irradiance']) 
        for k, source in enumerate(used_targets):
            if source == 'NP':
                groundtruth = groundtruth_NP
            elif source == 'EC':
                groundtruth = groundtruth_EC
            elif source == 'LV':
                groundtruth = groundtruth_LV
            
            squared_error_forecast  = (predictions_with_irradiance_fc_all[:,k]       - groundtruth)**2
            squared_error_original  = (predictions_with_irradiance_original_all[:,k] - groundtruth)**2
            absolute_error_forecast = np.abs(groundtruth - predictions_with_irradiance_fc_all[:,k])
            absolute_error_original = np.abs(groundtruth - predictions_with_irradiance_original_all[:,k])
    
    
            mse_original_irradiance = squared_error_original.mean()
            mse_forecast_irradiance = squared_error_forecast.mean()
    
            mae_original_irradiance = absolute_error_original.mean()
            mae_forecast_irradiance = absolute_error_forecast.mean()
            rmse_original_irradiance = math.sqrt(mse_original_irradiance)
            rmse_forecast_irradiance = math.sqrt(mse_forecast_irradiance)
    
            df_row = pd.DataFrame({ 'MAE'                   : [mae_forecast_irradiance,  mae_original_irradiance] , 
                                    'RMSE'                  : [rmse_forecast_irradiance, rmse_original_irradiance],
                                    'source'                : [source, source], 
                                    'month'                 : [months[i], months[i]], 
                                    'predict_irradiance'    : ['forecast', 'real']})
    
            df_rows_month = pd.concat([df_rows_month, df_row], ignore_index=True)
        
        df_metrics_comparison.append(df_rows_month)
    
    df_metrics_comparison = pd.concat(df_metrics_comparison, ignore_index=True)

    df_metrics_comparison.to_csv(os.path.join(lstm_model_path, f'comparison_metrics.csv'))
