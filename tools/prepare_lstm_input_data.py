import pandas as pd
import json
import os


def prepare_lstm_data(settlement = "Rural-LV1-101-2034", experiment = "BaseScenario", time_horizon = "0101-3112", scale_NP_EC_based_on_LV = False):
    """
    This function prepares the training, validation and test data for the LSTM model for three different targets: LV, EC, NP.
    It reads the raw data from the BIFROST experiment, scales the data and saves it to the correct path.
    Additionally, it saves the scaling factors for each target to a json file (scaling is done with Min-Max normalization of training data).
    """
    if scale_NP_EC_based_on_LV:
        additional_info = "_by_LV"
    else:
        additional_info = ""

    experiment_main_path = os.path.join('./experiments/',settlement,experiment,time_horizon)

    experiment_data_path = os.path.join(experiment_main_path,"data")
    lstm_data_path = os.path.join(experiment_main_path,"analysis","lstm","train_test_data")

    if os.path.exists(lstm_data_path)==False:
        os.makedirs(lstm_data_path) 

    measurement_data_path = f'{experiment_data_path}/grid_measurements.csv'
    weather_data_path     = f'{experiment_data_path}/weather_data.csv'
    sun_data_path         = f'{experiment_data_path}/sun_data.csv'


    # Start dataset loading
    df_measurements = pd.read_csv(measurement_data_path, low_memory=False, index_col=0)
    df_weather = pd.read_csv(weather_data_path, low_memory=False, index_col=0)
    df_sun = pd.read_csv(sun_data_path, low_memory=False, index_col=0)
    
    # Convert to datetime (all data is UTC time zone but sun behaves like CET time (NOT CEST))
    df_weather.index = pd.to_datetime(df_weather.index, utc=True) 
    df_measurements.index = pd.to_datetime(df_measurements.index, utc=True) 
    df_sun.index = pd.to_datetime(df_sun.index, utc=True) 

    # get keys for the columns in the raw data
    altitude_key = next(filter(lambda col:'SUN-ALTITUDE' in col, df_sun.columns))
    azimuth_key = next(filter(lambda col:'SUN-AZIMUTH' in col, df_sun.columns))
    keys = {}
    keys["LV"] = next(filter(lambda col:'LV-TRANSFORMER:' in col and 'ACTIVE-POWER-3P' in col, df_measurements.columns))
    keys["EC"] = next(filter(lambda col:'EC-TOTAL-POWER' in col and 'REC:' in col, df_measurements.columns))
    keys["NP"] = next(filter(lambda col:'LV-TRANSFORMER:' in col and 'NP-TOTAL-POWER:' in col, df_measurements.columns))

    # save data for each target key
    for key in keys.keys():
        #print(key)
        
        # prepare pd df for each target
        year_df = pd.DataFrame(columns=[key, 'sun_altitude', 'sun_azimuth', 'irradiance_real', 'irradiance_fc'])
        
        year_df[key]               = df_measurements[keys[key]] 
        year_df['irradiance_real'] = df_weather[year_df.index[0]: year_df.index[-1]]['real']
        year_df['irradiance_fc']   = df_weather[year_df.index[0]: year_df.index[-1]]['fc']
        year_df['sun_altitude']    = df_sun[year_df.index[0]: year_df.index[-1]][altitude_key].values
        year_df['sun_azimuth']     = df_sun[year_df.index[0]: year_df.index[-1]][azimuth_key].values

        # SCALING of Irradiance Forecast
        #df_combinded['irradiance_fc']  = df_combinded['irradiance_fc'] * 1.2 #ATTENTION

        # Save first 6 months as training data

        train_df =  year_df['2021-01-01':'2021-07-01']
        train_df.to_csv(os.path.join(lstm_data_path,'{}_train.csv'.format(key)))
        
        target_columns = list(train_df.columns)
        df_train_scaled = pd.DataFrame(columns = target_columns)
    
        #Min-Max normalization
        if key == 'LV' and scale_NP_EC_based_on_LV:
            # Use LV scaling dict for NP and EC
            scaling_dictionary = {}
            for column in target_columns:
                max_val = train_df.loc[:, column].values.max()
                min_val = train_df.loc[:, column].values.min()
                scaling_dictionary[column] = {'max' : max_val, 'min': min_val}

        elif not scale_NP_EC_based_on_LV:
            scaling_dictionary = {}
            for column in target_columns:
                max_val = train_df.loc[:, column].values.max()
                min_val = train_df.loc[:, column].values.min()
                scaling_dictionary[column] = {'max' : max_val, 'min': min_val}

        for i, column in enumerate(target_columns):
            if scale_NP_EC_based_on_LV and i==0:
                max_val = scaling_dictionary['LV']['max']
                min_val = scaling_dictionary['LV']['min']
            else:
                max_val = scaling_dictionary[column]['max']
                min_val = scaling_dictionary[column]['min']
            df_train_scaled[column] = ((train_df[column].values - min_val) / (max_val - min_val))
        df_train_scaled.index = train_df.index

        # save scaled training data
        df_train_scaled.to_csv(os.path.join(lstm_data_path,f'{key}_scaled{additional_info}_train.csv'))

        # Write Scaling factors:
        scaling_factors_path = os.path.join(lstm_data_path, f'{key}_scalings{additional_info}.json')
        with open(scaling_factors_path, 'w') as file:
            json.dump(scaling_dictionary, file)


        # Validation and test are split in 15 day segments for validation/testing:
        # Rest in 15 day chunks, split equally one by one to validation/test

        testvalid_df = year_df['2021-07-01':'2022-01-01']  

        index = 0  
        
        # Iterate over each month in the testvalid_df  
        for month in pd.date_range('2021-07-01', '2022-01-01', freq='MS'):  
        
            # Split the month into validation, test, and remaining segments  
            start_date = month.strftime('%Y-%m-%d')  
            end_date = (month + pd.Timedelta(days=14)).strftime('%Y-%m-%d')  
            val_df = testvalid_df[start_date:end_date]  
            
            start_date = end_date  
            end_date = (month + pd.Timedelta(days=29)).strftime('%Y-%m-%d')  
            test_df = testvalid_df[start_date:end_date]  
            

            val_df.to_csv(os.path.join(lstm_data_path,f'{key}_val{index}.csv'))  
            test_df.to_csv(os.path.join(lstm_data_path,f'{key}_test{index}.csv'))  
        
            # scale val and test values using the scaling factors from the training data
            df_test_scaled = pd.DataFrame(columns = target_columns)
            df_valid_scaled = pd.DataFrame(columns = target_columns) 

            for i, column in enumerate(target_columns):
                if scale_NP_EC_based_on_LV and i==0:
                    max_val = scaling_dictionary['LV']['max']
                    min_val = scaling_dictionary['LV']['min']
                else:
                    max_val = scaling_dictionary[column]['max']
                    min_val = scaling_dictionary[column]['min']
                df_valid_scaled[column] = ((val_df[column].values - min_val) / (max_val - min_val))
            df_valid_scaled.index = val_df.index
            
            for i, column in enumerate(target_columns):
                if scale_NP_EC_based_on_LV and i==0:
                    max_val = scaling_dictionary['LV']['max']
                    min_val = scaling_dictionary['LV']['min']
                else:
                    max_val = scaling_dictionary[column]['max']
                    min_val = scaling_dictionary[column]['min']
                df_test_scaled[column] = ((test_df[column].values - min_val) / (max_val - min_val))
            df_test_scaled.index = test_df.index
            
            # save scaled validation and test data
            df_valid_scaled.to_csv(os.path.join(lstm_data_path, f'{key}_scaled{additional_info}_val{index}.csv'))
            df_test_scaled.to_csv(os.path.join(lstm_data_path, f'{key}_scaled{additional_info}_test{index}.csv'))
                
            index += 1
    

if __name__ == '__main__':

    #os.chdir(os.path.dirname(__file__))

    settlement = "Rural-LV1-101-2034"
    experiment = "BaseScenario"
    time_horizon = "0101-3112"
    scale_NP_EC_based_on_LV = True # If True scaling parameters are used from LV for NP and EC

    prepare_lstm_data(settlement, experiment, time_horizon, scale_NP_EC_based_on_LV)