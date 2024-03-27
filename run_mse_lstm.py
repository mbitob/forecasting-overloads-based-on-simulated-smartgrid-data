import os, sys
import argparse

from tools.prepare_lstm_input_data import prepare_lstm_data
from tools.lstm_functions_mse import lstm_mse_train, lstm_mse_forecast


#####################################################
# define the arguments

parser = argparse.ArgumentParser()

# Experiment selection
parser.add_argument('--settlement',             type=str,   default = "Rural-LV1-101-2034",                 help='settlement name'                                                          )
parser.add_argument('--experiment',             type=str,   default = "BaseScenario_FCInterpolLin",         help='experiment name'                                                          )
parser.add_argument('--time_horizon',           type=str,   default = "0101-3112",                          help='time horizon'                                                             )

# LSTM parameters
parser.add_argument('--device',                 type=str,   default = 'cuda:0',                             help='cuda device'                                                              )
parser.add_argument('--batch_size',             type=int,   default = 64,                                   help='total batchsize for train and test'                                       )
parser.add_argument('--epochs',                 type=int,   default = 100,                                  help='epoch number'                                                             )
parser.add_argument('--lr',                     type=float, default = 1e-3,                                 help='learning rate'                                                            )
parser.add_argument('--lookback',               type=int,   default = 100,                                  help='look back of network'                                                     )

# Experiment variations
parser.add_argument('--n_inputs',               type=int,   default = 4,                    choices=[4],    help="currently only 4 is supported: Power/irradiance/altitude/weather(real/fc)")
parser.add_argument('--targets',                type=str,   default = ["NP","EC","LV"],                     help="xxx"                                         )
#parser.add_argument('--n_outputs',              type=int,   default = 1,                    choices=[3,1],  help="3 for LV,EC an NP, 1 for only LV"                                         )
parser.add_argument('--use_irradiance_real',    type=bool,  default = True,                                 help ="False -> forecasted irradiance, True -> real irradiance for training"    )
parser.add_argument('--multi_out',              type=bool,  default = True,                                help="xxx"                                         )

args = parser.parse_args()

#####################################################
# prepare configurations

if args.use_irradiance_real:
    additional_info = "_real_irradiance" 
else:  
    additional_info = "_forecast_irradiance"

if args.multi_out:
    args.n_outputs                  = 3
    args.analysis_name              = "MSE_multi_EC_NP_LV"+additional_info
    args.scale_NP_EC_based_on_LV    = True
else:
    args.n_outputs                  = 1
    args.scale_NP_EC_based_on_LV    = False


###################################################
#prepare data
print("Prepare data")

prepare_lstm_data(args.settlement, args.experiment, args.time_horizon, args.scale_NP_EC_based_on_LV)

####################################################
#train the model

if args.multi_out:
   print("Train model with multi output")
   lstm_mse_train(args)
else:
   for target in args.targets:
       print(f"Train model with single output {target}")
       args.analysis_name  = f"MSE_single_{target}{additional_info}"
       args.target         = target
       lstm_mse_train(args)

####################################################
# Evaluate forecasting
print("Evaluate forecasting")
if args.multi_out:
    print("Eval  multi output")
    lstm_mse_forecast(args)
else:
    for target in args.targets:
        print(f"Eval single output {target}")
        args.analysis_name  = f"MSE_single_{target}{additional_info}"
        args.target         = target
        lstm_mse_forecast(args)
