#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator

import sys
import itertools
import os
import socket
import copy
import random
from collections import defaultdict
from typing import Type
import argparse
import pickle

import argparse
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
import torch.distributed as dist
from torch.multiprocessing import Process

from torchmetrics.functional import r2_score as r2Functional

random.seed(42)

class SequenceDataset(Dataset):
    def __init__(self, dataframe, targets, features_conc,features_conc_bool, sequence_length,drop_heights=None,features_meteo = [],features_meteo_bool = [],start_buffer : int = 24*6*3,drop_range : tuple = (6*24*1,6*24*7)):
        if start_buffer > sequence_length:
            raise ValueError("buffer longer than sequence")
        elif start_buffer+drop_range[1] > sequence_length:
            raise ValueError("buffer cannot be guaranteed, second part of drop_range too large")
        else:
            self.start_buffer=start_buffer

        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[targets].values).float()
        self.X_c = torch.tensor(dataframe[features_conc].values).float()
        self.X_c_b = torch.tensor(dataframe[features_conc_bool].values).float()
        self.drop_range = drop_range
        self.features_conc = features_conc
        self.features_conc_bool = features_conc_bool
        self.drop_heights = drop_heights
        if features_meteo:
            self.X_m = torch.tensor(dataframe[features_meteo].values).float()
            self.X_m_b = torch.tensor(dataframe[features_meteo_bool].values).float()
        self.features_meteo = features_meteo
        self.features_meteo_bool = features_meteo_bool

    def __len__(self):
        return self.X_c.shape[0]

    def __getitem__(self, i):
        drop_length = random.randint(*self.drop_range)    
        if i >= self.sequence_length - 1:
            if not isinstance(self.drop_heights, type(None)):
                self.X_c[i-drop_length+1:i+1,self.drop_heights] = -1
                self.X_c_b[i-drop_length+1:i+1,self.drop_heights] = 1


            x_c = self.X_c[i-sequence_length+1:i+1]
            x_c_b = self.X_c_b[i-sequence_length+1:i+1]

            if self.features_meteo:
                x_m = self.X_m[i-sequence_length+1:i+1]
                x_m_b = self.X_m_b[i-sequence_length+1:i+1]
                x = torch.cat((x_m,x_c,x_m_b,x_c_b),1)
            else:
                x = torch.cat((x_c,x_c_b),1)
        else:
            padding_c = self.X_c[0].repeat(self.sequence_length - i - 1, 1)
            padding_c_b = self.X_c_b[0].repeat(self.sequence_length - i - 1, 1)
            
            x_c = torch.cat((padding_c, self.X_c),0)
            x_c_b = torch.cat((padding_c_b, self.X_c_b),0)
            j = sequence_length -1 
            x_c[j-drop_length+1:j+1] = -1
            x_c_b[j-drop_length+1:j+1] = 1

            x_c = self.X_c[:j+1]
            x_c_b = self.X_c_b[:j+1]

            if self.features_meteo:
                padding_m = self.X_m[0].repeat(self.sequence_length - i - 1, 1)
                padding_m_b = self.X_m_b[0].repeat(self.sequence_length -i -1, 1)
                x_m = torch.cat((padding_m, self.X_m),0)
                x_m_b = torch.cat((padding_m_b, self.X_m_b),0)
                x_m = self.X_m[:j+1]
                x_m_b = self.X_m[:j+1]
                x = torch.cat((x_m,x_c,x_m_b,x_c_b),1)
            else:
                x = torch.cat((x_c,x_c_b),1)
        return x, self.y[i]


class DeepRegressionLSTM(nn.Module):

    def __init__(self, input_size, hidden_size,num_layers,seq_length,dropout_val,num_classes = 18):
        super().__init__()
        self.input_size = input_size  
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.dropout_val = dropout_val

        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout = dropout_val
            
        )

        self.fc = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,device=x.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,device = x.device).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0,c0))
        out = self.fc(out)

        return out
    
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_loss = float('-inf')

    def early_stop(self, validation_loss):
        if validation_loss == -1000:
            return True
        
        elif validation_loss > self.max_validation_loss:
            self.max_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss <= (self.max_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def initData(data,features,target_sensors,forecast_lead=1):
    df = data[features].copy()

    forecast_lead = 1
    targets = []
    for target_sensor in target_sensors:
        target = f"{target_sensor}_lead{forecast_lead}"
        targets.append(target)
        df[target] = data[target_sensor].shift(-forecast_lead)
    df = df.iloc[:-forecast_lead]
    minmaxDict = {}
    if args.min_max:
        for c in df.columns:
            minmaxDict[c] = (df[c].min(),df[c].max())
            df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())

    boolDict = {}
    for c in features:
        fillValue = -1
        boolDict[c+"_Bool"]=df[c].isna().astype(int)
        df[c]= df[c].fillna(fillValue)
    df = pd.concat([df,pd.DataFrame(boolDict)],axis = 1)

    features_mask = [(f+"_Bool") for f in features]
    features = list(features)+list(features_mask)


    return {"features" : features, "targets" : targets,"minmaxDict" : minmaxDict,"data" : df}


def initSplit(df,targets,features,sequence_length,batch_size,split,validation_start = "2002-03-27 8:50:00",test_start = "2003-03-27 8:50:00"):
    match split:
        case "2Year":
            df_train = df.loc["2001-03-27 8:50:00":validation_start].copy()
            df_validate = df.loc[validation_start:test_start].copy()
        case "2nd":
            df_train = df.loc["2001-03-27 8:50:00":test_start:2].copy()
            df_validate = df.loc["2001-03-27 9:00:00":test_start:2].copy()
        case "1Year":
            df_train = df.loc["2001-03-27 8:50:00":"2001-09-27 8:50:00"].copy()
            df_validate = df.loc["2001-09-27 9:00:00" :validation_start].copy()
        case "8020":
            df_train = df.loc[:"2002-11-7 8:50:00"].copy()
            df_validate = df.loc["2002-11-7 8:50:00":test_start].copy()
        case "Identical":
            df_train = df.loc["2001-03-27 8:50:00":test_start].copy()
            df_validate = df.loc["2001-03-27 8:50:00":test_start].copy()
        case "Middle":
            df_train = df.loc[:validation_start].copy() + df.loc["2002-09-27 8:50:00":test_start].copy()
            df_validate = df.loc[validation_start:"2002-09-27 8:50:00"].copy()
        case "5th":
            raise NotImplementedError("havent done 5th yet")
        case _:
            raise IOError("no split selection found")

    df_test = df.loc[test_start:].copy()
    
    features_conc = [f for f in features if "Avg" in f]
    f_c = [f for f in features_conc if "Bool" not in f]
    f_c_b =[f for f in features_conc if "Bool" in f]
    print(f_c)
    features_meteo = [f for f in features if "Avg" not in f]
    f_m = [f for f in features_meteo if "Bool" not in f]
    f_m_b = [f for f in features_meteo if "Bool" in f]

    sequence_length = sequence_length
    batch_size = batch_size
    drop_heights = args.drop_heights
    random.seed(42)
    if drop_heights:
        drop_list = np.sort(np.random.choice(np.arange(0,len(f_c)),drop_heights,replace=False))
    else:
        drop_list = []
    drop_max =args.drop_max_days
    drop_min = args.drop_min_days
    if not drop_max:
        drop_max = 3
    if not drop_min:
        drop_min = 3
    print(f"dropping {drop_heights} heights, list is {drop_list}")
    print(f"dropping {drop_max} days")
    train_dataset = SequenceDataset(
        df_train,
        targets=targets,
        features_meteo=f_m,
        features_meteo_bool= f_m_b,
        features_conc = f_c,
        features_conc_bool = f_c_b,
        sequence_length=sequence_length,
        drop_range=(6*24*drop_min,6*24*drop_max),
        drop_heights=drop_list
    )

    test_dataset = SequenceDataset(
        df_test,
        targets=targets,
        features_meteo=f_m,
        features_meteo_bool = f_m_b,
        features_conc = f_c,
        features_conc_bool = f_c_b,
        sequence_length=sequence_length,
        drop_range=(6*24*drop_min,6*24*drop_max),
        drop_heights=drop_list
    )

    validation_dataset = SequenceDataset(
        df_validate,
        targets=targets,
        features_meteo=f_m,
        features_meteo_bool = f_m_b,
        features_conc = f_c,
        features_conc_bool = f_c_b,
        sequence_length=sequence_length,
        drop_range=(6*24*drop_min,6*24*drop_max),
        drop_heights=drop_list
    )

    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)#,sampler=validation_dist_sampler)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)#,sampler = train_dist_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)#,sampler = test_dist_sampler)

    return {"test_dataset" : test_dataset,"df_test" :df_test,"df_train" : df_train,"df_validate" : df_validate,"validation_loader":validation_loader, "train_loader" : train_loader, "test_loader": test_loader,}


def train_model(data_loader, model,seqType, loss_function, optimizer = "adam"):
    lossDivisor = 0
    total_loss = 0
    nanCount = 0
    model.train()
    for X, y in data_loader:

        X = X
        y = y
        output = model(X)
        if args.metric =="r2_score":
            if seqType == "Lst":
                loss = loss_function(output[:,-1,:].permute(1,0), y.permute(1,0))
            elif seqType =="Full":
                for i in range(144,output.shape[1]):
                    loss = loss_function(output[:,i,:].permute(1,0), y.permute(1,0))
            else:
                raise IOError("No SeqType given")
            
        elif args.metric =="mse":
            if seqType =="Lst":
                loss = loss_function(output[:,-1,:], y)
            else:
                for i in range(144,output.shape[1]):
                    loss = loss_function(output[:,i,:], y)
        else:
            print("no metric")

        if not torch.isnan(loss):
            total_loss+=loss.item()
            lossDivisor +=1
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()

    if total_loss == 0:
        if args.metric =="r2_score":
            return -1000,-10000000
        else:
            return 1000,1000000
    avg_loss = total_loss / lossDivisor
    return avg_loss,total_loss

def validate_model(data_loader, model, loss_function):
    lossDivisor = 0
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            X = X
            y = y
            output = model(X)
            if args.metric == "r2_score":
                loss = loss_function(output[:,-1,:].permute(1,0), y.permute(1,0))
            elif args.metric =="mse":
                loss = loss_function(output[:,-1,:], y)
            else:
                print("no good metric, how dis happun?!")
            if not torch.isnan(loss):
                total_loss += loss.item()
                lossDivisor+=1
        if total_loss == 0:
            if args.metric == "r2_score":
                return -1000,-1000000000
            else:
                return 1000,1000000
        avg_loss = total_loss / lossDivisor
        return avg_loss,total_loss


def predict(data_loader, model,save=None):
    if save:
        modelPath = "../Output/Models/"
        model.load_state_dict(torch.load(modelPath+save+"_best-model.pt",map_location={'cuda:0': 'cpu'}))
    output = torch.tensor([])
    model.eval()
    mses = []
    lossl2 = nn.MSELoss()
    with torch.no_grad():
        for X, y in data_loader:
            X = X
            y_star = model(X)
            heights = []
            for i in range(18):
                heights.append(lossl2(y_star[:,-1,i],y[:,i]))
            mses.append(heights)
            
    return mses


def do_training(model,optimizer,validation_loader,train_loader,loss_function,early_stopping,save= None,load="best",epochs = 60,log_wandb = True):

    best_model = model.state_dict()
    best_optim = optimizer.state_dict()


    train_avg_losses,validation_avg_losses = [],[]
    if args.metric =="r2_score":
        best_validation_loss = -1000 
        best_train_loss = -1000
    else:
        best_validation_loss = 1000 
        best_train_loss = 1000
    early_stopper = EarlyStopper(patience=early_stopping)
    for ix_epoch in range(epochs):
        train_avg_loss,train_loss = train_model(train_loader, model, optimizer=optimizer,loss_function=loss_function,seqType=args.seqType)
        train_avg_losses.append(train_avg_loss)
        
        validation_avg_loss,validation_loss = validate_model(validation_loader, model, loss_function=loss_function)
        validation_avg_losses.append(validation_avg_loss)

        if ix_epoch % 1 == 0:
            print(f"Epoch {ix_epoch}\n---------")

        if args.metric == "r2_score":
            comp = operator.gt
        else:
            comp = operator.lt
        if comp(validation_avg_loss, best_validation_loss):
            best_validation_loss = validation_avg_loss
            if save:
                best_model = model.state_dict()#model.module.state_dict()
                best_optim = optimizer.state_dict()
                torch.save(best_model,"../Output/Models/"+save+"_best-model.pt")
                torch.save(best_optim,"../Output/Models/"+save+"_best-optimizer.pt")
                print(f"saved best model at epoch {ix_epoch}")
        if comp(train_avg_loss, best_train_loss):
            best_train_loss = train_avg_loss

        print(f"Train_loss {train_avg_loss}, Validation_loss {validation_avg_loss}, best_train_loss {best_train_loss} best_validation_loss {best_validation_loss}")
        print()
        if early_stopper.early_stop(validation_avg_loss):             
            print("-----Stopping now-----")
            return train_avg_losses,validation_avg_losses,best_model
    return train_avg_losses,validation_avg_losses,best_model

def do_validating(rank,model,data,loss_function = r2Functional):
    validation_avg_loss,validation_loss = validate_model(rank,data,model,loss_function = loss_function)
    return validation_avg_loss,validation_loss


parser = argparse.ArgumentParser()
parser.add_argument('--index', type = int,default=52)
parser.add_argument('dataset', choices=["All", "Conc"])
parser.add_argument('--split', choices = ["2Year","1Year", "2nd","8020","Identical","Middle"],default="8020")
parser.add_argument('--seqType', choices=["Lst", "Full"],default="Lst")
parser.add_argument('--early_stopping',type = int,default=5)
parser.add_argument('--metric',choices = ["mse","mae","r2_score"],default="r2_score")
parser.add_argument('--drop_heights',type=int,default=18)
parser.add_argument('--drop_max_days', type=int)
parser.add_argument('--drop_min_days', type = int)
parser.add_argument('--drop_meteo',type=int)
parser.add_argument('--save_name', type=str)
parser.add_argument('--keep_meteo',type=int)
parser.add_argument("--min_max",type=bool,default=True)
parser.add_argument('--do',choices=["train","predict"],default=predict)
parser.add_argument('--feature_selection',type=bool,default=False)

args = parser.parse_args()


selected_meteo = ["TA_44m_degC","VPD_hPa","WD_deg","SW_OUT_Wm-2","LW_IN_Wm-2","LW_OUT_Wm-2","TS_1_50cm_degC","G_1_5cm_Wm-2"]

droppable_meteo = ["TA_44m_degC","PA_hPa","RH_%","VPD_hPa","WD_deg","WS_ms-1","SW_IN_Wm-2","SW_OUT_Wm-2","LW_IN_Wm-2","LW_OUT_Wm-2","NETRAD_Wm-2","P_mm","TS_1_50cm_degC","SWC_1_32cm_%","G_1_5cm_Wm-2"]

data_Concentration = pd.read_csv("../Daten/DE-Hai_profile_10min_qaqc_2001_2003.csv",sep="\t",parse_dates=["TIMESTAMP_END"],index_col=0)
data_Meteo = pd.read_csv("../Daten/DE-Hai_Meteo_all_10min_20010101_20031231.csv",parse_dates = ["TIMESTAMP_END"],index_col = 1)
data_Meteo.drop(columns = ["TIMESTAMP_START"],inplace = True)
data_Concentration = data_Concentration[~data_Concentration.index.duplicated(keep = "first")]
data_Concentration.dropna(how = "any",inplace=True)
data_Meteo = data_Meteo[~data_Meteo.index.duplicated(keep = "first")]

if args.keep_meteo:
    data_Meteo = data_Meteo[droppable_meteo[args.keep_meteo]]
    print(f"only using {data_Meteo.name} column from Meteo")

idx = data_Concentration.index.intersection(data_Meteo.index)
data_Concentration = data_Concentration.loc[idx]
data_Meteo = data_Meteo.loc[idx]
data_All = pd.concat([data_Meteo,data_Concentration],axis  =1)

allDict = initData(data = data_All,features = data_All.columns,target_sensors = data_Concentration.columns)
concDict = initData(data = data_All,features = data_Concentration.columns, target_sensors=data_Concentration.columns)
sequence_length = 6*24*10
batch_size = 16
parameterSetIndices = (0,1)


dataSelection = {"All" : allDict, "Conc" : concDict}
metricSelection = {"mse" : nn.MSELoss(),"mae" : nn.L1Loss(), "r2_score": r2Functional}


if args.drop_meteo:
    data_All.drop(columns = droppable_meteo[args.drop_meteo])
    print("dropped", droppable_meteo[args.drop_meteo])

parameters = {"dropout":  [0.2,0.1,0.05],"hidden_layer_size":[32,128,256],"learn_rate":  [0.05,0.005],"num_layers" :  [2,3],"weight_decay" : [0,0.01]}
def parameterProduct(parameters):
    return(dict(zip(parameters.keys(), x)) for x in itertools.product(*parameters.values()))
parameterSets = list(parameterProduct(parameters))
parameterSet = parameterSets[args.index]
print("using setIndice", args.index, "of", len(parameterSets))
fileName = "".join([(str(f)+"-") for f in parameterSet.values()])
print("fileName is " + fileName)

def runStuff(dictToInit,split,metric = r2Functional,early_stopping=5):

    allDict.update(initSplit(allDict["data"],features = allDict["features"],targets = allDict["targets"],split=split,sequence_length=sequence_length,batch_size=batch_size))
    concDict.update(initSplit(concDict["data"],features = concDict["features"],targets = concDict["targets"],split=split,sequence_length=sequence_length,batch_size=batch_size))
    initDict= dictToInit
 
    model = DeepRegressionLSTM(input_size=len(initDict["features"]), hidden_size=parameterSet["hidden_layer_size"],num_layers=parameterSet["num_layers"],dropout_val=parameterSet["dropout"],seq_length=sequence_length)
   # model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[device_id])
   
    toMaxOptimizer = args.metric == "r2_score"
    print(f"to maximize is {toMaxOptimizer}")
    optimizer = torch.optim.Adam(model.parameters(), lr=parameterSet["learn_rate"],weight_decay = parameterSet["weight_decay"],maximize=toMaxOptimizer)

    device_id = 0
    if args.do == 'train':
        tr,vd,bm = do_training(model,optimizer,validation_loader=initDict["validation_loader"],train_loader=initDict["train_loader"],loss_function=metric,early_stopping=early_stopping,save=args.save_name)
        outputFile = "../Output/Models/Training/"
    else:
        vd = predict(initDict["test_loader"],model=model,save=args.save_name)
        outputFile = "../Output/Models/Predictions/"
    if args.save_name:
        outputFile+=args.save_name
    with open(outputFile, "wb") as fp:
        pickle.dump(vd,fp)
    #dist.destroy_process_group()



if __name__ == "__main__":
#    world_size = int(os.environ['SLURM_NPROCS'])
#    world_rank = int(os.environ['SLURM_PROCID'])
#    ngpus_per_node=torch.cuda.device_count()
#    hostname = socket.gethostname()

    #torch.cuda.set_device(0)
    runStuff(dataSelection[args.dataset],split = args.split,metric=metricSelection[args.metric],early_stopping=args.early_stopping)
#    init_processes(world_rank, world_size,ngpus_per_node, hostname, backend='nccl')

# sweep_configuration = {
#     "method": "bayes",
#     "metric": {
#         "goal": "maximize",
#         "name": "test_avg_loss",
#     },
    # "parameters": {
    #     "dropout": {
    #         "values" : [0.05, 0.005]
    #         },
    #     "hidden_layer_size": {
    #         "values" : [32,128]
    #         },
    #     "learn_rate": {
    #         "values" : [0.05, 0.005]
    #         },
    #     "num_layers" : {
    #         "values" : [2,3]
    #         }
    # }

    
#def init_processes(Myrank, size,ngpus_per_node, hostname, backend='nccl'):
#    """ Initialize the distributed environment. """
#    dist.init_process_group("nccl",rank = Myrank, world_size = size)
#    
#    print("Initialized Rank:", dist.get_rank())
#    hostname = socket.gethostname()
#    ip_address = socket.gethostbyname(hostname)
#    print(ip_address)

#    device_id = Myrank % torch.cuda.device_count()
#    torch.cuda.set_device(Myrank)
#   print("rank is ",Myrank)