"""Data-loader builders for the ECG VAE (12-lead / 1-lead / 1-to-12).

    - generate_12_loader: build & save train/val/test loaders for the
      12-lead pipeline
    - generate_1_loader: same for the single-lead pipeline
    - generate_12_1_loader: loaders for the 1-to-12 cross-lead task

    Loaders are torch-saved to data/ and re-used by Trainer_12.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from utils.helper_data_split import (
    train_test_val_split,   
    run_simple_split,
    scale_all,
)

def generate_12_loader(cv=0, batch_size=1000, labels=['ncvd', 'pci_cabg'], binary=False, first_lead_only=False, all=True):
    """Build & save (torch) data loaders for the 12-lead LVF task.

    Loads the "all" prediction/recon pickles, derives the patient-level
    test set, splits train/val, z-scales with a train-fitted scaler,
    and torch-saves a recon + predict DataLoader pair per split.
    """
    
    data_pred_all = pd.read_pickle('data/predict_set_12_lvf_all_AS.p').rename(columns= {'lv_function':'lvf'}).dropna(subset=labels[0]).dropna(subset=['qrs']).reset_index(drop=True)
    data_pred_all = data_pred_all[data_pred_all['qrs'].apply(lambda x: type(x) == np.ndarray and x.shape == (12, 400))]
    data_recon_all = pd.read_pickle('data/recon_set_12_lvf_all_AS.p').dropna(subset=['qrs']).reset_index(drop=True)
    data_recon_all = data_recon_all[data_recon_all['qrs'].apply(lambda x: type(x) == np.ndarray and x.shape == (12, 400))]
    
    print(data_recon_all['PatientID'].nunique())
    print(data_recon_all['Date'].nunique())
    
    test_p = pd.read_pickle('data/pred_set_12_lvf_0_test2_matched.p').dropna(subset=labels[0]).dropna(subset=['qrs']).reset_index(drop=True)
    test_r = pd.read_pickle('data/recon_set_12_lvf_0_test2.p').dropna(subset=['qrs']).reset_index(drop=True)
    
    test_p2 = data_pred_all.query('PatientID in @test_p.PatientID.unique()').query('Date in @test_p.Date.unique()').dropna(subset=labels[0]).dropna(subset=['qrs']).reset_index(drop=True)
    test_p2 = test_p2[test_p2['qrs'].apply(lambda x: type(x) == np.ndarray and x.shape == (12, 400))]

    test_p2['qrs0'] = test_p2['qrs'].apply(lambda x: x[0][0])
    test_p2['qrs10'] = test_p2['qrs'].apply(lambda x: x[0][10])
    test_p2.drop_duplicates(subset=['PatientID', 'Date', 'qrs0', 'qrs10'], inplace=True)
    test_p = test_p2
    
    test_r2 = data_recon_all.query('PatientID in @test_r.PatientID.unique()').query('Date in @test_r.Date.unique()').query('Date not in @test_p.Date.unique()').dropna(subset=['qrs']).reset_index(drop=True)
    test_r2 = test_r2[test_r2['qrs'].apply(lambda x: type(x) == np.ndarray and x.shape == (12, 400))]
    test_r2['qrs0'] = test_r2['qrs'].apply(lambda x: x[0][0])
    test_r2['qrs10'] = test_r2['qrs'].apply(lambda x: x[0][10])
    test_r2.drop_duplicates(subset=['PatientID', 'Date', 'qrs0', 'qrs10'], inplace=True)
    test_r = test_r2
        
    print('Stats after removing test set:')
    print('Pred set: id, data, shape')
    print(test_p['PatientID'].nunique())
    print(test_p['Date'].nunique())
    print(test_p.shape)
    print('Recon set:')
    print(test_r['PatientID'].nunique(), test_r['Date'].nunique(), test_r.shape)
    

    if all:
        data_pred = data_pred_all.query('PatientID not in @test_p.PatientID.unique()')
        data_recon = data_recon_all.query('PatientID not in @test_r.PatientID.unique()')
        
        print('Stats after removing test set:')
        print('Pred set: id, data, shape')
        print(data_pred['PatientID'].nunique())
        print(data_pred['Date'].nunique())
        print(data_pred.shape)
        print('Recon set:')
        print(data_recon['PatientID'].nunique(), data_recon['Date'].nunique(), data_recon.shape)
    else:
        data_pred = pd.read_pickle('data/pred_set_12_lvf_0_train.p').dropna(subset=labels[0]).dropna(subset=['qrs']).reset_index(drop=True)
        data_recon = pd.read_pickle('data/recon_set_12_lvf_0_train.p').dropna(subset=['qrs']).reset_index(drop=True)
        
        data_pred2 = data_pred_all.query('PatientID in @data_pred.PatientID.unique()').query('Date in @data_pred.Date.unique()').dropna(subset='lvf').dropna(subset=['qrs']).reset_index(drop=True)
        data_pred2 = data_pred2[data_pred2['qrs'].apply(lambda x: type(x) == np.ndarray and x.shape == (12, 400))]
        data_pred2['qrs0'] = data_pred2['qrs'].apply(lambda x: x[0][0])
        data_pred2['qrs10'] = data_pred2['qrs'].apply(lambda x: x[0][10])
        data_pred2.drop_duplicates(subset=['PatientID', 'Date', 'qrs0', 'qrs10'], inplace=True)
        data_pred = data_pred2
        data_pred = data_pred.query('PatientID not in @test_p.PatientID.unique()')
        
        data_recon2 = data_recon_all.query('PatientID in @data_pred.PatientID.unique()').query('Date in @data_pred.Date.unique()').dropna(subset=['qrs']).reset_index(drop=True)
        data_recon2 = data_recon2[data_recon2['qrs'].apply(lambda x: type(x) == np.ndarray and x.shape == (12, 400))]
        data_recon2['qrs0'] = data_recon2['qrs'].apply(lambda x: x[0][0])
        data_recon2['qrs10'] = data_recon2['qrs'].apply(lambda x: x[0][10])
        data_recon2.drop_duplicates(subset=['PatientID', 'Date', 'qrs0', 'qrs10'], inplace=True)
        data_recon = data_recon2
        data_recon = data_recon.query('PatientID not in @test_r.PatientID.unique()')
        
        
        print('Stats after removing test set:')
        print('Pred set: id, data, shape')
        print(data_pred['PatientID'].nunique())
        print(data_pred['Date'].nunique())
        print(data_pred.shape)
        print('Recon set:')
        print(data_recon['PatientID'].nunique(), data_recon['Date'].nunique(), data_recon.shape)
        
    train_p, train_r, val_p, val_r = train_test_val_split(data_pred, data_recon, nr_splits=1, complete=False, random_state=cv, mi=False, a112=True)
    
    train_p, tests_p, _ = scale_all(train_p, [val_p[0], test_p], 2)
    val_p = tests_p[0]
    test_p = tests_p[1]
    
    train_r, tests_r, _ = scale_all(train_r, [val_r[0], test_r], 2)
    val_r = tests_r[0]
    test_r = tests_r[1]
    
    
    for pre, set_p, set_r in [('val', val_p, val_r), ('train', train_p, train_r), ('test', test_p, test_r)]:        
        set_r = set_r[set_r['qrs'].apply(lambda x: type(x) == np.ndarray and x.shape == (12, 400))]
        set_p = set_p[set_p['qrs'].apply(lambda x: type(x) == np.ndarray and x.shape == (12, 400))]
        if first_lead_only:
            set_r = torch.tensor(np.stack(set_r['qrs'].values))[:, 0, :].unsqueeze(1).expand(-1, 12, -1)
            set_p_r = torch.tensor(np.stack(set_p['qrs'].values))[:, 0, :].unsqueeze(1).expand(-1, 12, -1)
        else:
            set_r = torch.tensor(np.stack(set_r['qrs'].values))
            set_p_r = torch.tensor(np.stack(set_p['qrs'].values))
        if binary:
            set_p_p = torch.tensor(np.stack(set_p[labels[0]].values>1).astype(int))
        else:
            set_p_p = torch.tensor(np.stack(set_p[labels[0]].values))
        set_p_pi = torch.tensor(np.stack(set_p['PatientID'].values))
        set_p_index = torch.tensor(np.stack(set_p.index.values))
        
        loader_r = DataLoader(set_r, batch_size=batch_size, shuffle=True)
        data_p = TensorDataset(set_p_r, set_p_p, set_p_pi, set_p_index)
        loader_p = DataLoader(data_p, batch_size=batch_size, shuffle=True)   
             
        torch.save(loader_r, f'data/{pre}_loaderAS_12_{labels[0]}_{cv}{binary}_{first_lead_only}_{all}all1of12.pt')
        torch.save(loader_p, f'data/{pre}_loaderAS_12_{labels[0]}_{cv}{binary}_{first_lead_only}_{all}all1of12_p.pt')
     
     
def generate_1_loader(cv=0, batch_size=1000, labels=['ncvd', 'pci_cabg'], binary=False):
    """Build & save data loaders for the single-lead LVF task.

    Same flow as generate_12_loader but on the 1-lead data; the single
    lead is expanded (unsqueeze/expand) to (B, 12, 400) so the model
    sees the same shape as the 12-lead case.
    """
    
    data_pred = pd.read_pickle('data/pred_set_1_lvf_0_train2.p').dropna(subset=labels[0]).reset_index(drop=True)
    data_recon = pd.read_pickle('data/recon_set_1_lvf_0_train2.p').reset_index(drop=True)

    test_p = pd.read_pickle('data/pred_set_1_lvf_0_test2_matched.p').dropna(subset=labels[0]).reset_index(drop=True)
    test_r = pd.read_pickle('data/recon_set_1_lvf_0_test2.p').reset_index(drop=True)

    data_pred = data_pred.query('PatientUniqueID not in @test_p.PatientUniqueID.unique()')
    data_recon = data_recon.query('PatientUniqueID not in @test_r.PatientUniqueID.unique()')
    
    print('test_p', len(test_p))
    print('test_r', len(test_r))
    print('train_p', len(data_pred))
    print('train_r', len(data_recon))
    
      
    print('Stats after removing test set:')
    print('Pred set: id, data, shape')
    print(data_pred['PatientUniqueID'].nunique())
    print(data_pred['1lead_date'].nunique())
    print(data_pred.shape)
    print('Recon set:')
    print(data_recon['PatientUniqueID'].nunique(), data_recon['1lead_date'].nunique(), data_recon.shape)
    
         
    print('Stats after removing test set:')
    print('Pred set: id, data, shape')
    print(test_p['PatientUniqueID'].nunique())
    print(test_p['1lead_date'].nunique())
    print(test_p.shape)
    print('Recon set:')
    print(test_r['PatientUniqueID'].nunique(), test_r['1lead_date'].nunique(), test_r.shape)
    
    # if cv != 0:
    train_p, train_r, val_p, val_r = train_test_val_split(data_pred, data_recon, nr_splits=1, complete=False, random_state=cv, mi=False, a112=True)
    
    train_p, tests_p, scaler = scale_all(train_p, [val_p[0], test_p], 2, withings=True, a112=True)
    val_p = tests_p[0]
    test_p = tests_p[1]
    
    train_r, tests_r, _ = scale_all(train_r, [val_r[0], test_r], 2, withings=True, a112=True)
    val_r = tests_r[0]
    test_r = tests_r[1]
    
    for pre, set_p, set_r in [('val', val_p, val_r), ('train', train_p, train_r), ('test', test_p, test_r)]:        
        set_r = set_r[set_r['qrsw'].apply(lambda x: type(x) == np.ndarray and x.shape == (400,))]
        set_p = set_p[set_p['qrsw'].apply(lambda x: type(x) == np.ndarray and x.shape == (400,))]
        set_r = torch.tensor(np.stack(set_r['qrsw'].values))
        set_p_r = torch.tensor(np.stack(set_p['qrsw'].values))
        if binary:
            set_p_p = torch.tensor(np.stack(set_p[labels].values>1).astype(int)).long()
        else:
            set_p_p = torch.tensor(np.stack(set_p[labels].values))
        set_p_pi = torch.tensor(np.stack(set_p['PatientUniqueID'].values))
        set_p_index = torch.tensor(np.stack(set_p.index.values))
        
        # expand from 1,400 to 1,12,400, and copy 12x
        set_r = set_r.unsqueeze(1).expand(-1, 12, -1)
        set_p_r = set_p_r.unsqueeze(1).expand(-1, 12, -1)
        # set_p_p = set_p_p.unsqueeze(1).expand(-1, 12, -1)
         
        # set_r = set_r.unsqueeze(1).expand(-1, 12, -1)
            
        loader_r = DataLoader(set_r, batch_size=batch_size, shuffle=True)
        data_p = TensorDataset(set_p_r, set_p_p, set_p_pi, set_p_index)
        loader_p = DataLoader(data_p, batch_size=batch_size, shuffle=True)   
             
        torch.save(loader_r, f'data/{pre}_loaderAS_1_{labels[0]}_{cv}{binary}_False_Falseall1of12.pt')
        torch.save(loader_p, f'data/{pre}_loaderAS_1_{labels[0]}_{cv}{binary}_False_Falseall1of12_p.pt')