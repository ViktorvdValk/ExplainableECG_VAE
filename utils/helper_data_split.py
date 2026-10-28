"""Train/val/test splitting, label completion and scaling of the ECG data.

    - check_pi_overlap: verify patient sets do not leak between splits
    - split_pred / split_full / split_full_ex_prediction: group-wise
      (patient-level) splits
    - complete_pred_*: join ECG feature rows to clinical labels (angio /
      LVF / MI) by patient+date
    - match_withings: match home-monitor (Withings) recordings to
      hospital ECGs by date window + RRI similarity
    - train_test_val_split: the main split orchestrator
    - scale_data / scale_all / scale_save: StandardScaler helpers
    - run_*: scripted pipelines that build the pickled .p datasets
      consumed by the loaders (helper_load.py)
"""

import os
import pickle
import copy
import numpy as np
import pandas as pd
from joblib import dump
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

def check_pi_overlap(train, test, a112=False):
    """Assert no patient appears in both train and test/validation."""

    try:
        pi_train = train.PatientUniqueID.unique()
        pi_test = test.PatientUniqueID.unique()
    except AttributeError:
        pi_train = train.PatientID.unique()
        pi_test = test.PatientID.unique()
    for pi in pi_train:
        assert pi not in pi_test, "PatientID in training set, should not be in test/validation set"


def split_pred(data_in, random_state=42, mi=False, a112=False):
    """Patient-level split of the (labelled) prediction dataset.

    Uses StratifiedGroupKFold and keeps the first train/test pair,
    stratified on the target label (pci / MI_label / sex).
    """
    cv = StratifiedGroupKFold(
        n_splits=6, random_state=random_state, shuffle=True)
    if mi:
        train_idxs1, train_idxs2, _, _, _, _ = cv.split(
            data_in.index.values, data_in.pci_bool.values, data_in.PatientID)
    elif a112:
        if 'MI_label' in data_in.columns:
            train_idxs1, train_idxs2, _, _, _, _ = cv.split(
                data_in.index.values, data_in.MI_label.values, data_in.PatientUniqueID)
        else:
            try:
                train_idxs1, train_idxs2, _, _, _, _ = cv.split(
                    data_in.index.values, data_in.lvf.values.astype('int'), data_in.PatientUniqueID)
            except AttributeError:
                train_idxs1, train_idxs2, _, _, _, _ = cv.split(
                    data_in.index.values, data_in.lvf.values.astype('int'), data_in.PatientID)
    else:
        train_idxs1, train_idxs2, _, _, _, _ = cv.split(
            data_in.index.values, data_in.sex.values, data_in.PatientID)
    
    print(len(train_idxs1[0]), len(train_idxs1[1]))
    print(len(train_idxs2[0]), len(train_idxs2[1]))   
    train = data_in.iloc[train_idxs1[0]]
    test = data_in.iloc[train_idxs1[1]]
    check_pi_overlap(train, test, a112=a112)
    return train, test


def split_full(data, train_size, random_state=42, a112=False):
    """GroupShuffleSplit of the full ECG dataset (train_size / rest)."""

    gss = GroupShuffleSplit(
        n_splits=2, train_size=train_size, random_state=random_state)
    try:
        train_idx, _ = gss.split(data.index, data.index, data.PatientUniqueID)
    except AttributeError:
    	train_idx, _ = gss.split(data.index, data.index, data.PatientID)
    
    train_f = data.iloc[train_idx[0]]
    test_f = data.iloc[train_idx[1]]
    check_pi_overlap(train_f, test_f, a112=a112)
    return train_f, test_f


def split_full_ex_prediction(data, test_p, random_state=42, a112=False):
    """Split the full dataset, excluding patients already reserved for
    the prediction test set (prevents patient leakage).
    """

    try:
        pi_test = test_p.PatientUniqueID.unique()
        p_m = len(test_p.PatientUniqueID) / len(data.PatientUniqueID)   
        test1_f = copy.deepcopy(data.query('PatientUniqueID in @pi_test'))
    except AttributeError:
        pi_test = test_p.PatientID.unique()
        p_m = len(test_p.PatientID) / len(data.PatientID)
        test1_f = copy.deepcopy(data.query('PatientID in @pi_test'))
    data.drop(test1_f.index, inplace=True)
    train_f, test2_f = split_full(data, train_size=0.85, a112=a112)

    test_f = pd.concat([test1_f, test2_f])
    check_pi_overlap(train_f, test_f, a112=a112)
    return train_f, test_f

def complete_pred_angio(df_lvf, df):
    """Join ECG rows to angiography labels (RCA/RDA/RCx, PCI, graft...)."""


    df_lvf.ECG_date = pd.to_datetime(df_lvf.ECG_date)
    df.Date = pd.to_datetime(df.Date)   
    for ind in df.index:
        pi = df.loc[ind, 'PatientID']
        date = df.loc[ind, 'Date']
        if date in df_lvf.query('PatientID == @pi').ECG_date.values:
            row = df_lvf.query(
                '((PatientID == @pi) and (ECG_date == @date))')
            df.at[ind, 'RCA'] = row.RCA.values[0]
            df.at[ind, 'RDA'] = row.RDA.values[0]
            df.at[ind, 'RCx'] = row.RCx.values[0]
            df.at[ind, 'RCA_both'] = row.RCA_both.values[0]
            df.at[ind, 'RDA_both'] = row.RDA_both.values[0]
            df.at[ind, 'RCx_both'] = row.RCx_both.values[0]
            df.at[ind, 'ncvd'] = row.ncvd.values[0]
            df.at[ind, 'pci_bool'] = row.pci_bool.values[0]
            df.at[ind, 'graft'] = row.graft.values[0]
            df.at[ind, 'sex'] = row.sex.values[0]
            df.at[ind, 'age'] = row.age.values[0]
            df.at[ind, 'DBC'] = row.DBC.values[0]
            df.at[ind, 'occlusion'] = row.occlusion_bool.values[0]
            df.at[ind, 'creatinine'] = row.creatinine.values[0]
            df.at[ind, 'CK_max'] = row.CK_max.values[0]
            df.at[ind, 'trop_max'] = row.trop_max.values[0]
        else:
            df.drop(ind, inplace=True)

    df.dropna(how='any', subset=['RCA'], inplace=True)
    return df

def complete_pred_lvf2(df_lvf, df):
    """Join ECG rows to LV function labels (LVF, dilation, hypertrophy...)."""


    for ind in df.index:
        pi = df.loc[ind, 'PatientID']
        date = df.loc[ind, 'Date']
        if date in df_lvf.query('PatientID == @pi').Date.values:
            row = df_lvf.query(
                '((PatientID == @pi) and (Date == @date))')
            df.at[ind, 'lv_function'] = row.lv_function.values[0]
            df.at[ind, 't'] = row.t.values[0]
            df.at[ind, 'time'] = row.time.values[0]
            df.at[ind, 'sex'] = row.sex.values[0]
            df.at[ind, 'age'] = row.age.values[0]
            df.at[ind, 'dead_int'] = row.dead_int.values[0]
            df.at[ind, 'lv_function'] = row.LVF.values[0]
            df.at[ind, 'rv_function'] = row.RVF.values[0]
            df.at[ind, 'lv_dilated'] = row.LV_dilated.values[0]
            df.at[ind, 'rv_dilated'] = row.RV_dilated.values[0]
            df.at[ind, 'lv_hypertrophy'] = row.LV_hypertrophy.values[0]
            df.at[ind, 'klepvitia'] = row.Klepvitia.values[0]
            
        else:
            df.drop(ind, inplace=True)

    df.dropna(how='any', subset=['lv_function'], inplace=True)
    return df


def complete_pred_MI(df, data_full):
    """Join MI-label dataset rows to the full ECG dataset by patient+date."""

    df.reset_index(drop=True, inplace=True)
    for m in ['qrs', 'mean_rri']:
        df[m] = None
        df[m] = df[m].astype(object)
    count, count2 = 0, 0 
    df['ECG_date'] = pd.to_datetime(df['ECG_date'])
    data_full['Date'] = pd.to_datetime(data_full['Date'])
    for ind in df.index:
        pi = df.loc[ind, 'PatientID']
        date = df.loc[ind, 'ECG_date']
        if data_full.query('PatientID == @pi').empty:
            count2 += 1
            continue
        if date in data_full.query('PatientID == @pi').Date.values:
            value = data_full.query(
                '((PatientID == @pi) and (Date == @date))')
            row = df.loc[ind]
            for v in value.iterrows():
                ind = v[0]
                df.at[ind, 'PatientID'] = pi
                df.at[ind, 'ECG_date'] = date
                df.at[ind, 'HK_date'] = row.HK_date
                df.at[ind, 'sex'] = row.sex
                df.at[ind, 'age'] = row.age
                df.at[ind, 'DBC'] = row.DBC
                df.at[ind, 'dead'] = row.dead
                df.at[ind, 'creatinine'] = row.creatinine
                df.at[ind, 'CK_max'] = row.CK_max
                df.at[ind, 'Death_cause'] = row.Death_cause
                df.at[ind, 'trop_max'] = row.trop_max
                df.at[ind, 'label_pci'] = row.label_pci
                df.at[ind, 'label_DOM'] = row.label_DOM
                df.at[ind, 'label_main'] = row.label_main
                df.at[ind, 'label_sub'] = row.label_sub
                df.at[ind, 'label_no_cvd'] = row.label_no_cvd
                df.at[ind, 'label_occlusion'] = row.label_occlusion
                df.at[ind, 'label_pci_bool'] = row.label_pci_bool
                df.at[ind, 'label_graft'] = row.label_graft

                for m in ['qrs','mean_rri']:
                    df.at[ind, m] = v[1][m]    
        else:
            count += 1
    print(count, count2)
    df = df[df['qrs'].apply(lambda x: x is not None)]
    return df


def complete_pred_mi(df_labels, df):    
    """Label each row with an MI type (LAD=1, LCx=2, RCA=3, other=0)."""

    for ind in df.index:
        pi = df.loc[ind, 'PatientID']
        if pi in df_labels.index:
            if df_labels.loc[pi, 'label'] == 'LAD':
                df.loc[ind, 'MI_type'] = 1
            elif df_labels.loc[pi, 'label'] == 'LCx':
                df.loc[ind, 'MI_type'] = 2
            elif df_labels.loc[pi, 'label'] == 'RCA':
                df.loc[ind, 'MI_type'] = 3
            else:
                df.loc[ind, 'MI_type'] = 0
        else:
            df.loc[ind, 'MI_type'] = np.nan
    return df


def match_withings(data_withings, data_full, d=3):
    """Match home ECG (Withings) recordings to hospital ECG rows.

    A Withings record is linked to a hospital row when it falls within
    `d` days and has the closest mean RRI.
    """
    pis = []
    rows = []
    for pi in data_withings.PatientID.unique():
        if pi in data_full.PatientID.unique():
            pis.append(pi)
    for pi in pis:
        temp_w = data_withings.query("PatientID == @pi").index
        temp_d = data_full.query("PatientID == @pi").index

        for ind_d in temp_d:
            date_d = data_full.loc[ind_d, 'Date']
            date_dp3 = date_d + pd.Timedelta(days=d)
            data_dm3 = date_d - pd.Timedelta(days=d)
            rri = data_full.loc[ind_d, 'mean_rri'].mean()

            ind_ws = []
            rri_ws = []
            for ind_w in temp_w:
                date_w = data_withings.loc[ind_w, 'Date']
                if data_dm3 < date_w < date_dp3:
                    ind_ws.append(ind_w)
                    rri_ws.append(
                        abs(data_withings.loc[ind_w, 'mean rri'] - rri))

            for i, rri_i in enumerate(rri_ws):
                if rri_i == np.min(rri_ws):
                    rec = {c: data_full.loc[ind_d, c] for c in data_full.columns}
                    rec['qrs_w'] = data_withings.loc[ind_ws[i], 'qrs']
                    rec['mean_rri_w'] = data_withings.loc[ind_ws[i], 'mean rri']
                    rows.append(rec)

    if rows:
        return pd.DataFrame(rows, columns=list(data_full.columns) + ['qrs_w', 'mean_rri_w'])
    # keep columns stable when there are no matches
    return pd.DataFrame({c: [] for c in list(data_full.columns) + ['qrs_w', 'mean_rri_w']})


def train_test_val_split(data_pred, data_full, nr_splits=3, complete=True, random_state=42, mi=False, a112=False):
    """Main split orchestrator: n rounds of (pred split, full split).

    Returns train_p (prediction), train_f (reconstruction) and the
    list of held-out test sets (predict + recon).
    """
    test_sets_f, test_sets_p = [], []
    if complete and not a112:
        data_in = copy.deepcopy(data_full[data_full['qrs'] != ''])
        if mi:
            data_pred = complete_pred_angio(data_pred, data_full)
        else:
            data_pred = complete_pred_lvf2(data_pred, data_full)

    else:
        data_in = data_full
        
    for _ in range(nr_splits):
        train_p, test_p = split_pred(data_pred, random_state=random_state, mi=mi, a112=a112)
        train_f, test_f = split_full_ex_prediction(data_in, test_p=test_p, random_state=random_state, a112=a112)
        data_pred = copy.deepcopy(train_p)
        data_in = copy.deepcopy(train_f)
        test_sets_f.append(test_f)
        test_sets_p.append(test_p)

    print('------------ Check PI Overlap ------------')
    for i in range(nr_splits):
        check_pi_overlap(train_f, test_sets_f[i], a112=a112)

    return train_p, train_f, test_sets_p, test_sets_f


def scale_data(data, column, scaler=None, a12=False):
    """Z-score scale one column of (array-valued) ECG data via StandardScaler."""

    if column == 'mean_rri_w':
        data_in = data[column].values
    elif a12:
        data_in = data  
    else:
        data_in = np.stack(data[column].values.squeeze())
    _shape = data_in.shape
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(data_in.reshape(-1, 1))

    data_out = scaler.transform(data_in.reshape(-1, 1)).reshape(
        _shape
    )
    return data_out, scaler

def scale_save(data, scaler=None, path='data/training_set', leads=1):
    """Scale the QRS column of a dataset in place and save it to pickle."""

    if leads == 1:
        qrs = 'qrsw'
    else:
        qrs = 'qrs'
    data_scaled, scaler = scale_data(data, qrs)
    for i, j in enumerate(data.index):
 
        data.at[j, qrs] = data_scaled[i]

    data.to_pickle(path)
    return scaler


def scale_all(train, tests, nr_splits, predict=False, withings=False, mi=False, a112=False):
    """Fit the scaler on train and apply it to all held-out splits."""

    if mi:
        columns = ['qrs', 'mean_rri']
    else:
        columns = ['qrs']
    if predict:
        columns.append('age')
    if withings:
        columns.append('qrs_w')
        columns.append('mean_rri_w')
    
    if a112:
        columns = ['qrsw']

    for m in columns:
        train_scaled, scaler = scale_data(train, m)
        train[m] = list(train_scaled)

        for i in range(nr_splits):
            test_scaled, _ = scale_data(tests[i], m, scaler=scaler)
            tests[i][m] = list(test_scaled)
    return train, tests, scaler


def load_full(filename):
    """Load a pickled full ECG dataset (parent of cwd) and drop empty rows."""

    path = os.path.join(Path(os.getcwd()).parent, filename)
    df = pd.read_pickle(path)
    df = df[df['qrs'] != '']
    return df.reset_index(drop=True)


def run_splits_12_1_CNN(label='MI', leads=1, rs=0):
    """Scripted pipeline: build train/val/test for the CNN 12-lead/1-lead task."""

    
    if label == 'MI':
        mi = True
    else:
        mi = False
    if leads == 12:
        double = True
    else:
        double = False
    end = f'{leads}_{label}_{rs}.p'
    
    if mi:
        # Filter out the data that has different MI label then RDA, RCx, RCA
        data_train_val = data_train_val.query('MI_label in ["RDA", "RCx", "RCA"]')
    else:
        data_train_val = pd.read_pickle(f'data/combined4_{label}_{leads}_lead_qrs.p')
                
    data_train_val.reset_index(drop=True, inplace=True)
    
    data_train, data_test = split_pred(data_train_val, random_state=rs, a112=True)
    data_train, data_val = split_pred(data_train, random_state=rs, a112=True)

    if double:
        data_test = pd.concat([data_test, data_val])
        data_test.reset_index(drop=True, inplace=True)
        data_train_val2 = pd.read_pickle(f'data/just4_{label}_{leads}_lead_qrs.p')
        if mi:
            data_train_val2 = data_train_val2.query('MI_label in ["RDA", "RCx", "RCA"]')    
        data_train_val2.reset_index(drop=True, inplace=True)
        
        data_train2, data_val1 = split_pred(data_train_val2, random_state=rs, a112=True)
        data_train2, data_val2 = split_pred(data_train2, random_state=rs, a112=True)
        
        data_train = pd.concat([data_train, data_train2])
        data_val = pd.concat([data_val1, data_val2])
        
        data_train.reset_index(drop=True, inplace=True)
        data_val.reset_index(drop=True, inplace=True)
        
        
    scaler = scale_save(data_train, path='data/train_set_' + end, leads=leads)
    dump(scaler, f'data/scaler_{label}_{leads}.pkl')
    
    _ = scale_save(data_val, scaler=scaler, path='data/val_set1_' + end, leads=leads)
    _ = scale_save(data_test, scaler=scaler, path='data/test_set' + end, leads=leads)
        
def scale_1_12(data, additional_data, cv=0, scaler=None, columns=['qrsw_p', 'qrs12_0', 'qrs12_1', 'qrs12_2', 'qrs12_3', 'qrs12_4']):
    """Scale the multi-column 1-to-12 lead feature set and save scalers."""

    
    val = additional_data[0]
    test = additional_data[1]
    train_data, val_data, test_data = [], [], []
    
    data12_0 = np.stack(data[['qrs12_0']].values.squeeze())
    data12_1 = np.stack(data[['qrs12_1']].values.squeeze())
    data12_2 = np.stack(data[['qrs12_2']].values.squeeze())
    data12_3 = np.stack(data[['qrs12_3']].values.squeeze())
    data12_4 = np.stack(data[['qrs12_4']].values.squeeze())
    data12 = np.concatenate([data12_0, data12_1, data12_2, data12_3, data12_4])
    
    train_data.append(data.PatientUniqueID.values)
    val_data.append(val.PatientUniqueID.values)
    test_data.append(test.PatientUniqueID.values)
    
    _, scaler12 = scale_data(data12, 'qrs12_0', scaler=None, a12=True)
    with open(f'data/scaler_qrs12_{cv}.pkl' , 'wb') as f:
        pickle.dump(scaler12, f)
    
    for i in range(len(columns)):
        column = columns[i]
        
        if i == 0:
            data_scaled, scaler = scale_data(data, column, scaler=None)
            train_data.append(data_scaled) 
            with open(f'data/scaler_{column}_{cv}.pkl', 'wb') as f:
                pickle.dump(scaler, f)
        else:
            data_scaled, _ = scale_data(data, column, scaler=scaler12)
            train_data.append(data_scaled)
            scaler = scaler12
        
        val_scaled, _ = scale_data(val, column, scaler=scaler)
        val_data.append(val_scaled)
                
        test_scaled, _ = scale_data(test, column, scaler=scaler)
        test_data.append(test_scaled)
    return train_data, val_data, test_data
        

def run_simple_split(path, leads=1, rs=0, qrs='qrsw_p'):
    """Simple 85/15 grouped split of a pickled dataset, with scaling."""

    df = pd.read_pickle(path)

    # only select row where df[qrs] is not a float and has the shape of 12, 400 or 400
    df = df[df[qrs].apply(lambda x: type(x) == np.ndarray and x.shape == (400,))]    
    df = df[[False if x[0] == 0.0 else True for x in df[qrs]]] 
        
    train, test = split_full(df, 0.85, random_state=rs, a112=leads==1)
    train, val = split_full(train, 0.85, random_state=rs, a112=leads==1)
    
    train, val, test = scale_1_12(train, [val, test], cv=rs)

    return train, val, test
    
    

def run_splits_12_1(nr_splits, save=False, test=False, label='MI', leads=1):
    """Scripted pipeline: predict/recon split for the 1-to-12 lead task."""

    
    end = f'{leads}_{label}_112.p'
    data_pred = pd.read_pickle(f'data/combined_1_12_lead_after_{label}_qrsw.p')
    data_full = pd.read_pickle(f'data/combined_{leads}_lead_after_{label}_nl_qrsw_all.p')

    if label == 'echo':
        data_pred.dropna(subset=['lvf'], inplace=True)
    train_pc, train_f, test_sets_pc, test_sets_f = train_test_val_split(
        data_pred, data_full, nr_splits=nr_splits, complete=False, a112=True)
    
    train_pc, test_sets_pc = scale_all(
        train_pc, test_sets_pc, nr_splits=nr_splits, predict=True, a112=True)
    train_f, test_sets_f = scale_all(
        train_f, test_sets_f, nr_splits=nr_splits, predict=False, a112=True)
    
    if save:
        train_pc.to_pickle('data/training_set_predict' + end)
        train_f.to_pickle('data/training_set_recon' + end)
        for i in range(nr_splits):
            if test and i == 0:
                test_sets_pc[i].to_pickle('data/test_set_predict' + end)
                test_sets_f[i].to_pickle('data/test_set_recon' + end)
            else:
                test_sets_pc[i].to_pickle(
                    'data/val_set' + str(i) + '_predict' + end)
                test_sets_f[i].to_pickle(
                    'data/val_set' + str(i) + '_recon' + end)

def run_splits(nr_splits, save=False, test=False):
    """Scripted pipeline: default train/val/test splits (ASSIST dataset)."""

    end = '_AS_AS.p'
    data_pred = pd.read_pickle("data/ASSIST_for_preds_labeled_OpenAI2.p")
    mi = True

    data_pred.reset_index(drop=True, inplace=True)
    data_full = load_full('ecg12_400_all_embed_AS.p')

    train_pc, train_f, test_sets_pc, test_sets_f = train_test_val_split(
        data_pred, data_full, nr_splits=nr_splits, mi=mi)

    train_pc, test_sets_pc, scaler_p = scale_all(
        train_pc, test_sets_pc, nr_splits=nr_splits, predict=True, mi=mi)
    train_f, test_sets_f, scaler_f = scale_all(
        train_f, test_sets_f, nr_splits=nr_splits, predict=False, mi=mi)

    if save:
        train_pc.to_pickle('data/training_set_predict' + end)
        train_f.to_pickle('data/training_set_recon' + end)
        for i in range(nr_splits):
            if test and i == 0:
                test_sets_pc[i].to_pickle('data/test_set_predict' + end)
                test_sets_f[i].to_pickle('data/test_set_recon' + end)
            else:
                test_sets_pc[i].to_pickle(
                    'data/val_set' + str(i) + '_predict' + end)
                test_sets_f[i].to_pickle(
                    'data/val_set' + str(i) + '_recon' + end)


def run_withings(nr_splits, save=True):
    """Scripted pipeline: match Withings recordings and build the matched sets."""

    df_full = load_full('ecg12_400_all_embed8_augmented4.p')
    df_withings = load_full('output/ecg_withings_embed_sep.p')
    df_matched = match_withings(df_withings, df_full, d=3)

    test_sets_m = []
    for _ in range(nr_splits):
        train_m, test_m = split_full(df_matched, 0.85)
        df_matched = copy.deepcopy(train_m)
        test_sets_m.append(test_m)

    train_m, test_sets_m = scale_all(
        train_m, test_sets_m, nr_splits=nr_splits, withings=True)

    if save:
        train_m.to_pickle('data/training_set_matched2.p')
        train_m.iloc[:1000].to_pickle('data/training_set_matched_s2.p')
        train_m.drop_duplicates(subset=['PatientID']).to_pickle(
            'data/training_set_matched_un2.p')

        for i in range(nr_splits):
            if i == 0:
                test_sets_m[i].to_pickle('data/test_set_matched2.p')
                test_sets_m[i].iloc[:1000].to_pickle(
                    'data/test_set_matched_s2.p')
                test_sets_m[i].drop_duplicates(subset=['PatientID']).to_pickle(
                    'data/test_set_matched_un2.p')

            else:
                test_sets_m[i].to_pickle(
                    'data/val_set' + str(i) + '_matched2.p')
                test_sets_m[i].iloc[:1000].to_pickle(
                    'data/val_set' + str(i) + '_matched_s2.p')
                test_sets_m[i].drop_duplicates(subset=['PatientID']).to_pickle(
                    'data/val_set' + str(i) + '_matched_un2.p')


def create_small_data_set(path):
    """Create a de-duplicated small version of a dataset (one row per patient+date)."""

    data = pd.read_pickle('data/' + path + '.p')
    data_s = data.drop_duplicates(subset=['PatientID', 'Date'])
    data_s.to_pickle('data/' + path + '_un.p')


def run_small_data_set2():
    """Create small datasets for the matched (home ECG) pipeline."""

    for path in ['training_set_matched', 'val_set1_matched',
                  'val_set2_matched', 'test_set_matched']:
        create_small_data_set(path)


def run_small_data_set():
    """Create small train/test sets from one val set (quick dev splits)."""

    nr_splits = 5
    data_p = pd.read_pickle('data/val_set2_predict.p')
    data_full = pd.read_pickle('data/val_set2_recon.p')
    train_pc, train_f, test_sets_pc, test_sets_f = train_test_val_split(
        data_p, data_full, nr_splits=nr_splits, complete=False)
    train_pc.to_pickle('data/training_set_predict_s.p')
    train_f.to_pickle('data/training_set_recon_s.p')
    for i in range(nr_splits):
        if i == 0:
            test_sets_pc[i].to_pickle('data/test_set_predict_s.p')
            test_sets_f[i].to_pickle('data/test_set_recon_s.p')
        else:
            test_sets_pc[i].to_pickle(
                'data/val_set' + str(i) + '_predict_s.p')
            test_sets_f[i].to_pickle('data/val_set' + str(i) + '_recon_s.p')


def run_just_scale():
    """Re-scale already-split datasets in place (no re-splitting)."""


    train_pc = pd.read_pickle('data/training_set_predict.p')
    train_f = pd.read_pickle('data/training_set_recon.p')
    train_f = train_f[train_f['qrs'] != ""]
    test_sets_pc0 = pd.read_pickle('data/test_set_predict.p')
    test_sets_f0 = pd.read_pickle('data/test_set_recon.p')
    test_sets_pc1 = pd.read_pickle('data/val_set1_predict.p')
    test_sets_f1 = pd.read_pickle('data/val_set1_recon.p')
    test_sets_pc2 = pd.read_pickle('data/val_set2_predict.p')
    test_sets_f2 = pd.read_pickle('data/val_set2_recon.p')

    test_sets_f, test_sets_pc = [], []
    test_sets_f.append(test_sets_f0[test_sets_f0['qrs'] != ''])
    test_sets_pc.append(test_sets_pc0)
    test_sets_f.append(test_sets_f1[test_sets_f1['qrs'] != ''])
    test_sets_pc.append(test_sets_pc1)
    test_sets_f.append(test_sets_f2[test_sets_f2['qrs'] != ''])
    test_sets_pc.append(test_sets_pc2)

    train_pc, test_sets_pc = scale_all(
        train_pc, test_sets_pc, nr_splits=1)
    train_f, test_sets_f = scale_all(train_f, test_sets_f, nr_splits=1)
    train_pc.to_pickle('data/training_set_predict.p')
    train_f.to_pickle('data/training_set_recon.p')
    test_sets_pc[0].to_pickle('data/test_set_predict.p')
    test_sets_f[0].to_pickle('data/test_set_recon.p')
    test_sets_pc[1].to_pickle('data/val_set1_predict.p')
    test_sets_f[1].to_pickle('data/val_set1_recon.p')
    test_sets_pc[2].to_pickle('data/val_set2_predict.p')
    test_sets_f[2].to_pickle('data/val_set2_recon.p')
