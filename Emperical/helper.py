import sys, os, pickle
import pandas as pd

def pytorch_data_loader(label_type='17_labels', end_date = '2022-06-01'):
    '''
    
    '''
    ecg_paths = "/data/padmalab/ecg/data/processed/ECG_days_till_dth_df_070321.pickle"
    age_sex_path = "/data/padmalab/ecg/data/processed/xml_df/ecg_age_sex.pickle"
    ecg_split_path = "/data/padmalab/ecg/data/processed/ECG_all_dx_study_data_splits_with_pacemaker_051021.pickle"
    CV_number = 1
    
    #################### ecg label ##############################
    with open(ecg_paths, 'rb') as f:
        ecg_df = pickle.load(f)
        ecg_df = ecg_df[ecg_df['dateAcquired'] <= end_date]
        ecg_df['days_till_death'] = ecg_df['days_till_death'].dt.days
        ecg_df['time'] = ecg_df['days_till_death']
        ecg_df['event'] = ecg_df['death_occ']
        ecg_df['path'] = ecg_df.index.str[2:-1]
        ecg_df = ecg_df[['time', 'event', 'path']]
    #################### ecg splits ##############################

    with open(ecg_split_path,"rb") as f:
        ecg_splits = pickle.load(f)
        
    val = ecg_df[ecg_df.index.isin(ecg_splits['CV Test Sets - 5 folds']['Fold-'+str(CV_number)])]
    val.index = val.index.str[2:-1]

    train = ecg_df[ecg_df.index.isin(ecg_splits['CV Train Sets - 5 folds']['Fold-'+str(CV_number)])]
    train.index = train.index.str[2:-1]

    test = ecg_df[ecg_df.index.isin(ecg_splits['Holdout Set - all ECGs'])]
    test.index = test.index.str[2:-1]

    #################### age_sex_path ##############################

    age_sex_path_df = pd.read_pickle(age_sex_path)
    age_sex_path_df['ecgId'] = age_sex_path_df['ecgId'].str[2:-1]
    age_sex_path_df = age_sex_path_df.set_index('ecgId')
    age_sex_path_df['SEX'] = age_sex_path_df['SEX'].astype(float)

    #################### Diagnosis Label path ##############################

    if label_type == '17_labels':
        label_df = pd.read_pickle("/data/padmalab/ecg/data/processed/label_df/ECG_17_labels_25posDAD_10posED_ignore_ed_ver5.pickle")
        label_df = label_df.set_index('ecgid')

    elif label_type == '1414_labels':
        label_df = pd.read_pickle("/data/padmalab/ecg/data/processed/label_df/pretrain_labels/A1_v5_1000_1414_label_df.pickle")        

    label_df = label_df.astype('float32')
    label_df.index = label_df.index.str[2:-1]
    return age_sex_path_df, label_df, train, val, test