from sklearn.metrics import average_precision_score, precision_recall_curve, auc, roc_curve, confusion_matrix, precision_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def get_auc_data(labels, pred):
    '''
    Input:
    labels: list of binary outputs
    pred: list of prediction value
    Output:
    fpr, tpr, roc_auc, 
    thresholds: list of thresholds
    '''
    ap = average_precision_score(labels, pred)
    lr_precision, lr_recall, _ = precision_recall_curve(labels, pred)
    auprc = auc(lr_recall, lr_precision)
    fpr, tpr, thresholds = roc_curve(labels, pred)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc, thresholds

def draw_fairness_ratio(group_1, group_2, 
                        fig_name='draw_fairness_ratio.png', 
                        title='Eval metrics for Young/Senior in COVID-19 Diagnosis model'):
    '''
    Draw Fairness Ratio plot
    
    Input: 
    group_1/group_2: dict
    fig_name: string
    Output:
    save figure
    '''
    colors = ['#e41a1c','#377eb8','#4daf4a','#b3e2cd','#984ea3','#ff7f00','#ffff33','#a65628']
    plt.figure(figsize=(12,7))

    plt.plot(group_1['threshold'], 
             list(np.array(group_1['demographic']) / np.array(group_2['demographic'])),
             color=colors[0],
            label='Demographic') #, marker='o'

    plt.plot(group_1['threshold'], 
             list(np.array(group_1['accuracy']) / np.array(group_2['accuracy'])),
             color=colors[1],
             label='Accuracy')

    plt.plot(group_1['threshold'], 
             list(np.array(group_1['tpr']) / np.array(group_2['tpr'])),
             color=colors[2],
            label='True Positive Rate')

    plt.plot(group_1['threshold'], 
             list(np.array(group_1['fpr']) / np.array(group_2['fpr'])),
             color=colors[3],
            label='False Positive Rate')

    plt.plot(group_1['threshold'], 
             list(np.array(group_1['prec']) / np.array(group_2['prec'])),
             color=colors[4],
            label='Precision')

    plt.plot(group_1['threshold'], 
             list(np.array(group_1['rec']) / np.array(group_2['rec'])),
             color=colors[5],
            label='Recall')

    plt.plot(group_1['threshold'], 
             list(np.array(group_1['error_rate']) / np.array(group_2['error_rate'])),
             color=colors[6],
            label='Error Rate')

    plt.plot(group_1['threshold'], 
             list(np.array(group_1['NEC']) / np.array(group_2['NEC'])),
             color=colors[7],
            label='Normalized Expected Cost')

    plt.title('Eval metrics for Young/Senior in COVID-19 Diagnosis model', fontsize=14)
    plt.hlines(y=1, xmin=0.0, xmax=1.0, color='green', alpha=0.5)

    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Ratio', fontsize=14)
    plt.yscale('log')
    plt.legend()
    plt.savefig(fig_name)
    
    

from sklearn.metrics import average_precision_score, precision_recall_curve, auc, roc_curve, confusion_matrix, precision_score, recall_score

def generate_evals(y_true, y_pred, C_fn, C_fp):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    demographic = (tp+fp)/(tn+fp+fn+tp)
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    fnr = fn/(fn+tp)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred) 
    error_rate = 1 - accuracy

    ######
    p_pos = (tp + fn)/(tn+fp+fn+tp)
    p_neg = (tn + fp)/(tn+fp+fn+tp)
    total_cost = p_pos*C_fn + p_neg*C_fp
    PC_pos = p_pos*C_fn/total_cost
    NEC = fnr*PC_pos + fpr*(1-PC_pos)
    return demographic, accuracy, tpr, fpr, prec, rec, error_rate, NEC

def generate_groups_result(group_1_df, group_2_df, thresholds,
                          group_1_c_fn, group_1_c_fp,
                          group_2_c_fn, group_2_c_fp,
                          ):
    group_1_result = {
        'threshold': [],
        'demographic': [],
        'accuracy': [],
        'tpr': [],
        'fpr': [],
        'prec': [],
        'rec': [],
        'error_rate': [],
        'NEC': []
    }

    group_2_result = {
        'threshold': [],
        'demographic': [],
        'accuracy': [],
        'tpr': [],
        'fpr': [],
        'prec': [],
        'rec': [],
        'error_rate': [],
        'NEC': []
    }

    with tqdm(total=len(thresholds)) as pbar:
        for threshold in thresholds[3:]:
        # for threshold in [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]:    
            group_1_result['threshold'].append(threshold)
            group_2_result['threshold'].append(threshold)

            group_1_df['pred_label'] = 1
            group_1_df.loc[group_1_df['pred'] <= threshold, 'pred_label'] = 0
            y_true = group_1_df['COVID_label']
            y_pred = group_1_df['pred_label']        
            demographic, accuracy, tpr, fpr, prec, rec, error_rate, NEC = generate_evals(y_true, y_pred, group_1_c_fn, group_1_c_fp)
            group_1_result['demographic'].append(demographic)
            group_1_result['accuracy'].append(accuracy)
            group_1_result['tpr'].append(tpr)
            group_1_result['fpr'].append(fpr)
            group_1_result['prec'].append(prec)
            group_1_result['rec'].append(rec)
            group_1_result['error_rate'].append(error_rate)
            group_1_result['NEC'].append(NEC)

            group_2_df['pred_label'] = 1
            group_2_df.loc[group_2_df['pred'] <= threshold, 'pred_label'] = 0
            y_true = group_2_df['COVID_label']
            y_pred = group_2_df['pred_label']
            demographic, accuracy, tpr, fpr, prec, rec, error_rate, NEC = generate_evals(y_true, y_pred, group_2_c_fn, group_2_c_fp)
            group_2_result['demographic'].append(demographic)
            group_2_result['accuracy'].append(accuracy)
            group_2_result['tpr'].append(tpr)
            group_2_result['fpr'].append(fpr)
            group_2_result['prec'].append(prec)
            group_2_result['rec'].append(rec)
            group_2_result['error_rate'].append(error_rate)
            group_2_result['NEC'].append(NEC)
            pbar.update(1)
            
    return group_1_result, group_2_result