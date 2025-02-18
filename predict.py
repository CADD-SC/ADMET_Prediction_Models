from sklearn.ensemble import RandomForestClassifier ###1. RandomForest
from sklearn import svm ###2. SVM
import xgboost as xgb ###3. XGB
from sklearn.ensemble import GradientBoostingClassifier ###4.GB
import time
import pandas as pd
import numpy as np
from numpy import array
import pickle
import os
from os import path
import sklearn
from sklearn.metrics import roc_auc_score, make_scorer, recall_score, accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from hpsklearn import HyperoptEstimator, any_preprocessing, any_classifier
from hpsklearn import random_forest_classifier, gradient_boosting_classifier, svc, xgboost_classification, k_neighbors_classifier
from sklearn.svm import SVC
from hyperopt import tpe, hp
import argparse
from utils import data_preprocessing

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default='example_train.csv', help='Path to the input file')
    parser.add_argument('--model_name', type=str, default=None, help='User-defined model name')
    args  =parser.parse_args()
    print(args)
    features = eval(open(f'./features.txt', 'r').read())

    print(f'Prediction model for {args.model_name}')
    print(f'Loading the data set : {args.file_name}')

    # prprocessing data feature
    df = pd.read_csv(args.file_name)
    df['bioclass']=1
    scaled_data = data_preprocessing(df)

    result_df = pd.DataFrame({'SMILES':scaled_data['SMILES']})
    if path.exists(f'models/{args.model_name}.pkl'):
        learner_model = pickle.load(open(f'models/{args.model_name}.pkl', 'rb'))
    elif path.exists(f'{args.model_name}.pkl'):
        learner_model = pickle.load(open(f'{args.model_name}.pkl','rb'))
    else:
        print('Check the model path (.pkl) !!')
        exit(-1)

    # predict label (int)
    predicted = learner_model.predict(scaled_data[features].values)
    result_df[f'{args.model_name}']=learner_model.predict(scaled_data[features].values)

    # predict prediction prob (float)
    predict_proba = learner_model.predict_proba(scaled_data[features].values)
    max_proba = np.max(predict_proba, axis=1)
    result_df['pred_prob']=max_proba
    print(f'Output for prediction of {args.model_name} \n {result_df}')
    result_df.to_csv(f'{args.model_name}_predict_results.csv', index=False)
