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
from torch.utils.data import Dataset
from dgllife.utils import ScaffoldSplitter

class MoleculeDataset(Dataset):
    def __init__(self, smiles, labels):
        self.smiles = smiles
        self.labels = labels

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx]

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default='smiles.csv', help='Path to the input file')
    parser.add_argument('--model_name', type=str, default=None, help='User-defined model name')
    parser.add_argument('--max_eval', type=int, default=200, help='Maximum number of model evaluation')
    parser.add_argument('--time_out', type=int, default=120, help='Maximum trial time out')
    parser.add_argument('--cross_validation', action='store_true', default=False, help='Use this option in case of cross validation')
    parser.add_argument('--training', action='store_true', default=False, help='Use this option in case of training model')
    parser.add_argument('--split', choices=['scaffold', 'stratify'], default='stratify', help='Dataset split method in cross validation')
    parser.add_argument('--k_fold', type=int, default=5, help='k-fold cross validtion')

    args  =parser.parse_args()
    print(args)
    features = eval(open(f'./features.txt', 'r').read())
    if args.model_name is None:
        args.model_name = args.file_name[:-4]
    print(f'User-defined model name: {args.model_name}')

    if args.training:
        train_data = pd.read_csv(f'train_{args.file_name}')
        test_data = pd.read_csv(f'test_{args.file_name}')
        print(f'Loading train/test set: {len(train_data)}, {len(test_data)}')
        print(f'Hyperparameters for hpsklearn \n max_eval : {args.max_eval} \n time_out : {args.time_out}')
    
        clf_name = f'{args.max_eval}_{args.time_out}'
    
        print(f'Start training model \n')
    
        tt = time.time()
        estimator = HyperoptEstimator( classifier=any_classifier('cla'),
                                       max_evals = args.max_eval,
                                       trial_timeout = args.time_out,
                                       preprocessing=any_preprocessing('pre'),
                                       algo=tpe.suggest)
        
        estimator.fit(train_data[features].values, np.array(train_data['bioclass']))
        et = time.time()
        print(f'Time for model.fit: {et-tt}')
    
        confidence_score = estimator.score(test_data[features].values, np.array(test_data['bioclass']))
        print('Confidence score of trained model', confidence_score)
        
        best_model = estimator.best_model()['learner']
        print('Best performance model: ', best_model)
        
        pickle.dump(best_model, open(f'./{args.model_name}.pkl', 'wb'))
        
        predicted = best_model.predict(test_data[features].values)
        try:
            predicted_proba = best_model.predict_proba(test_data[features].values)[:,1]
        except:
            predicted_proba = best_model.decision_function(test_data[features].values)

        #confusion matrix print
        tn, fp, fn, tp = confusion_matrix(test_data['bioclass'], predicted).ravel()
        print('Confusion matrix (Se)', tp/(tp+fn))
        print('Confusion matrix (Sp)', tn/(tn+fp))
        print('Confusion matrix (acc)', (tp+tn)/(tp+fp+tn+fn))
        print('Confusion matrix (mcc)', ((tp*tn)-(fp*fn))/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5 )
        print('AUC', roc_auc_score(test_data['bioclass'], predicted_proba) )
        print(f'========== End of traininig model for {args.model_name} using {args.file_name}')

    if args.cross_validation:
        print(f'Start {args.k_fold}-fold cross validation using {args.file_name}')
        train_data = pd.read_csv(f'train_{args.file_name}')
        X = train_data[features]
        y = train_data['bioclass']
        n_samples, n_features = X.shape
    
        tprs, aucs = [], []
        if args.split == 'stratify':
            print(f'train/test data divided by stratify-based method \n')
            cv = StratifiedKFold(n_splits=args.k_fold) 
            folds = cv.split(X, y)
        elif args.split =='scaffold':
            print(f'train/test data divided by scaffold-based method \n')
            list_smiles = train_data['SMILES'].tolist()
            list_labels = train_data['bioclass'].tolist()
            dataset = MoleculeDataset(list_smiles, list_labels)
            splitter = ScaffoldSplitter()
            folds = splitter.k_fold_split(dataset=dataset, k=args.k_fold)

        for fold_idx, (train_subset, test_subset) in enumerate(folds):

            clf_name = f'{fold_idx}_{args.model_name}'
            try:
                train = train_subset.indices
                test  = test_subset.indices
            except:
                train = train_subset
                test  = test_subset

            print(f'[fold idx {fold_idx+1}] train: {len(train)}, test: {len(test)}') 
            
            classifier = HyperoptEstimator( classifier = any_classifier('cla'),
                                            max_evals = args.max_eval,
                                            trial_timeout = args.time_out,
                                            preprocessing= any_preprocessing('pre'),
                                            algo=tpe.suggest)
    
            classifier.fit(X.iloc[train].values, y.iloc[train].values)

            classifier = classifier.best_model()['learner']
            print(classifier, type(classifier))

            predicted = classifier.predict(X.iloc[test].values)
            predicted_proba = classifier.predict_proba(X.iloc[test].values)[:,1]
            
            #confusion matrix print
            tn, fp, fn, tp = confusion_matrix(y.iloc[test].values, predicted).ravel()
            print('Confusion matrix (Se)', tp/(tp+fn))
            print('Confusion matrix (Sp)', tn/(tn+fp))
            print('Confusion matrix (acc)', (tp+tn)/(tp+fp+tn+fn))
            print('Confusion matrix (mcc)', ((tp*tn)-(fp*fn))/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5 )
            print('AUC', roc_auc_score(y.iloc[test].values, predicted_proba) )
            print(f'========== End of {fold_idx+1}-fold cross validation model for {args.model_name} using {args.file_name}')
