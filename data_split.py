import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
import pickle
import joblib
import os
from utils import data_preprocessing

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default='smiles.csv', help='Path to the input file')
    parser.add_argument('--split_mode', choices=['scaffold', 'stratify'], default='stratify', help='Seperate method of dataset for train/test set')
    parser.add_argument('--test_frac', type=float, default=0.2, help='Test set fraction (default: 0.2)')
    args  =parser.parse_args()
    
    df = pd.read_csv(args.file_name)
    # preprocess from smiles to features
    print(f'Start data pre-processing for {args.file_name}')
    df = data_preprocessing(df)

    if args.split_mode == 'stratify' :
        print('stratified split')
        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(
                                                  df, 
                                                  random_state=42, 
                                                  test_size = args.test_frac, 
                                                  stratify=df['bioclass'], 
                                                  shuffle = True
                                                )
        print(f'Size of train dataset: {len(train_data)} ')
        print(f'Size of test  dataset: {len(test_data)} ')
        train_data.to_csv(f'train_{args.file_name}',index=False)
        test_data.to_csv(f'test_{args.file_name}',index=False)
    elif args.split_mode =='scaffold' :
        print('scaffold-based split')
        from dgllife.utils import ScaffoldSplitter
        df = df.rename(columns={'SMILES': 'smiles'})
        train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
                              df,
                              frac_train=1-args.test_frac,
                              frac_val=0,
                              frac_test=args.test_frac,
                              )
        train_data = df.iloc[train_set.indices]
        test_data  = df.iloc[test_set.indices]
        print(f'Size of train dataset: {len(train_data)} ')
        print(f'Size of test  dataset: {len(test_data)} ')
        train_data.to_csv(f'train_{args.file_name}',index=False)
        test_data.to_csv(f'test_{args.file_name}',index=False)
