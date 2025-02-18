import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
import argparse
import pickle
import joblib
import os
#from hpsklearn.components import classifier

pd.set_option('display.max_columns', None)

def data_preprocessing(tmp_df):
    org_df = len(tmp_df)
    tmp_df['mol'] = [Chem.MolFromSmiles(s) for s in tmp_df['SMILES']] #filtering invalid mol
    data = tmp_df.dropna(subset=['mol']) 
    print('Remove invalid mol object: ', org_df - len(data))
    data.loc[:, 'SMILES'] = canonical_smiles(data['SMILES']) #canonicalize smi
    data = data.drop_duplicates(subset=['SMILES'], keep='first') #remove duplicate
    data = data[['SMILES','bioclass']]
    all_data = calculate_allfeatures(data) #1234 featurize , smi, bioclass
    all_data = standard_scaled(all_data)
    return all_data

def canonical_smiles(smiles_list):
    clean_smiles=[]
    for smiles in smiles_list:
        try:
            cpd=str(smiles).split('.')
            cpd_longest=max(cpd,key=len)
            can_smi = Chem.CanonSmiles(cpd_longest)
            clean_smiles.append(can_smi)
        except:
            print('!!!!!!!!except smi:', smiles)
            continue
    return clean_smiles

def calculate_allfeatures(df):
    dump_list = [(smi, bioclass) for smi, bioclass in zip (df['SMILES'], df['bioclass'])]
    descriptors_list = []
    ecfp6_list = []
    from multiprocessing import Pool
    with Pool(processes=24) as p:
        store_results = p.map(calculate_ecfp, dump_list)
    # Create a pandas DataFrame of the calculated features
    ecfp6_list =       [store_results[idx][0] for idx in range(len(dump_list))]
    descriptors_list = [store_results[idx][1] for idx in range(len(dump_list))]
    bioclass_list =    [store_results[idx][2] for idx in range(len(dump_list))]
    smiles_list =      [store_results[idx][3] for idx in range(len(dump_list))]
    descriptors_df = pd.DataFrame(descriptors_list)
    all_df = pd.DataFrame({'SMILES': smiles_list, 'bioclass': bioclass_list})
    ecfp6_df = pd.DataFrame(ecfp6_list, columns=['ECFP6_{}'.format(i+1) for i in range(ecfp6_list[0].size)])
    features_df = pd.concat([descriptors_df, ecfp6_df, all_df], axis=1)
    return features_df

def calculate_ecfp(tup_input):
    smiles, bioclass= tup_input[0], tup_input[1]
    mol = Chem.MolFromSmiles(smiles)
    desc = {}
    for desc_name, desc_func in Descriptors.descList:
        desc[desc_name] = desc_func(mol)
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
    return np.array(ecfp), desc, bioclass, smiles

def standard_scaled(all_data):
    categorical_columns = all_data.select_dtypes(exclude=['int64', 'float64']).columns
    all_data[categorical_columns] = all_data[categorical_columns].fillna('empty')
    all_data.fillna(0, inplace=True)
    features_df = all_data.drop(columns=['SMILES', 'bioclass', 'SPS', 'AvgIpc', 'NumAmideBonds', 'NumAtomStereoCenters', 'NumBridgeheadAtoms', 'NumHeterocycles', 'NumSpiroAtoms', 'NumUnspecifiedAtomStereoCenters', 'Phi'])

    ### load_scaler for all compounds
    scaler_load = joblib.load('scaler/new_scaler.pkl')
    results_df = scaler_load.transform(features_df)
    results_df = pd.DataFrame(results_df, columns=features_df.columns)
    results_df = results_df.map(lambda x: round(x, 10))

    return pd.concat([results_df, all_data[['SMILES','bioclass']]], axis=1)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default='Papp_final_data')
    parser.add_argument('--feature_name', type=str, default='Caco2_permeability')
    args  =parser.parse_args()

    features = eval(open(f'./features.txt', 'r').read())#CYP1A2_inhibition/
    print(len(features))
    tmp_df = pd.read_csv('test_example.csv')

    # preprocess the smiles
    scaled_all_data = data_preprocessing(tmp_df)
    print(scaled_all_data)
    exit(-1)
