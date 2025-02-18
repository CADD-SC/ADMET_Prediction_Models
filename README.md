## Requirements
 * Python >= 3.9
 * scikit-learn >= 1.26.4
 * numpy == 11.5.0
 * hpsklearn == 1.0.3
 * hyperopt == 0.2.7
 * xgboost == 2.0.3
 * rdkit
 * pandas
 * dgllife

## Clone the repository
#### For Windows
```
# Install Git LFS (if you haven't already)

git lfs install

#Clone the repository:

git clone https://github.com/CADD-SC/ADMET_Prediction_Models.git

#Navigate to the cloned repository:

cd ADMET_Prediction_Models

#Fetch the LFS files:

git lfs pull
```
#### For Linux
```
# Install Git LFS (if not already installed)
sudo apt-get install git-lfs

# Initialize Git LFS
git lfs install

# Clone the repository
git clone https://github.com/CADD-SC/ADMET_Prediction_Models.git

# Navigate to the cloned repository
cd ADMET_Prediction_Models

# Fetch the LFS files
git lfs pull
```
#### Note: if you still face any problem, downlaod the models files (.pkl) individually and place in the working directory 
## Dataset
 All dataset used in manuscript are contained in data/ folder
 At each folder, train, test, and valid (external set) datasets are saved as separated files.
#### Classification criteria
 The models classify a compound based on its predicted values. For example, in the BBB permeability, if the logBB is greater than or equal to -1, the compound is classified as class 1 (permeable); otherwise, it is classified as class 0 (non-permeable). Please refer to the below table.
| Property | Criteria (labeled as 1) |
| ------ | ------ |
| Caco-2 permeability | Papp >= 8× 10-6 cm/s |
| P-gp substrate | ER >= 2 |
| BBB permeability | logBB >= -1 |
| CYP1A2 inhibition | IC50 or AC50 < 10µM |
| CYP2C9 inhibition | IC50 or AC50 < 10µM |
| CYP2C19 inhibition | IC50 or AC50 < 10µM |
| CYP2D6 inhibition | IC50 or AC50 < 10µM |
| CYP3A4 inhibition | IC50 or AC50 < 10µM |
| HLM stability | t1/2 > 30 min |
| RLM stability | t1/2 > 30 min |
| hERG inhibition | IC50 < 10µM  |

## Model training
#### Data preparation (pre-processing & splitting)
 You can use your SMILES data to train new model.
 or We provide an input example for training as `“smiles.csv”`.
 The dataset should contain the ‘SMILES’, 'bioclass' columns. (see smiles.csv)
```
 python data_split.py --file_name smiles.csv \
                    --split_mode stratify \
                    --test_frac 0.2
```
 `--file_name` : Path to the input file
 `--split_mode` : Split method of dataset for train/test set [choose mode between stratify and scaffold]
 `--test_frac` : Test set fraction (default: 0.2)

 The output data will be located in the current folder as follows :
```
 ./train_smiles.csv
 ./test_smiles.csv
```

#### Train the model
```
 python model.py --file_name smiles.csv \
                 --model_name test \
                 --max_eval 200 \
                 --time_out 120 \
                 --training
```
`--file_name` : Path to the input file
`--model_name` : User-defined model name (if not defined, the model save as input file name)
`--max_eval`: Maximum number of model evaluation for hyperopt-sklearn
`--time_out`: Maximum trial time out for hyperopt-sklearn
`--training`: turn on the training model

 The trained model file (.pkl) will be located in the current folder as follows :
```
./test.pkl
```

## Prediction
 You can predict your SMILES data using trained model.
 or just test an input example for prediction as `“tmp.csv”`.
 The dataset should contain the ‘SMILES’ columns. (see tmp.csv)
 
 The best performance models in the manuscript are available to predict.
 (defined model_name as `BBB`, `CYP1A2`, `CYP2C19`, `CYP2C9`, `CYP2D6`, `CYP3A4`, `HCLint`, `P_gp_subs`, `Papp`, `RCLint`, `hERG_inh`)
 When the prediction for BBB:
```
 python predict.py --file_name tmp.csv \
                   --model_name BBB
```

 The result file will be located in the current folder as follows :
```
 ./BBB_predict_results.csv
```
