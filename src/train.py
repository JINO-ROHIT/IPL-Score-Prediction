from ast import parse
import joblib
import os
import argparse
import config
import model_dispatcher
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import ensemble
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

df = pd.read_csv(config.TRAINING_FILE)
#print(df.columns)


def data_encoding( encoding_strategy , encoding_data , encoding_columns ):
    
    if encoding_strategy == "LabelEncoding":
        print("LabelEncoding chosen")
        Encoder = LabelEncoder()
        for column in encoding_columns :
            print("column",column )
            encoding_data[ column ] = Encoder.fit_transform(tuple(encoding_data[ column ]))
        
    elif encoding_strategy == "OneHotEncoding":
        print("OneHotEncoding chosen")
        encoding_data = pd.get_dummies(encoding_data)
        
    dtypes_list =['float64','float32','int64','int32']
    encoding_data.astype( dtypes_list[0] ).dtypes
    
    return encoding_data

cat_cols = ['venue','bat_team','bowl_team']
encoding_strategy = ['LabelEncoding','OneHotEncoding']

encoded_df = data_encoding(encoding_strategy[1], df, cat_cols) #ohe for forest based algorithm
encoded_df = encoded_df.drop(['venue_Barabati Stadium','bat_team_Chennai Super Kings','bowl_team_Chennai Super Kings'],axis =1)


def run(fold,model):

    df_train = encoded_df[encoded_df.kfold != fold].reset_index(drop=True)
    df_valid = encoded_df[encoded_df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop("total", axis=1).values
    y_train = df_train.total.values
 
    x_valid = df_valid.drop("total", axis=1).values
    y_valid = df_valid.total.values
 
    clf = model_dispatcher.models[model]
    clf.fit(x_train, y_train)
    preds = clf.predict(x_valid)
 
    error = np.sqrt(mean_squared_error(y_valid,preds))
    print(f"Fold = {fold}, error = {error}")

    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT,f"dt_{fold}.bin"))
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fold',
        type = int
    )
    parser.add_argument(
        '--model',
        type = str
    )
    args = parser.parse_args()
    run(
        fold= args.fold,
        model = args.model
    )