from ast import parse
import joblib
import os
import argparse
import config
import pickle
import model_dispatcher
import pandas as pd
import json
import numpy as np
from sklearn import metrics
from sklearn import ensemble
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

df = pd.read_csv(config.TRAINING_FILE)
df.venue = df.venue.str.lower()
df.bat_team = df.bat_team.str.lower()
df.bowl_team = df.bowl_team.str.lower()
venue = df.venue.unique()
team = df.bat_team.unique()


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
encoded_df = encoded_df.drop(['venue_barabati stadium','bat_team_chennai super kings','bowl_team_chennai super kings'],axis =1)
encoded_df = encoded_df.drop(['kfold'], axis= 1)
#print(len(encoded_df.columns))
#exit()

x_train = encoded_df.drop('total',axis = 1)
y_train = encoded_df.total
 
clf = ensemble.RandomForestRegressor()
clf.fit(x_train, y_train)

def predict(runs, wickets, overs, bat_team, bowl_team , venue):
    X_pred = np.zeros(len(x_train.columns))
    #print(X_pred.shape)
    
    if bat_team != "chennai super kings":
        
        bat_team_index = np.where(x_train.columns == "bat_team_" + bat_team)[0][0]
        #print(bat_team_index)
    
    if bowl_team != "chennai super kings":
        
        bowl_team_index = np.where(x_train.columns == "bowl_team_" + bowl_team)[0][0]
        
    if venue != "barabati stadium":
        venue_index = np.where(x_train.columns == "venue_" +venue)[0][0]
    
    numeric_columns = [runs, wickets, overs]
    for i in range(3):
        X_pred[i] = numeric_columns[i]
        
    
    result = clf.predict([X_pred])
    return result

score = predict(36, 2, 7.3, "delhi capitals", "mumbai indians", "m chinnaswamy stadium")
print(score)
 

with open("ipl.pickle", "wb")as f:
    pickle.dump(clf, f)


features = {"columns": x_train.columns.to_list(), "team": list(team), "venue": list(venue)}

with open("features.json", "w") as f:
    json.dump(features, f)