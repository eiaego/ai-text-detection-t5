import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# prepare datasets
# 1 for binary classification, 1 for multiclass

def prepare_datasets():
    df_binary_big = pd.read_csv('dataset_multi.csv')
    for i in range(len(df_binary_big)):
        if df_binary_big.iloc[i]['text'] == None:
            df_binary_big.iloc[i].drop()
        elif df_binary_big.iloc[i]['label'] == 'human':
            df_binary_big.iloc[i]['label'] = 0
        else:
            df_binary_big.iloc[i]['label'] = 1

    df_multi = pd.read_csv('dataset_multi.csv')
    df_multi['label'], mappings_multi = pd.factorize(df_multi['label'], sort=True)

    df_train_binary_big, df_test_binary_big = train_test_split(df_binary_big, test_size=0.2, stratify=df_binary_big['label']) 
    df_train_multi, df_test_multi = train_test_split(df_multi, test_size=0.2, stratify=df_multi['label']) 

    # Filter null entries

    df_train_binary_big = df_train_binary_big[df_train_binary_big['text'] == df_train_binary_big['text']]
    df_test_binary_big = df_test_binary_big[df_test_binary_big['text'] == df_test_binary_big['text']]
    df_train_multi = df_train_multi[df_train_multi['text'] == df_train_multi['text']]
    df_test_multi = df_test_multi[df_test_multi['text'] == df_test_multi['text']]

    return df_train_binary_big, df_test_binary_big, df_train_multi, df_test_multi