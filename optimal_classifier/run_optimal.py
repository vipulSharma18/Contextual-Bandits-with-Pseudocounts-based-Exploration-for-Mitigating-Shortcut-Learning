'''
Optimal classifier. 
To obtain optimal classifications, we use Linear Discriminant Analysis 
(LDA, as implemented in Sklearn with the least-squares solver). 
the optimal classifier was fit to and probed with the same embedded base inputs used to train the corresponding model in a given experiment. 
'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from joblib import dump, load
import wandb

def load_lda(path): 
    loaded_lda = load(path)

def train_optimal(setting, seed, df): 
    s = setting.split('_')
    p_s, a_s = float(s[0]), int(s[1])
    df = df[(df['p_s']==p_s) & (df['a_s']==a_s)]
    X, y = np.stack(df['x']), df['class_label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1000, stratify=y, random_state=seed)
    print(f"Size of train {X_train.shape}, {y_train.shape}, size of test {X_val.shape}, {y_val.shape}")
    lda = LinearDiscriminantAnalysis(solver='lsqr')
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred)
    y_pred = lda.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred)
    print("Train Accuracy:", train_accuracy, "Val Accuracy:", val_accuracy)
    dump(lda, 'model_weights/lda_model_'+setting+'_'+str(seed)+'.joblib')
    return train_accuracy, val_accuracy
    
    
if __name__=='__main__': 
    data_path = '../data/'
    #test = 'synthetic_test_data.csv'
    train = 'synthetic_train_data.csv'
    df = pd.read_csv(data_path+train)
    df['x'] = df['x'].apply(lambda x: [float(i) for i in x[1:-1].split(',')])
    wandb.init(
        project="RL_Project_CSCI2951F", 
        config={
            'architecture': 'LDA',
            'task': 'red vs green'
        })
    settings = ['0.6_1', '0.6_2', '0.6_3', '0.6_4', '0.6_5', '0.7_1', '0.7_2', '0.7_3', '0.7_4', '0.7_5', '0.8_1', '0.8_2', '0.8_3', '0.8_4', '0.8_5', '0.9_1', '0.9_2', '0.9_3', '0.9_4', '0.9_5']
    for setting in settings: 
        print("==========================================================================")
        print("Running experiment for setting", setting)
        print("==========================================================================")
        for seed in [1,42,89,23,113]:
            print("Running for seed", seed, "of experiment", setting) 
            train_acc, val_acc = train_optimal(setting, seed, df)
            s = setting.split('_')
            p_s, a_s = float(s[0]), int(s[1])
            wandb.log({'p_s': p_s, 'a_s': a_s, 'seed': seed, 'train_accuracy': train_acc, 'val_accuracy': val_acc})
            print("Seed completed execution!", seed, setting)
            print("------------------------------------------------------------------")
        print("Experiment complete", setting)