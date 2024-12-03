import numpy as np
from numpy.typing import NDArray
import pickle
from urllib.parse import uses_params
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

def saveSKLModel(fn:str, model) -> None:
    '''save SKLearn model as pickle'''
    with open(fn, 'wb') as f:
        pickle.dump(model, f)

# Convert one-hot to categorical
def onehot2cat(y: NDArray) -> NDArray:
    return np.argmax(y, axis=1)

def loadDataset(fn:str, toCat:bool=False) -> NDArray:
    '''load dataset'''
    with open(fn, 'rb') as f:
        data = pickle.load(f)
        
    X = data['X'] 
    if toCat: y = onehot2cat(data['Y'])
    else:     y = data['Y'] 
    
    return X, y

fnt = 'wtdt-part.pickle'
X, y = loadDataset(fnt, toCat=True)

C = 0.22984652084449086
kernel = "linear"
#degree = best_params["degree"]
#gamma = best_params["gamma"]
#coef0 = best_params["coef0"]
max_iter = 10
scaler = 'MinMaxScaler'

steps = []
if scaler == "StandardScaler":
    steps.append
if scaler == "MinMaxScaler":
    steps.append(("scaler", MinMaxScaler()))    

# Create the SVM model
steps.append(("svc", SVC(
    C=C,
    kernel=kernel,
    #degree=degree,
    #gamma=gamma,
    #coef0=coef0,
    max_iter=max_iter,
    random_state=42
)))

pipeline = Pipeline(steps)

# Train the pipeline on the entire training set
pipeline.fit(X, y)

saveSKLModel("T1-SVM.pickle", pipeline)
try:
    print("Sending model to Dropzone...")
    with open("autoSender.py") as file:
        exec(file.read())
except FileNotFoundError:
    print("The autoSender.py file was not found.")