import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits

data = pd.read_csv("kc_house_data.csv")
train1 = data.drop(['id', 'price'],axis=1)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date ]
data['date'] = conv_dates
train1 = data.drop(['id', 'price'],axis=1)
from sklearn.cross_validation import train_test_split
x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =2)
#x_train,y_train = train1,labels
reg.fit(x_train,y_train)
original_params={'n_estimators': 2000, 'max_depth' : 6, 'min_samples_split' : 2,'learning_rate' : 0.1, 'loss' : 'ls'}
#original_params={'max_depth' : 6, 'min_samples_split' : 2,'learning_rate' : 0.1, 'loss' : 'ls'}
params = dict(original_params)
from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(x_train, y_train)
def input_to_one_hot(dt):
    # initialize the target vector with zero values
    enc_input = np.zeros(19)
    # set the numerical input as they are
    enc_input[0] = dt['bedrooms']
    enc_input[1] = dt['bathrooms']
    enc_input[2] = dt['sqft_living']
    enc_input[3] = dt['sqft_lot']
    enc_input[4] = dt['floors']
    enc_input[5] = dt['grade']
    enc_input[6] = dt['sqft_above']
    enc_input[7] = dt['sqft_basement']
    enc_input[8] = dt['yr_built']
    enc_input[9] = dt['yr_renovated']
    enc_input[10] = dt['zipcode']
    enc_input[11] = dt['lat']
    enc_input[12] = dt['long']
    return enc_input

import pickle
model = 'model.pkl'
model_pkl = open(model,'wb')
pickle.dump(clf,model_pkl)
model_pkl.close()






