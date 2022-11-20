import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
import sklearn.preprocessing


def model_columns(train,validate,test):
    '''This function will provide my models with the correct features to run for their x and y values'''

    # columns used as independent variables for modeling
    x_cols = ['baths','sqft']
    y_train = train['property_value']

    # changing y train into a dataframe to append the new column with predicted values 
    y_train = pd.DataFrame(y_train)

    # assigning features to x/y train, validate and test 
    X_train = train[x_cols]

    X_validate = validate[x_cols]
    y_validate = validate['property_value']

    # changing y train into a dataframe to append the new column with predicted values 
    y_validate= pd.DataFrame(y_validate)


    X_test = test[x_cols]
    y_test = test['property_value']

    return X_train, X_validate, y_train, y_validate, X_test, y_test


def scaling(X_train,X_validate,X_test):
    '''function will scale features and assign values to x,y train, validate and test'''
    # applying scaling to all the data splits.
    scaler = sklearn.preprocessing.RobustScaler()
    scaler.fit(X_train)

    # transforming train, validate and test datasets
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

def baseline_model(y_train,y_validate):
    '''this function will create my baseline model'''

    # Predict property_value_pred_mean
    prop_value_pred_mean = y_train['property_value'].mean()
    y_train['prop_value_pred_mean'] = prop_value_pred_mean
    y_validate['prop_value_pred_mean'] = prop_value_pred_mean

    # compute prop_value_pred_median
    prop_value_pred_median = y_train['property_value'].median()
    y_train['prop_value_pred_median'] = prop_value_pred_median
    y_validate['prop_value_pred_median'] = prop_value_pred_median

    # RMSE of prop_value_pred_median
    rmse_baseline_train = mean_squared_error(y_train.property_value, y_train.prop_value_pred_median)**(1/2)
    rmse_baseline_validate = mean_squared_error(y_validate.property_value, y_validate.prop_value_pred_median)**(1/2)

    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_baseline_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_baseline_validate, 2))

def linearOLS(X_train,X_validate, y_train,y_validate):
    '''this function will create my linear regression OLS model'''
    # create the model object
    lm = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train, y_train.property_value)

    # predict train
    y_train['property_value_pred_lm'] = lm.predict(X_train)

    # evaluate: rmse
    rmse_linearreg_train = mean_squared_error(y_train.property_value, y_train.property_value_pred_lm)**(1/2)

    # predict validate
    y_validate['property_value_pred_lm'] = lm.predict(X_validate)

    # evaluate: rmse
    rmse_linearreg_validate = mean_squared_error(y_validate.property_value, y_validate.property_value_pred_lm)**(1/2)

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_linearreg_train, 
      "\nValidation/Out-of-Sample: ", rmse_linearreg_validate)

def lassolars(X_train,X_validate, y_train,y_validate):
    # create the model object
    lars = LassoLars(alpha=1.0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lars.fit(X_train, y_train.property_value)

    # predict train
    y_train['property_value_pred_lars'] = lars.predict(X_train)

    # evaluate: rmse
    rmse_laso_lars_train = mean_squared_error(y_train.property_value, y_train.property_value_pred_lars)**(1/2)

    # predict validate
    y_validate['property_value_pred_lars'] = lars.predict(X_validate)

    # evaluate: rmse
    rmse_laso_lars_validate = mean_squared_error(y_validate.property_value, y_validate.property_value_pred_lars)**(1/2)

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_laso_lars_train, 
      "\nValidation/Out-of-Sample: ", rmse_laso_lars_validate)

def tweedie(X_train,X_validate,y_train,y_validate):
    # create the model object
    glm = TweedieRegressor(power=1, alpha=0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train, y_train.property_value)

    # predict train
    y_train['property_value_pred_glm'] = glm.predict(X_train)

    # evaluate: rmse
    rmse_tweedie_train = mean_squared_error(y_train.property_value, y_train.property_value_pred_glm)**(1/2)

    # predict validate
    y_validate['property_value_pred_glm'] = glm.predict(X_validate)

    # evaluate: rmse
    rmse_tweedie_validate = mean_squared_error(y_validate.property_value, y_validate.property_value_pred_glm)**(1/2)

    print("RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample: ", rmse_tweedie_train, 
      "\nValidation/Out-of-Sample: ", rmse_tweedie_validate)

def linear_reg(X_train,X_validate,y_train,y_validate):
    '''this function will create a polynomial regression that will be used in the linear regression'''
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=4)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate)
    
    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.property_value)

    # predict train
    y_train['property_value_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: rmse
    rmse_linear_train = mean_squared_error(y_train.property_value, y_train.property_value_pred_lm2)**(1/2)

    # predict validate
    y_validate['property_value_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_linear_validate = mean_squared_error(y_validate.property_value, y_validate.property_value_pred_lm2)**(1/2)

    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_linear_train, 
      "\nValidation/Out-of-Sample: ", rmse_linear_validate)

