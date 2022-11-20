import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split

def train_validate_test_split(df, seed=123):
    '''
    This function takes in a dataframe and an integer for a setting a seed
    and splits the data into train, validate and test. The function returns, 
    in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    return train, validate, test  

def scale():
    # assigning features to x/y train, validate and test 
    X_train = train[x_cols]
    y_train = train['property_value']

    X_validate = validate[x_cols]
    y_validate = validate['property_value']

    X_test = test[x_cols]
    y_test = test['property_value']

    # applying scaling to all the data splits.
    scaler = sklearn.preprocessing.RobustScaler()
    scaler.fit(X_train)

    # transforming train, validate and test datasets
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

def baseline_model():
    ''' this model is creating and printing the baseline RMSE'''
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # 1. Predict property_value_pred_mean
    prop_value_pred_mean = y_train['property_value'].mean()
    y_train['prop_value_pred_mean'] = prop_value_pred_mean
    y_validate['prop_value_pred_mean'] = prop_value_pred_mean

    # 2. compute prop_value_pred_median
    prop_value_pred_median = y_train['property_value'].median()
    y_train['prop_value_pred_median'] = prop_value_pred_median
    y_validate['prop_value_pred_median'] = prop_value_pred_median

    # 3. RMSE of prop_value_pred_median
    rmse_baseline_train = mean_squared_error(y_train.property_value, y_train.prop_value_pred_median)**(1/2)
    rmse_baseline_validate = mean_squared_error(y_validate.property_value, y_validate.prop_value_pred_median)**(1/2)

    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

def linearOLS(X_train, y_train):
    '''this function provides the results of the linear regression model'''
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

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate,"\nPerformance Compared to Baseline for OLS using LinearRegression\nTraining/In-Sample: ", difference)

def lassoLars():
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

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)

def tweedie():
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

    print("RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)    

def linearReg():
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

    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)