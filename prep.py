import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pydataset import data

# train test split from sklearn
from sklearn.model_selection import train_test_split
# imputer from sklearn

# our own acquire script:
import acquire 

# ------------------- ZILLOW DATA -------------------

def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''

    # removing outliers
    df = df[df.baths <= 8]
    
    df = df[df.beds <= 8]

    df = df[df.sqft < 5000]

    df = df[df.property_value > 0]


    # Convert binary categorical variables to numeric
    cleanup_fips = {"fips":{6037: 'Los Angeles CA', 6059:'Orange County CA', 6111: 'Ventura County CA'} }
    df = df.replace(cleanup_fips)

    #get dummies for fips 
    dummy_df = pd.get_dummies(df[['fips']], dummy_na=False, \
                              drop_first=True)
    
    #adding dummies to dataframe
    df = pd.concat([df, dummy_df], axis=1)
    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)      
    
    return train, validate, test   

def wrangle_zillow():
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = prepare_zillow(get_zillow_data())
    
    return train, validate, test