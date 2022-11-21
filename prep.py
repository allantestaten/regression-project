import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pydataset import data
import acquire 
from sklearn.model_selection import train_test_split

# ------------------- ZILLOW DATA -------------------

def get_zillow_data():
    '''
    This function reads in zillow data from local copy as a df
    '''
        
    # Reads local copy of csv 
    df = pd.read_csv('zillow.csv')

    # renaming column names to more readable format
    df = df.rename(columns = {'bedroomcnt':'beds', 'roomcnt':'total_rooms',
                              'bathroomcnt':'baths', 
                              'calculatedfinishedsquarefeet':'sqft',
                              'taxvaluedollarcnt':'property_value', 
                              'yearbuilt':'year_built',})
    
    return df
    
def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''
    
    # removing properties based on what I beleive is a realistic single family home 
    df = df[df.baths <= 6]
    
    df = df[df.beds <= 6]

    df = df[df.sqft < 2500]

    df = df[df['property_value']< 1_000_000]

    df = df.replace(0, np.nan, inplace=False)

    df = df.dropna()


    # Convert binary categorical variables to objects with name of location
    cleanup_fips = {"fips":{6037: 'Los Angeles CA', 6059:'Orange County CA', 6111: 'Ventura County CA'} }    
    df = df.replace(cleanup_fips)
    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)      
    
    return train, validate, test    

def wrangle_zillow():
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = prepare_zillow(get_zillow_data())
    
    return train, validate, test