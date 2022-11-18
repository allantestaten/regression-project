import pandas as pd
import numpy as np
from env import get_db_url
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from pydataset import data
from sklearn.model_selection import train_test_split


# train test split from sklearn
from sklearn.model_selection import train_test_split
# imputer from sklearn
from sklearn.impute import SimpleImputer

#------------------- ACQUIRE OR GET ZILLOW DATA -------------------#
def new_zillow_data():
    '''
    This function reads the zillow data from the Codeup db into a df.
    '''
    # Create SQL query.
    sql_query = """SELECT bedroomcnt,bathroomcnt,fips,calculatedfinishedsquarefeet,taxvaluedollarcnt
                FROM predictions_2017
                JOIN properties_2017 using (parcelid)
                JOIN propertylandusetype as pl using (propertylandusetypeid)
                WHERE pl.propertylandusetypeid = '261'"""
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('zillow'))

    # Save data to csv 
    filepath = Path('zillow.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index =False)
    
    return df

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
#------------------- PREPARE ZILLOW DATA FOR EXPLORATION -------------------#
def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''
    
    # removing outliers and null
    df = df[df.baths <= 6]
    
    df = df[df.beds <= 6]

    df = df[df.sqft <= 2500]

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