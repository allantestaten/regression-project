import pandas as pd
import numpy as np
from env import get_db_url

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



