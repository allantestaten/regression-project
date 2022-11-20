#imports used for explore file
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
from tabulate import tabulate

def rooms_count(train):
    #Feature Engineering adding an additional column
    train['rooms_count']= train['baths']+train['beds']

    # reset index 
    train=train.reset_index()  

    #Rounding up in rooms count 
    train['rooms_count'] = train['rooms_count'].apply(np.ceil)

    #Rounding up in bathrooms count 
    train['baths'] = train['baths'].apply(np.ceil)

    return train  
    

def statistic_table(df):
    ''' This function will create a table of information '''
    #calculating median of property values 
    median = df.property_value.median() 

    #calculating mean of property values 
    mean = df.property_value.mean()

    # difference between mean and median 
    difference = mean - median

    #provides data for table
    df = [["Median", median], 
        ["Mean", mean],
        ["Difference", difference]]
        
    #define header names
    col_names = ["Metric", "Value"]
  
    #display table
    print(tabulate(df, headers=col_names))   

def value_counts(df):
    '''this function will provide the value counts for the counties'''

    # Count of properties in each County
    return df.fips.value_counts()
    

def boxplot(df,column,Title):
    ''' this function will provide a boxplot of the data'''
    # creating boxplot 
    sns.boxplot(df[column]).set(title= Title)

def two_variable_boxplots(df,x,y,Title):
    # box plot Bedrooms vs Property value 
    sns.boxplot(data=df, x=df[x], y=df[y]).set(title= Title)

def hists(train):
    '''this function will produce histograms of property values for each county'''
    sns.histplot(data=train, x="property_value", hue="fips")


def variances(train):
    '''this function will provide the variances of the properties by county'''
    la = train[train.fips == 'Los Angeles CA'].property_value
    oc = train[train.fips == 'Orange County CA'].property_value
    vc = train[train.fips == 'Ventura County CA'].property_value

    # variance of prices in La County
    var_la = la.var()

    # variance of prices in Orange County
    var_oc = oc.var()

    # variance of prices in Ventura County
    vc_var= vc.var()

    print(f'Los Angeles County Property Value Variance = {var_la:.4f}')
    print(f'Orange County Property Value Variance = {var_oc:.4f}')
    print(f'Ventura County Property Value Variance = {vc_var:.4f}')
    

def stats_property_location(train):
    '''this function will provide results of statistical test'''   

    # creating dataframe for each county's property values 
    la = train[train.fips == 'Los Angeles CA'].property_value
    oc = train[train.fips == 'Orange County CA'].property_value
    vc = train[train.fips == 'Ventura County CA'].property_value

    # results of statistical test 
    results, p = stats.kruskal(la, oc, vc)

    # print results of statistical test 
    print(f'Kruska Result = {results:.4f}')
    print(f'p = {p}')

def scatter_plot(train):
    '''this function will produce a scatter plot of data'''
    #visualization of sqft vs property value 
    sns.regplot(x="sqft",
                y="property_value", 
                data=train).set(title='Sqft and Value')

def correlation_stat_test(df,column):
    '''this function will produce results of pearsonr test'''

    #pearsonr test 
    corr, p = stats.pearsonr(df[column], df.property_value)
    print(f'Correlation Strength = {corr:.4f}')
    print(f'p = {p}')
 
