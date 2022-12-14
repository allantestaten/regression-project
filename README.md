# Predicting California Home Prices
 
# Project Description
As a member of the Zillow Data Science team I am tasked with creating a model to improve upon the original model that has been created to predict the tax assessed value of single family homes. 
 
# Project Goal
* Construct a Machine Learning Regression model that predicts propery tax assessed values ('taxvaluedollarcnt') of Single Family Properties using attributes of the properties.
* Find the key drivers of property value for single family properties.
* Deliver a report that the data science team can read through and replicate, understand what steps were taken, why and what the outcome was.
* Make recommendations on what works or doesn't work in the prediction of home values.

 
# Initial Thoughts
 
Home value will depend heavily on the bedrooms, bathrooms and square footage of the home. 

# The Plan
 
* Acquire data from Codeup database using mySQL Workbench
 
* Prepare data
   * Created columns representing anticipated drivers that are easy for machine learning model to process
       * beds
       * baths
       * sqft
       * fips
       * property_value
       * total rooms 

 
* Explore data in search of drivers of propety value
   * Answer the following initial questions
       * What is the median home price?
       * What is the mean home price?
       * Is there a signfiicant difference in property value across the three counties?
       * Is there a correlation between square footage and property value?
       * Is there a correlation between the bathrooms and property value?
      
* Develop a Model to predict an accurate value of the property
   * Use drivers supported by statistical test results to build predictive models
   * Evaluate all models on train 
   * Evaluate top models on validate data 
   * Select the best model based on lowest Root Mean Squared Error
   * Evaluate the best model on test data
 
* Draw conclusions
 
# Data Dictionary

| Feature | Definition |
|:--------|:-----------|
|Beds| Number of bedrooms in the home|
|Baths| Number of bathrooms in the home|
|sqft| The square footage of the home|
|rooms_count| Represents the total number of rooms in the home|
|fips| The county the property is located in Los Angelese County CA, Ventura CA or Orange County CA|

# Steps to Reproduce
1) Clone this repo
2) Acquire the data from mySQL workbench database 
3) Create env file with username, password and codeup host name 
4) Include the function below in your env file
def get_db_url(db, user = username, host = host, password = password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
5) Put the data in the file containing the cloned repo
6) Run notebook
 
# Takeaways and Conclusions
- The median property value is 304,004 dollars
- Statistical evidence supports correlations between property value and bedrooms, bathrooms, total rooms and square feet of a home respectively
- Statistical testing supports a significant relationship between fips code and property value 
- There appears to be a higher frequency of homes being sold in Los Angeles compared to the other counties
- There is a significant difference in the mean property value of the homes across Los Angeles, Ventura and Orange County 


# Recommendations
* Have a meeting to discuss how to analyze features specifc to location
* Request more data to balance out the representation of transactions in Orange County and Ventura County 


# Next Steps
* Take time to build a model solely based on location features
* Look further into Los Angeles transactions impact on data