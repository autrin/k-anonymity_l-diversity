# from ucimlrepo import fetch_ucirepo 
import pandas as pd

# fetch dataset 
# adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
# X = adult.data.features 
# y = adult.data.targets 
# print(X) 
    # There are 14 features
# metadata 
# print(adult.metadata) 
  
# variable information 
# print(adult.variables) 

# Define column names based on the description from 'adult.names'
column_names = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "salary"
]

adult_data = pd.read_csv('k-anonymity_l-diversity/adult.data', header=None, names=column_names, na_values=" ?")

adult_test = pd.read_csv('k-anonymity_l-diversity/adult.test', header=None, names=column_names, skiprows=1, na_values=" ?")

print(adult_data.head())
print(adult_test.head())

with open('k-anonymity_l-diversity/adult.names', 'r') as f:
    names_content = f.read()

# Print the content of the 'adult.names' file
# print(names_content)
