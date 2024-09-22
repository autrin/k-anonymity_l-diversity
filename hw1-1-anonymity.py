import pandas as pd
import numpy as np
import math

# Define column names based on the description from 'adult.names'
column_names = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "salary"
]

adult_data = pd.read_csv('adult.data', header=None, names=column_names, na_values=" ?")

adult_test = pd.read_csv('adult.test', header=None, names=column_names, skiprows=1, na_values=" ?") # skipping the first line bc it was not important

# Replace NaN in the 'occupation' column with 'Unknown'
adult_data.fillna({'occupation':'Unknown'}, inplace=True)
print(adult_data.isnull().sum())
print(adult_data.head())
print(adult_test.head())

with open('adult.names', 'r') as f:
    names_content = f.read()

# Print the content of the 'adult.names' file
# print(names_content)
print(adult_data['salary'])
adult_data['age'].sort_values(ascending=False)
# Generalize the age based on the hierarchy in hierarchy.txt
def generalize_age(age, level):
    if level == 1:
        # Level 1: No generalization, return the precise age
        return age
    elif level == 2:
        # Level 2: Group into small age ranges
        if 17 <= age <= 19:
            return '17-19'
        elif 20 <= age <= 29:
            return '20-29'
        elif 30 <= age <= 39:
            return '30-39'
        elif 40 <= age <= 49:
            return '40-49'
        elif 50 <= age <= 59:
            return '50-59'
        elif 60 <= age <= 69:
            return '60-69'
        elif 70 <= age <= 79:
            return '70-79'
        elif 80 <= age <= 89:
            return '80-89'
        elif age >= 90:
            return '90+'
    elif level == 3:
        # Level 3: Group into medium ranges
        if 17 <= age <= 29:
            return '17-29'
        elif 30 <= age <= 49:
            return '30-49'
        elif 50 <= age <= 69:
            return '50-69'
        elif age >= 70:
            return '70+'
    elif level == 4:
        # Level 4: Group into broad categories
        if 17 <= age <= 29:
            return 'Young Adult (17-29)'
        elif 30 <= age <= 49:
            return 'Middle Age (30-49)'
        elif 50 <= age <= 69:
            return 'Senior (50-69)'
        elif age >= 70:
            return 'Elderly (70+)'
    elif level == 5:
        # Level 5: Generalize to "All Ages"
        return 'All Ages'
    
    return 'All Ages'

age = 91
generalized_age = generalize_age(age, 2)
print(f'Generalized age at level 2: {generalized_age}')

age = 35
generalized_age = generalize_age(age, 3)
print(f'Generalized age at level 3: {generalized_age}')

adult_data['education'].unique()

# Education generaliztion
def generalize_education(education, level):
    education = education.strip()
    
    if level == 1:
        return education
    
    elif level == 2:
        if education in ['Preschool', '1st-4th', '5th-6th']:
            return 'Primary School'
        elif education in ['7th-8th', '9th']:
            return 'Middle School'
        elif education in ['10th', '11th', '12th', 'HS-grad']:
            return 'High School'
        elif education in ['Assoc-voc', 'Prof-school', 'Some-college']:
            return 'Vocational School'
        elif education in ['Assoc-acdm', 'Bachelors']:
            return 'Undergraduate School'
        elif education in ['Masters', 'Doctorate']:
            return 'Graduate School'
    
    elif level == 3:
        if education in ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad']:
            return 'Compulsory School'
        elif education in ['Assoc-voc', 'Prof-school', 'Some-college', 'Assoc-acdm', 'Bachelors']:
            return 'Basic Degree'
        elif education in ['Masters', 'Doctorate']:
            return 'Advanced Degree'
    
    elif level == 4:
        if education == 'Compulsory School':
            return 'Compulsory School'
        elif education in ['Basic Degree', 'Advanced Degree']:
            return 'Advanced School'
    
    elif level == 5:
        return 'Education'

    return 'Education'

education = 'Bachelors'
generalized_education = generalize_education(education, 3)
print(f'Generalized education at level 3: {generalized_education}')

# Marital Status Generalization
def generalize_marital_status(status, level):
    status = status.strip()
    
    if level == 1:
        return status
    
    elif level == 2:
        if status in ['Separated', 'Married-spouse-absent']:
            return 'Separated-Married'
        elif status in ['Never-married', 'Widowed', 'Divorced']:
            return 'Single'
        elif status in ['Married-civ-spouse', 'Married-AF-spouse']:
            return 'Married-Together'
    
    elif level == 3:
        if status in ['Separated', 'Married-spouse-absent', 'Married-civ-spouse', 'Married-AF-spouse']:
            return 'Married'
        elif status in ['Never-married', 'Widowed', 'Divorced']:
            return 'Not-Married'
    
    elif level == 4:
        return '*'
    
    return '*'

marital_status = 'Married-civ-spouse'
generalized_status = generalize_marital_status(marital_status, 3)
print(f'Generalized marital status at level 3: {generalized_status}')

# Race Generalization
def generalize_race(race, level):
    race = race.strip()
    
    if level == 1:
        return race
    
    elif level == 2:
        if race == 'White':
            return 'Caucasian'
        elif race == 'Asian-Pac-Islander':
            return 'Asian'
        elif race == 'Amer-Indian-Eskimo':
            return 'Indigenous'
        elif race == 'Black':
            return 'African Descent'
        elif race == 'Other':
            return 'Other'
    
    elif level == 3:
        if race in ['White', 'Black']:
            return 'Western Origin'
        elif race in ['Asian-Pac-Islander', 'Amer-Indian-Eskimo']:
            return 'Eastern Origin'
        elif race == 'Other':
            return 'Other'
    
    elif level == 4:
        if race in ['White', 'Black']:
            return 'Western Origin'
        else:
            return 'Non-Western Origin'
    
    elif level == 5:
        return 'Race'
    return 'Race'
race = 'Asian-Pac-Islander'
generalized_race = generalize_race(race, 3)
print(f'Generalized race at level 3: {generalized_race}')
# create a diffent dataset with the generalized data
generalized_data = adult_data.copy()
generalized_data # will be generalized in the next steps
def generalize_QIs(data, educationLevel, maritalStatusLevel, raceLevel, ageLevel):
    data['age'] = data['age'].apply(lambda x : generalize_age(x, ageLevel)) 
    data['education'] = data['education'].apply(lambda x : generalize_education(x, educationLevel))
    data['marital_status'] = data['marital_status'].apply(lambda x : generalize_marital_status(x, maritalStatusLevel))
    data['race'] = data['race'].apply(lambda x : generalize_race(x, raceLevel))
    return data

def check_k_anonymity_le50(data, k1):
    """
    Check if the dataset meets the k-anonymity requirement.
    
    :param data: The dataset to check.
    :param k1: The k-anonymity level for users with salaries ≤ 50K.
    :param k2: The k-anonymity level for users with salaries > 50K.
    :return: True if the dataset meets the k-anonymity requirement, False otherwise.
    """
    # Print unique values in the salary column for debugging
    # print("Unique salary values:", data['salary'].unique())
    
    # Split the dataset into two parts based on the salary
    data_le_50k = data[data['salary'] == ' <=50K']
    
    # Print the number of rows in each part for debugging
    # print("Number of rows with salary <= 50K:", len(data_le_50k))
    # print("Number of rows with salary > 50K:", len(data_gt_50k))
    
    # Group by QIs for each part
    grouped_le_50k = data_le_50k.groupby(['age', 'education', 'marital_status', 'race'])
    
    # Check k-anonymity for users with salaries ≤ 50K
    for _, group in grouped_le_50k:
        if len(group) < k1:
            return False, data_le_50k
    
    
    
    return True, data_le_50k

def check_k_anonymity_gt50(data, k2):
    """
    Check if the dataset meets the k-anonymity requirement.
    
    :param data: The dataset to check.
    :param k1: The k-anonymity level for users with salaries ≤ 50K.
    :param k2: The k-anonymity level for users with salaries > 50K.
    :return: True if the dataset meets the k-anonymity requirement, False otherwise.
    """
    # Print unique values in the salary column for debugging
    # print("Unique salary values:", data['salary'].unique())
    
    # Split the dataset into two parts based on the salary
    data_gt_50k = data[data['salary'] == ' >50K']
    
    # Print the number of rows in each part for debugging
    # print("Number of rows with salary <= 50K:", len(data_le_50k))
    # print("Number of rows with salary > 50K:", len(data_gt_50k))
    
    # Group by QIs for each part
    grouped_gt_50k = data_gt_50k.groupby(['age', 'education', 'marital_status', 'race'])
    
    
    # Check k-anonymity for users with salaries > 50K
    for _, group in grouped_gt_50k:
        if len(group) < k2:
            return False, data_gt_50k
    
    return True, data_gt_50k
# Generalizing the data with salary of <=50k while meeting the k-anonymity requirements
# Max level of generalization for Race, Education, Marital Status, Age is 5, 5, 4, 5 respectively.
# Also attributes can have different levels of generalization in the same dataset
generalized_data = adult_data.copy()
k1 = 10
k2 = 5

found = False
global gen_levels
for i in range(1, 6):
    for j in range(1, 6):
        for k in range(1, 5):
            for l in range(1, 6):
                generalized_data = adult_data.copy()
                generalized_data = generalize_QIs(generalized_data, i, j, k, l)
                
                # Check if the dataset meets the k-anonymity requirement
                is_k_anonymous_le_50k, table_le_50k= check_k_anonymity_le50(generalized_data, k1)

                if is_k_anonymous_le_50k:
                    table_le_50k.to_csv(f'hw1-1-generalized_data_le_50k_{i}_{j}_{k}_{l}.csv', index=False)
                    print(f'Generalization levels for <=50k dataset: Race={k}, Education={i}, Marital Status={j}, Age={l}')
                    gen_levels_le_50k = [i, j, k, l]
                    generalization_levels_le50k = {
                        'age': l,
                        'education': i, 
                        'marital_status': j, 
                        'race': k 
                    }
                    found = True
                    break
            if found:
                break
        if found:
            break
    if found:
        break
# Generalizing the data with salary of >50k while meeting the k-anonymity requirements
found = False
for j in range(1, 6):
    for l in range(1, 6):
        for k in range(1, 5):
            for i in range(1, 6):
                generalized_data = adult_data.copy()
                generalized_data = generalize_QIs(generalized_data, i, j, k, l)
                
                # Check if the dataset meets the k-anonymity requirement
                is_k_anonymous_gt_50k, table_gt_50k= check_k_anonymity_gt50(generalized_data, k2)

                if is_k_anonymous_gt_50k:
                    table_gt_50k.to_csv(f'hw1-1-generalized_data_gt_50k_{i}_{j}_{k}_{l}.csv', index=False)
                    print(f'Generalization levels for >50k dataset: Race={k}, Education={i}, Marital Status={j}, Age={l}')
                    gen_levels_gt_50k = [i, j, k, l]
                    generalization_levels_gt50k = {
                        'age': l,
                        'education': i, 
                        'marital_status': j, 
                        'race': k 
                    }
                    found = True
                    break
            if found:
                break
        if found:
            break
    if found:
        break
print(f'Is k-anonymous: {is_k_anonymous_le_50k} for <=50k dataset')
print(f'Is k-anonymous: {is_k_anonymous_gt_50k} for >50k dataset')
print("Table for users with salaries ≤ 50K:")

table_le_50k
# For table_le_50k
q_block_counts_le_50k = table_le_50k.groupby(['age', 'education', 'marital_status', 'race']).size().reset_index(name='counts')
print("\n Q* block counts for users with salaries ≤ 50K:")
print(q_block_counts_le_50k)


# For table_gt_50k
q_block_counts_gt_50k = table_gt_50k.groupby(['age', 'education', 'marital_status', 'race']).size().reset_index(name='counts')
print("\n Q* block counts for users with salaries > 50K:")
print(q_block_counts_gt_50k)
print("Table for users with salaries > 50K:")

table_gt_50k
# Verify group sizes for users with salaries ≤ 50K
if (q_block_counts_le_50k['counts'] >= k1).all():
    print("All Q* blocks for users with salaries ≤ 50K meet the k-anonymity requirement.")
else:
    print("Some Q* blocks for users with salaries ≤ 50K do not meet the k-anonymity requirement.")

# Verify group sizes for users with salaries > 50K
if (q_block_counts_gt_50k['counts'] >= k2).all():
    print("All Q* blocks for users with salaries > 50K meet the k-anonymity requirement.")
else:
    print("Some Q* blocks for users with salaries > 50K do not meet the k-anonymity requirement.")
num_attr = len(table_gt_50k.columns)
num_attr
len(table_le_50k.columns)
len(generalized_data.columns)
# Calculate the distortion

def calculate_distortion(gen_levels):
    # For all atributes: generalization level / max generalization level
    sum_of_generalizations = gen_levels['age']/5 + gen_levels['marital_status']/4 + gen_levels['education']/5 + gen_levels['age']/5
    sum_of_generalizations = sum_of_generalizations + (15 - 4) # Add the genralization level of the other attributes/columns which are 1's
    return sum_of_generalizations / num_attr

print(f'Distortion for data >50k: {calculate_distortion(generalization_levels_gt50k)}')
print(f'Distortion for data <=50k: {calculate_distortion(generalization_levels_le50k)}')
# On average, the attributes have been generalized to about 94.33% of their maximum generalization levels.
# Calculate precision
table_gt_50k_len = len(table_gt_50k.index)
table_gt_50k_len
table_le_50k_len = len(table_le_50k.index)
table_le_50k_len
def calculate_precision(data, generalization_levels, hierarchy_depths):
    """
    Calculate the precision.
    
    :param data: pandas DataFrame of the dataset
    :param generalization_levels: Dictionary of generalization levels for each attribute
    :param hierarchy_depths: Dictionary of the depth of the value generalization hierarchy for each attribute
    :return: Precision value
    """
    N_A = len(generalization_levels)  # Number of QI attributes
    PT = len(data.index)  # Total number of records
    sum_generalization_height = 0

    # Calculate the total generalization height normalized by the depth of the hierarchy
    for attribute, level in generalization_levels.items():
        depth = hierarchy_depths[attribute]
        for _ in range(PT):  # The generalization level is uniform across all records
            sum_generalization_height += level / depth

    precision = 1 - (sum_generalization_height / (PT * N_A))
    return precision

hierarchy_depths = {
    'age': 5, 
    'education': 5,
    'marital_status': 4,
    'race': 5 
}

precision_value = calculate_precision(table_le_50k, generalization_levels_le50k, hierarchy_depths)
print(f'Precision for <=50k dataset: {precision_value}')

precision_value = calculate_precision(table_gt_50k, generalization_levels_gt50k, hierarchy_depths)
print(f'Precision for >50K dataset: {precision_value}')
table_gt_50k
#Calculate entropy and l-diversity
#">50K" dataset didn't meet the l-diversity requirements when l = 3:
def calculate_entropy(group):
    """ Calculate entropy for a single group of records """
    if len(group) == 0:
        return 0
    value_counts = group.value_counts()
    probabilities = value_counts / value_counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy

def check_l_diversity(data, l):
    """ Check if the dataset satisfies l-diversity """
    entropy_threshold = math.log(l)  # Define the minimum entropy threshold
    fails = 0  # Track number of failures
    
    # Group data by QIs and calculate entropy for each group
    grouped = data.groupby(['age', 'education', 'marital_status', 'race'])
    for name, group in grouped:
        entropy = calculate_entropy(group['occupation'])
        if entropy < entropy_threshold:
            print(f'Group {name} fails to meet the entropy l-diversity with entropy {entropy:.4f}')
            fails += 1
    if fails == 0:
        print("All groups meet the l-diversity requirement.")
        return True
    else:
        print(f"{fails} groups fail to meet the l-diversity requirement.")
        return False

l = 3
print("Checking >50K dataset for l-diversity:")
satisfied_l_diversity_gt_50k = check_l_diversity(table_gt_50k, l)

# The generalized dataset with salary <=50k meets the requirements of l-diversity:
print("Checking <=50K dataset for l-diversity:")
satisfied_l_diversity_le_50k = check_l_diversity(table_le_50k, l)
# Increase the generalization level for a specific attribute, say 'education' in table_gt_50k
def apply_adjusted_generalization(data, generalization_levels, attribute, condition_value, generalization_function, condition_attribute='education', generalization_level=4):
    """
    Apply generalization based on a condition on the attribute.
    condition_value:  the value you're checking against
    """
    def generalized_value(row, attribute, condition_value, generalization_function, generalization_level):
        if row[condition_attribute] == condition_value and generalization_level > 0:
            return generalization_function(row[attribute], generalization_level)
        return row[attribute]
    
    # Apply generalization
    data[attribute] = data.apply(lambda row: generalized_value(row, attribute, condition_value, generalization_function, generalization_level), axis=1)
    
    # Update the generalization level
    generalization_levels[attribute] = generalization_level
    return data


# Apply adjusted generalization
generalized_data_gt_50k_adjusted = apply_adjusted_generalization(
    data=table_gt_50k.copy(), 
    generalization_levels=generalization_levels_gt50k, 
    attribute='marital_status', 
    condition_value='Non-Western Origin',  # The group condition (you can adjust as needed)
    generalization_function=generalize_education, 
    condition_attribute='race',  # This ensures that the generalization happens only for the 'Non-Western Origin' group
    generalization_level=generalization_levels_gt50k['marital_status'] + 1  # Increment education's generalization level
)


# generalized_data_gt_50k_adjusted = apply_adjusted_generalization(
#     data=table_gt_50k.copy(), 
#     generalization_levels=generalization_levels_gt50k, 
#     attribute='race', 
#     condition_value='Basic Degree',  # The group condition
#     generalization_function=generalize_race, 
#     condition_attribute='education',  # This ensures that the generalization happens for the 'Basic Degree' group in education
#     generalization_level=generalization_levels_gt50k['race'] + 1  # Increment race's generalization level
# )

# Check l-diversity again
print("Rechecking >50K dataset for l-diversity after adjustment:")
satisfied_l_diversity_gt_50k_adjusted = check_l_diversity(generalized_data_gt_50k_adjusted, l)
print(satisfied_l_diversity_gt_50k_adjusted)

# Save the adjusted dataset
generalized_data_gt_50k_adjusted.to_csv('hw1-1-generalized_data_gt_50k_adjusted_for_lDiversity.csv', index=False)

# Calculate Recursive (c, ℓ)-diversity
def check_recursive_diversity(data, l, c, attribute_groups, detailed=False):
    all_diverse = True
    failed_groups = []
    for name, group in data.groupby(attribute_groups):
        occupation_counts = group['occupation'].value_counts()
        if len(occupation_counts) < l:
            if detailed:
                failed_groups.append(name)
            all_diverse = False
            continue
        
        sorted_counts = occupation_counts.sort_values(ascending=False)
        threshold = c * sorted_counts.iloc[l-1] if len(sorted_counts) >= l else 0
        if any(sorted_counts.iloc[:l-1] > threshold):
            if detailed:
                failed_groups.append(name)
            all_diverse = False

    return all_diverse, failed_groups
def auto_adjust_generalization(data, generalization_levels, l, c, attribute_groups, max_attempts=10):
    attempts = 0
    while attempts < max_attempts:
        # Check diversity and get detailed information about failing groups
        diverse, failed_groups = check_recursive_diversity(data, l, c, attribute_groups, detailed=True)

        if diverse:
            print("All groups meet the recursive (c, l)-diversity requirement.")
            break
        else:
            print(f"Adjusting generalization levels due to failures in groups: {failed_groups}")
            no_more_adjustments = True  # Track whether we can adjust any further
            
            # Increase generalization for failed groups
            for group in failed_groups:
                group_conditions = dict(zip(attribute_groups, group))  # Map group attributes to values

                # Adjust the generalization based on the specific group that failed
                for attribute, value in group_conditions.items():
                    current_level = generalization_levels[attribute]
                    max_depth = hierarchy_depths[attribute]

                    print(f"Before Adjustment: Attribute: {attribute}, Current Level: {current_level}, Max Depth: {max_depth}")

                    # If we're not yet at the max generalization level, increase the level
                    if current_level < max_depth:
                        # Optionally adjust the increment based on difficulty of group generalization
                        new_generalization_level = current_level + 1

                        # Apply the adjustment to generalize more
                        data = apply_adjusted_generalization(
                            data,
                            generalization_levels,
                            attribute=attribute,
                            condition_value=value,
                            generalization_function=get_generalization_function(attribute),
                            condition_attribute=attribute,
                            generalization_level=new_generalization_level  # Increase generalization by 1 level
                        )
                        
                        generalization_levels[attribute] = new_generalization_level  # Update level
                        no_more_adjustments = False  # We made at least one adjustment

                        print(f"After Adjustment: Attribute: {attribute}, New Generalization Level: {new_generalization_level}")
                    else:
                        print(f"Attribute {attribute} is already at its maximum generalization level.")
            
            # Recheck diversity after generalization adjustments in the same loop iteration
            diverse, failed_groups = check_recursive_diversity(data, l, c, attribute_groups, detailed=True)

            # If no groups fail after the adjustments, break the loop
            if diverse:
                print("All groups meet the recursive (c, l)-diversity requirement after adjustments.")
                break

            # If no attributes could be adjusted, break the loop
            if no_more_adjustments:
                print("No further adjustments possible. Exiting.")
                break
        
        attempts += 1
    
    if attempts == max_attempts:
        print("Maximum adjustment attempts reached, some groups may still fail the diversity requirements.")

def get_generalization_function(attribute):
    """
    Returns the correct generalization function based on the attribute.
    """
    if attribute == 'education':
        return generalize_education
    elif attribute == 'race':
        return generalize_race
    elif attribute == 'marital_status':
        return generalize_marital_status
    elif attribute == 'age':
        return generalize_age
    else:
        raise ValueError(f"Unknown attribute: {attribute}")
l_recursive_data_c_point5 = table_gt_50k.copy()
generalization_levels_recursive_data_c_point5 = { # Race=4, Education=4, Marital Status=2, Age=5
    'age': 5,
    'education': 4,
    'marital_status': 2,
    'race': 4
}
# When k = 5 and l = 3, c = 0.5
auto_adjust_generalization(l_recursive_data_c_point5, generalization_levels_recursive_data_c_point5, l=3, c=0.5, attribute_groups=['race', 'marital_status'])
l_recursive_data_c_1 = table_gt_50k.copy()
generalization_levels_recursive_data_c_1 = {
    'age': 5,
    'education': 4,
    'marital_status': 2,
    'race': 4
}
auto_adjust_generalization(l_recursive_data_c_1, generalization_levels_recursive_data_c_1, l=3, c=1, attribute_groups=['education', 'race'])
l_recursive_data_c_2 = table_gt_50k.copy()
generalization_levels_recursive_data_c_2 = {
    'age': 5,
    'education': 4,
    'marital_status': 2,
    'race': 4
}
# When k = 5 and l = 3, c = 2
auto_adjust_generalization(l_recursive_data_c_2, generalization_levels_recursive_data_c_2, l=3, c=2, attribute_groups=['education', 'race'])
# Calculate distortion and precision when k=5
print(f'Distortion for data with salary >50k when c=0.5: {calculate_distortion(generalization_levels_recursive_data_c_point5)}')
print("precision for data with salary >50k when c=0.5: ", calculate_precision(l_recursive_data_c_point5, generalization_levels_recursive_data_c_point5, hierarchy_depths))
print(f'Distortion for data with salary >50k when c=1: {calculate_distortion(generalization_levels_recursive_data_c_1)}')
print("precision for data with salary >50k when c=1: ", calculate_precision(l_recursive_data_c_1, generalization_levels_recursive_data_c_1, hierarchy_depths))
print(f'Distortion for data with salary >50k when c=2: {calculate_distortion(generalization_levels_recursive_data_c_2)}')
print("precision for data with salary >50k when c=2: ", calculate_precision(l_recursive_data_c_2, generalization_levels_recursive_data_c_2, hierarchy_depths))
