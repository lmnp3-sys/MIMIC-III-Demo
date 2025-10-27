"""
LMNP
MIMIC III Project
08/24/25

Medical Information Mart for Intensive Care:
- Large, publicly available dataset developed by the MIT lab
for Computational Physiology.
- Contains de-identifies health data from > 40k ICU patients
admitted to the Beth Israel Deaconess Medical Center (Boston)
between 2001 and 2012.
    Includes:
        Demographics
        Vital signs
        Lab test results
        Procedures
        Medications
        Notes from caregivers
        Mortality data

    Column:
    SUBJECT_ID                  Unique patient identifier
    HADM_ID                     Unique hospital admission ID
    ADMITTIME / DISCHTIME       Timestamps for admission and discharge
    DEATHTIME                   Time of death
    ADMISSION_TYPE              Type of admission (EMG, ELECTIVE, URGENT)
    ADMISSION_LOCATION          Where the patient came from (EMG room, clinic)
    DISCHARGE_LOCATION          Where they went after discharge (home, rehab)
    INSURANCE, LANG,            Demographic info
    RELIGION, MARITAL_STATUS,
    ETHNICITY
    DIAGNOSIS                   Free-text primary diagnosis

Some relevant questions:
1. What is the distribution of admission type (EMG, URGENT, ELECTIVE?
2. Does admission type correlate with in-hospital mortality?
3. What are the most common primary diagnoses listed for ICU admissions?
"""

# Open data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
# aren't critical errors so got rid of the warnings


'''1. What is the distribution of admission type (EMG, URGENT, ELECTIVE?'''

df = pd.read_csv("ADMISSIONS.csv", parse_dates=['admittime', 'dischtime','deathtime'])
# when data contains date or datetime values like above, we need to tell pandas
# to interpret them as actual datetime objects -- that's what parse_dates does

# print(df.head()) # preview few rows
# print(df.info()) # data types or non-null values
# print(df.describe()) # for numeric summaries
# print(df.value_counts()) # understand distributions

print(df.head())
print(df['admission_type'].value_counts(dropna=False)) #check all admission types including missing

missing_adm_type = df['admission_type'].isnull().sum()
print(f"Missing Admission Types: {missing_adm_type}")

# is missing is small, drop these rows
df = df.dropna(subset=['admission_type'])

# strip whitespace in case the values have spaces
df['admission_type'] = df['admission_type'].str.strip()

# calculate distribution and %
counts = df['admission_type'].value_counts()
print(counts)

percentages = df['admission_type'].value_counts(normalize=True) * 100
percentages = percentages.round(2)
# normalize=True --- Proportions (floats)
# normalize=False --- Counts (integers)

plt.figure(figsize=(8, 5))
ax = sns.barplot(x=counts.index, y=counts.values, palette='deep', hue=None)
# counts.index == admission type names
# counts.values == number of times each type appeared

# # Add value on top of each bar
# for i, value in enumerate(percentages):
#     plt.text(i, value, str(value) + "%", ha='center', va='bottom', fontsize=10)

for admin_type, value in percentages.items():
    plt.text(admin_type, value, f"{value:.2f}%", ha='center', va='bottom', fontsize=10)

plt.title('Admission Type Distribution')
plt.xlabel('Admission Type')
plt.ylabel('No. Admissions')
plt.tight_layout()
plt.show()

print("-" * 30)


'''2. Use Carmer's V to identify corr between two categorical columns'''
# check correlation between two qualitative data using chi2
counts = df['discharge_location'].value_counts()
print(counts)

mortality_cate = ['DEAD/EXPIRED']
df['mortality'] = df['discharge_location'].apply(lambda x: 1 if x in mortality_cate else 0)

print(df['mortality'].value_counts())
print("-" * 30)

def cramers_v(x, y):
    '''
    n - total number of observations
    k and r - number of categories in the two variables
    x and y are Pandas series
    '''
    # contingency table
    confusion_matrix = pd.crosstab(x, y)

    # chi2 test
    # p = p-value
    # dof = degree of freedom
    # expected - expected frequencies
    chi2, p, dof, expected = chi2_contingency(confusion_matrix)

    # sample size
    n = confusion_matrix.sum().sum()

    # minimum dimension - 1
    k = min(confusion_matrix.shape)

    # Cramer's v formula
    return np.sqrt(chi2 / n * (k - 1))


def interpret_cramers_v(v):
    if v < 0.1:
        return "Very weak association"
    elif v < 0.3:
        return "Weak association"
    elif v < 0.5:
        return "Moderate association"
    else:
        return "Strong association"

result = cramers_v(df['admission_type'], df['mortality'])
print(f"Cramer's V between admission type and mortality: {result:.3f}\n -> {interpret_cramers_v(result)}")
print("-" * 30)

'''3. What are the most common primary diagnoses listed for EMERGENCY admissions?'''
# group hadm_id with admission_location for admissions
# group diagnosis with hadm_id
emerg = df[df['admission_location'] == "EMERGENCY ROOM ADMIT"]

# count number of each diagnosis
common_diag = emerg['diagnosis'].value_counts()

most_common = common_diag.idxmax()
most_common_count = common_diag.max()
print(f"Common diagnosis for EMG: {most_common} with {most_common_count}")
print("-" * 30)


'''4. What % of patients died during or shortly after their ICU stay?'''


