# -*- coding: utf-8 -*-
"""CAM_DS_C101_Demo  2.1.2.ipynb

# Demonstration 2.1.2 Testing hypotheses with $t$-tests, ANOVA, and chi-square

This demonstration is divided into three sections: Hypothesis testing with a $t$-test,  hypothesis testing with ANOVA, and hypothesis testing with chi-square.

## a) Hypothesis testing with a $t$-test
Follow the demonstration to learn how to perform **hypothesis testing with a $t$-test** using Python. In this demonstration, you will learn how to:
- explore the data set
- formulate the hypotheses
- conduct a $t$-test using Python.
"""

# Import the required libraries.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import the ab_test.csv file (data set) from GitHub with a url.
url = "https://raw.githubusercontent.com/fourthrevlxd/cam_dsb/main/ab_test.csv"

# Load the data set by specifying the url.
data = pd.read_csv(url)

# View the DataFrame.
data.head()

# Drop the 'Unnamed: 0' column – redundant index.
data.drop(columns=['Unnamed: 0'],
          inplace=True)

# View the first 10 rows of the DataFrame.
print(data.shape)
data.head(10)

# Exploratory data analysis.
# Create three plots: histogram, countplot, and boxplot.
# Set the style of Seaborn for better visualisation.
sns.set_style("whitegrid")

# a) Create the histogram.
# Distribution of time spent on the page.
plt.figure(figsize=(8, 6))
sns.histplot(data=data,
             x='time_spent_on_the_page',
             hue='landing_page',
             element='step',
             stat='density',
             common_norm=False)

plt.title('Distribution of time spent on the page landing page')
plt.show()

# b) Create a barplot.
# Conversions by landing page version.
plt.figure(figsize=(8, 6))
sns.countplot(data=data,
              x='landing_page',
              hue='converted')

plt.title('Conversions by landing page version')
plt.show()

# c) Create a boxplot.
# Time spent on the page to preferred language.
plt.figure(figsize=(8, 6))
sns.boxplot(data=data,
            x='language_preferred',
            y='time_spent_on_the_page',
            hue='landing_page')

plt.title('Time spent on the landing page to preferred language')
plt.show()

"""Formulate the hypotheses as follows:
- $H_0$: There is **no difference** in time spent between the old website and the new website.
- $H_1$: There is a **significant difference** in time spent between the old website and the new website.

"""

# Import the ttest_ind function from the scipy.stats module.
from scipy.stats import ttest_ind

# Specify the control and treatment groups.
control_group = data[data['group'] == 'control']
treatment_group = data[data['group'] == 'treatment']

# T-test for time spent on the page.
t_stat_time, p_val_time = ttest_ind(control_group['time_spent_on_the_page'],
                                    treatment_group['time_spent_on_the_page'])

# Display the results.
print("T-statistic:", t_stat_time)
print("P-value:", p_val_time)

"""## b) Hypothesis testing with ANOVA
Follow the demonstration to see the importance of **hypothesis testing with ANOVA (analysis of variance)**. In this video, you will learn how to:
- create and explore the data set
- formulate the hypotheses
- conduct an ANOVA using Python.
"""

# Import the required libraries.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set a random seed for reproducibility.
np.random.seed(42)

# Create a hypothetical data set.
data = {'Design': np.repeat(['A', 'B', 'C'], 30),
        'ConversionRate': (np.random.normal(loc=2.5, scale=0.5, size=30).tolist() +
                           np.random.normal(loc=3.0, scale=0.5, size=30).tolist() +
                           np.random.normal(loc=2.8, scale=0.5, size=30).tolist())}

# Create a DataFrame.
df = pd.DataFrame(data)

# View the DataFrame.
print(df.shape)
df

# Exploratory data analysis.
# Create a histogram to visualise the distribution of the data.

# Set the style of Seaborn for better visualisation.
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))
sns.histplot(data=df,
             x='ConversionRate',
             hue='Design',
             element='poly',
             stat='density',
             common_norm=False)

plt.title('Distribution of the conversion rate per design')
plt.show()

"""Formulate the hypotheses as follows:
- $H_0$: $\mu_a=\mu_b=\mu_c$.
- $H_1$: At **least one** $\mu$ will differ.

"""

# Import required library.
import scipy.stats as stats

# Perform ANOVA.
f_statistic, p_value = stats.f_oneway(
    df['ConversionRate'][df['Design'] == 'A'],
    df['ConversionRate'][df['Design'] == 'B'],
    df['ConversionRate'][df['Design'] == 'C'])

# View the output.
print("F-statistic:", f_statistic)
print("p-value:", p_value)

"""## c) Hypothesis testing with chi-square
Follow the demonstration to learn the importance of **hypothesis testing with chi-square**. In this video, you will learn how to:
- create and explore the data set
- formulate the hypotheses
- conduct a chi-square test using Python.
"""

# Import the required libraries.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Create a hypothetical data set.
data = {'Design': ['A', 'B', 'C',
                   'A', 'B', 'C',
                   'A', 'B', 'C'],
        'AgeGroup': ['18-25', '18-25', '18-25',
                     '26-35', '26-35', '26-35',
                     '36-45', '36-45', '36-45'],
        'Count': [30, 14, 6,
                  29, 22, 9,
                  12, 18, 20]}

# Convert data into a DataFrame.
df_cs = pd.DataFrame(data)

# View the output.
print(df_cs.shape)
df_cs

# Exploratory data analysis.
# Create a barplot to visualise the data.

# Set the style of Seaborn for better visualisation.
sns.set(style="whitegrid")

# Create a barplot.
plt.figure(figsize=(10, 6))
sns.barplot(x='Design',
            y='Count',
            hue='AgeGroup',
            data=df_cs,
            palette='viridis')

plt.title('Count by design and age group')
plt.xlabel('Design type')
plt.ylabel('Count per age group')
plt.show()

"""Formulate the hypotheses as follows:
- $H_0$: Design and age group **are** independent.
- $H_1$: Design and age group **are not** independent.
"""

# Change the DataFrame format.
df_cs_pivot = df_cs.pivot(index='AgeGroup',
                          columns='Design',
                          values='Count')

# View the output.
print(df_cs_pivot.shape)
df_cs_pivot

# Import the required library.
from scipy.stats import chi2_contingency

# Perform chi-square test.
chi2_stat, p_value, dof, ex = chi2_contingency(df_cs_pivot)


# View the output.
print("Chi-square statistic:", chi2_stat)
print("p-value:", p_value)

"""# Key information
These demonstrations illustrated the importance of hypothesis testing with a $t$-test, ANOVA, and chi-square. The type of test you use to perform hypothesis testing will depend on the business scenario and the data set.

## Reflect
What are the practical applications of this technique?

> Select the pen from the toolbar to add your entry.
"""
