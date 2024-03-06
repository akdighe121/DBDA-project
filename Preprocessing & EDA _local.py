#!/usr/bin/env python
# coding: utf-8

# # Project Title:  Predicting App Success On Google Playstore

# # 1. Introduction to the Dataset:
# 

# # 1.1. Welcome to the world of mobile apps! The Google Play Store, like a giant market, is home to a ton of apps that cover all sorts of interests. As a junior analyst, I set out to explore and understand the stories behind these apps. Let's dive into the data and uncover the exciting insights it holds!

# # 1.3. Importing the dependecies

# In[25]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
get_ipython().system('pip install CurrencyConverter')
from currency_converter import CurrencyConverter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor
from keras.layers import Dense,BatchNormalization,Flatten
from keras.models import Sequential
import keras
from keras.metrics import mean_absolute_error
warnings.filterwarnings('ignore')


# # 2. Data Preparation and Cleaning
- load the csv file with pandas
- creating the dataframe and understanding the data present in the dataset
- dealing with the missing data and the incorrect records
# In[ ]:


df=pd.read_csv("Google-Playstore.csv")


# In[2]:


#Print first 10 records
df.head(10)


# In[3]:


df.info()


# In[4]:


df.columns

## Issues List For the Dataset
# - Missing values in several cols: Rating, rating count, Installs, minimum and maximum installs, currency and more
# - Convert all columns to snake_case
# - Drop these columns: App ID, minimum and maximum installs, minimum android version, developer ID, website and email, privacy policy link.
# - Incorrect data types for release data and size
# - Music and education is represented by different labels
# - Drop unnecessary categories
# In[5]:


#Display the data types of each column in the dataframe
df.dtypes


# In[6]:


#Let's print first row and compare it against the header or another row that is correct.

df.iloc[0]


# In[7]:


#Return the count of distincted values

df.nunique()


# In[8]:


df.describe()


# In[9]:


#Descriptive analysis(summary statistics)

df_description = df.describe().applymap(lambda x: format(x, '.6f'))
df_description_str = df_description.to_string()
print(df_description_str)


# In[10]:


df.shape


# # Observations
# 
# 1. There are '2312944' rows and '24' columns in the dataset
# 2. The columns are of different data types
# 3. The columns in the datasets are:
#     - 'App Name', 'App Id', 'Category', 'Rating', 'Rating Count', 'Installs',
#        'Minimum Installs', 'Maximum Installs', 'Free', 'Price', 'Currency',
#        'Size, 'Minimum Android', `Developer Id`, `Developer Website`,
#        `Developer Email`, `Released`, `Last Updated`, `Content Rating`,
#        `Privacy Policy`, `Ad Supported`, `In App Purchases`, `Editors Choice`,
#        `Scraped Time`
# 4. dtypes: bool = 4, float64 = 4, int64 = 1, object = 15
# 5. There are some missing values in the dataset which we will read in details and deal later on in the notebook.  
#     - `Developer Website` have almost 33% and `Privacy Policy` have almost 18% of missing values 
# 6. Few columns need remove as they do not contribute to the overall results of the dataset
#     - like: `App Id`, `Developer Id`, `Developer Website`, `Developer Email`, `Privacy Policy`, `Editors Choice`, `Minimum Android` & `Scraped Time`
# 7. There are some columns which are of object data type but they should be of numeric data type, we will convert them later on in the notebook.
#     - `Installs` & `Size`

# # 2.1 Drop the columns that are not important

# In[32]:


# Drop unnecessary columns from the DataFrame
df = df.drop(['App Id', 'Developer Id', 'Developer Website', 'Developer Email', 'Privacy Policy', 'Editors Choice', 'Minimum Android', 'Scraped Time'], axis='columns')


# In[33]:


# Rename columns by converting them to lowercase, removing leading/trailing whitespaces, and replacing spaces with underscores
df.rename(lambda x: x.lower().strip().replace(' ', '_'), axis='columns', inplace=True)


# In[34]:


# Display the first five rows of the DataFrame
df.head()


# # 2.2 Tackle Null/Missing values

# In[35]:


df.isnull().sum()


# In[36]:


# Select and display only the columns where the count of missing values is greater than 0
df.isnull().sum()[df.isnull().sum()>0]

- Before going ahead, let's remove the rows with missing values in the `app_name`, `installs`, `minimum_installs`, and `currency` columns, as they are very less in number and will not affect our analysis.
# In[42]:


# Drop rows with missing values in specified columns ('app_name', 'installs', 'minimum_installs', 'currency')
# The inplace=True parameter modifies the DataFrame in place
df.dropna(subset=['app_name', 'installs', 'minimum_installs','size' ,'currency'], inplace=True)


# In[43]:


# Select and display only the columns where the count of missing values is greater than 0
df.isnull().sum()[df.isnull().sum()>0]


# In[45]:


df.duplicated().sum()


# In[46]:


# Drop duplicate rows in the DataFrame, keeping only the first occurrence of each set of duplicated rows
df.drop_duplicates(inplace=True)


# In[47]:


df.duplicated().sum()


# In[50]:


df['rating'].describe()


# In[51]:


# Calculate the average rating in each 'Category'
df.groupby('category')['rating'].mean()


# In[52]:


# Calculate the average rating in each 'Category'
average_ratings = df.groupby('category')['rating'].transform('mean')

# Replace missing values in 'Rating' with the respective average ratings of their Category
df['rating'] = df['rating'].fillna(average_ratings)

- For `rating_count` we will replace the respective missing values of rating_count, with respect of the average rating count of their 'Category'
# In[53]:


# Calculate the average rating in each 'Category'
average_rating_count = df.groupby('category')['rating_count'].transform('mean')

# Replace missing values in 'Rating' with the respective average ratings of their Category
df['rating_count'] = df['rating_count'].fillna(average_rating_count)


# In[58]:


df = df.dropna(subset=['released'])


# In[60]:


df['released']


# In[62]:


df['installs'].unique()


# In[66]:


df['installs'] = df['installs'].str.split('+').str[0]  # Remove the + symbol
df['installs'].replace(',','', regex=True, inplace=True)   # Remove the + symbol


# In[67]:


df['installs'] = df['installs'].replace (to_replace=[np.nan,""], value=0).astype('int64')   # Converting it to the int type


# In[68]:


df['installs'].unique()


# In[70]:


df['currency'].unique()


# In[72]:


df['size'].unique()


# #  3. The Size of data can be in GB,MB and KB we will convert the data into the size in MB

# In[73]:


df['size'] = df['size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)


# In[102]:


#df['size'] = df['size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)

- Here we get mismatched value with the data we got the value 1,018 we can drop it or we can assume as it may be a '.'(dot) that would incorrectly added to the dataset. so let assume it as dot for now and replace the ',' with dot '.'.
# In[76]:


df['size'] = df['size'].apply(lambda x: str(x).replace(',', '.') if ',' in str(x) else x)


# # 3.1 Convert kbs to mb

# In[77]:


df['size'] = df['size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)


# # 3.2 Convert GBs to MBs

# In[78]:


df['size'] = df['size'].apply(lambda x: float(str(x).replace('G', '')) * 1000 if 'G' in str(x) else x)


# In[79]:


df.dtypes['size']


# In[84]:


df['last_updated']


# In[85]:


df['released']


# In[87]:


df['free']


# In[89]:


# lets clean the Content rating column
df['content_rating'].unique()


# In[90]:


df['content_rating'].value_counts()


# # Observations :
# 
# 1. We have varies Categories in the Content Rating Columns:
#    - Everyone
#    - Teen
#    - Mature 17+
#    - Everyone
#    - Unrated
#    - Adults only 18+
# 
# Now, we makes this Categories to a simple 3 Categories for better Understanding:
# Everyone, teen, Adults
#   - Mature 17+ ---> to Adults
#   - Everyone 10+ ---> to TEEN
#   - Unrated ---> to Everyone
#   - Adults only 18+ ---> to Adults
#   

# In[91]:


df["content_rating"] = df["content_rating"].replace("Unrated", "Everyone")
df["content_rating"] = df["content_rating"].replace("Everyone 10+", "Teen")
df["content_rating"] = df["content_rating"].replace("Mature 17+", "Adults")
df["content_rating"] = df["content_rating"].replace("Adults only 18+", "Adults")


# In[92]:


df['content_rating'].unique()


# In[93]:


# Creataing the column type for free and paid Apps by using the Free column, it's helpfull while dealing with the paid and Free Apps
df['type'] = np.where(df['free'] == True,'Free','Paid')
df.drop(['free'],axis=1, inplace= True )


# In[94]:


df.info()


# In[95]:


df['rating'].unique()


# In[96]:


df['rating_count'].max()


# In[99]:


df['rating_type'] = 'NoRatingProvided'
df.loc[(df['rating_count'] > 0) & (df['rating_count'] <=10000.0), 'rating_type'] = 'Less than 10k'
df.loc[(df['rating_count'] > 10000) & (df['rating_count'] <=500000.0), 'rating_type'] = 'Between 10k and 500k'
df.loc[(df['rating_count'] > 500000) & (df['rating_count'] <=138557570.0), 'rating_type'] = 'More than 500k'
df['rating_type'].value_counts()


# In[100]:


df['rating_type']


# In[101]:


df.info()


# In[111]:


df.head()


# In[110]:


df.drop(columns=['Rating type'], inplace=True)


# In[112]:


df.info()


# # 4. Calculate the correlation coefficients between numerical columns

# In[113]:


numeric_columns = df.select_dtypes(include='number')
correlation = numeric_columns.corr()
correlation


# # 4.1 Represent the relationships between different numerical variables using heatmap.

# In[116]:


plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f",linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# # 4.2 Boxplot to understand the distribution and variability of numerical data.

# In[117]:


# filter out only the numerical columns.
numerical_columns = df.select_dtypes(include='number').columns
# Create box plots for numerical columns
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numerical_columns])
plt.title('Box Plots for Numerical Columns')
plt.show()


# # 5. Question and Answers:
#  - What are the top 10 Categories that are installed from the Google Play Store?
#  - Which are the Categories that are getting installed the most in the top Categories?
#  - which is the highest rated categories?
#  - How does the size of the application impacts the installation?
#  - What are the top 5 paid Apps based with highest ratings and installs?

# In[119]:


df['category'].unique()


# In[120]:


top_category = df.category.value_counts().reset_index().rename(columns={'category':'Count','index':'category'})


# In[121]:


top_category


# In[122]:


category_installs = df.groupby(['category'])[['installs']].sum()


# In[123]:


category_installs


# In[124]:


top_category_installs = pd.merge(top_category, category_installs, on='category')
top_category_installs.head(10)


# In[125]:


top_10_categories_installs = top_category_installs.head(10).sort_values(by = ['installs'], ascending= False)


# In[126]:


plt.figure(figsize=(14,7))
plt.xticks(rotation=60)
plt.xlabel("category")
plt.ylabel("Number of applications")
plt.title("Top 10 installed categories")
sns.barplot(x= top_10_categories_installs.category, y = top_10_categories_installs.installs)


# # What are the 10 Categories in playstore as per the count?

# In[127]:


plt.figure(figsize=(14,7))
plt.xticks(rotation=60)
plt.xlabel("category")
plt.ylabel("Number of applications")
plt.title("top 10 categories")
sns.barplot(x = top_10_categories_installs.category, y = top_10_categories_installs.Count)


# # Visulize total Categories and the Count of Apps in each Catgory

# In[128]:


plt.figure(figsize=(14,7))
plt.xticks(rotation=90)
plt.xlabel("category_installs")
plt.ylabel("Number of applications")
plt.title("Total categories and count of Applications in each category")
sns.barplot(x = top_category_installs.category, y =top_category_installs.Count)


# # Visualize Total Categories and installed Applications in each category

# In[129]:


plt.figure(figsize=(14,7))
plt.xticks(rotation=90)
plt.xlabel("category")
plt.ylabel("Number of applications")
plt.title("Total categories and Installation of Applications in each category")
sns.barplot(x = top_category_installs.category, y =top_category_installs.installs)


# # Rating Distribution 

# In[131]:


plt.figure(figsize=(14,7))
g = sns.kdeplot(df.rating, color="Blue", shade= True)
g.set_xlabel("rating")
g.set_ylabel("Frequency")
plt.title('Distribution of rating', size = 20)


# In[132]:


plt.title("rating")
sns.histplot(df.rating, kde=True,bins=5)


# # Observation
# From the above two plots we can see that most people does not give a rating, but one more thing that comes out from this graph as well is that people tend to give 4+ rating the most

# # Q. What is highest rated category?

# In[133]:


plt.figure(figsize=(14,7))
plt.xticks(rotation=90)
plt.xlabel("Higest Rated Category")
plt.ylabel("Number of applications")
plt.title("All Categories Rating ")
sns.barplot(x = df.category, y = df.rating)


# # Ans- From the above plot we can see that Role Playing is the Highest Rated Category

# In[135]:


df['content_rating'].unique()


# In[136]:


plt.figure(figsize=(8,8))
plt.title("Content Rating and Maximum installations ")
sns.scatterplot(x = 'maximum_installs', y ='rating_count',data=df, hue='content_rating')


# # Observation
# 
#  This scatter plot shows us that : if we exclude everyone from the plot and when focus on teen and Adults we can see that teens have much engagement in terms of downloads and rating Count.

# In[137]:


# Visulize the installation Types in each category

df['installs'].min(),df['installs'].max()


# there is high variance in the number of installs, we need to reduce it so we can use a log value for this column,otherwise it would be unable to see the data when we visulize

# In[138]:


category_type_installs = df.groupby(['category'])[['installs']].sum().reset_index()
category_type_installs['log_installs'] = np.log10(category_type_installs['installs'])


# In[139]:


plt.figure(figsize=(18,9))
plt.xticks(rotation=65, fontsize=9)
plt.xlabel("category")
plt.ylabel("installs")
plt.title("Number of installed Apps type wise according to Category")
sns.barplot(x= 'category', y = 'log_installs',  data=category_type_installs)


# # Ans - from the above plot we can see that size impacts the number of installations. Applications with lager size are less installed by the end user.

# In[140]:


df.corr()


# In[155]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cbar=True, cmap='coolwarm')
plt.show()


# # Q - what are the Top 5 Free Apps based with highest rating and Installs?

# In[144]:


free_apps = df[(df.type=='Free') & (df.installs >= 5000000)]
free_apps = free_apps.groupby('app_name')['rating'].max().sort_values(ascending= False)
free_apps = free_apps.head(5)


# In[145]:


plt.figure(figsize=(8,4))
plt.xlabel("rating")
sns.set_theme(style="whitegrid")
plt.title("Top 5 Free Rated Apps")
sns.lineplot(x= free_apps.values, y =free_apps.index,color='Blue')


# # Ans - Photo Frame, Video maker with photo & music, kuku FM - Love stories, Audio Books & Podcasts, Plank Workout a Home.

# # Now, Visulize the categories that have the top 10 Max Installations

# In[146]:


plt.figure(figsize=(8,6))
data = df.groupby('category')['maximum_installs'].max().sort_values(ascending = True)
data = data.head(10)
labels = data.keys()
plt.pie(data, labels= labels, autopct='%0f%%')
plt.title("Top 10 Max installations Category wise", fontsize=14)


# # Visulize the top 10 installation categories that adults have installed the most

# In[148]:


df['content_rating'].unique()


# In[149]:


plt.figure(figsize=(8,6))
Adult = df[(df['content_rating']=='Adults')]
Adult = Adult.groupby(['category'])['maximum_installs'].max().sort_values(ascending=False)
Adult = Adult.head(10)
labels = Adult.keys()
plt.pie(x = Adult, autopct="%.1f%%", labels=labels)
plt.title("Adults Installing apps in terms of category", fontsize=14)


# # Observation :
# 
# - Most of the Adults showing interest in download the social, Action and Communication Category

# # Visulize Teens Installing the apps in terms of Category

# In[151]:


plt.figure(figsize=(8,6))
Teen = df[(df['content_rating']=='Teen')]
Teen = Teen.groupby(['category'])['maximum_installs'].max().sort_values(ascending=False)
Teen = Teen.head(10)
labels = Teen.keys()
plt.pie(x= Teen, autopct="%.1f%%", labels=labels)
plt.title("Teen Installing apps in terms of category")


# # Summary and Conclusion:
# 
# - People are more interested to install the gamming Apps, the top Rating is given to the gaming apps.
# - InAppPurchase are correlated to App Rating. So we can say that if app provides customers support and have subscription plans it will helps to engage customers.
# - Most of the Adults installed the social and communication Apps.
# - Most of the installation are done by teen and the most are vidio players and editors. Video players and editors are most in demand.
# - Size of application varies the installation.
# - People aare mostly downloaded the free apps 
# - The installation of free apps is High
# - The availability of the free apps is very high.

# In[156]:


# Step 1: Sort the DataFrame 'df' by 'rating_count' in descending order
# Step 2: Select the top 10 rows from the sorted DataFrame
# Step 3: Select only the columns 'app_name', 'rating', and 'category' for the top-rated apps
top_rated_apps = df.sort_values(by='rating_count', ascending=False).head(10)[['app_name', 'rating', 'category']]

# Step 4: Print a message indicating that the following apps are the top 10 rated
print("Top 10 Rated Apps:")

# Step 5: Display the DataFrame containing the top-rated apps
top_rated_apps


# # Q. Which is the percentage of Paid and Free apps?

# In[160]:


# Step 1: Set the size of the figure
plt.figure(figsize=(8, 8))

# Step 2: Calculate the count of each category in the 'free' column
count = df['type'].value_counts()

# Step 3: Create a pie chart using the count, explode the 'Free' slice (to emphasize it), 
# set labels, autopct for percentage display, shadow for 3D effect, and startangle for rotation
plt.pie(count, explode=(0.25,0), labels=['free', 'Paid'], autopct='%1.1f%%', shadow=True, startangle=135)

# Step 4: Set the title of the pie chart
plt.title('Percent of Free Vs Paid Apps in store', size = 16)

# Step 5: Show the plot
plt.show()


# # Observation
# 
# - 2% of Apps are Paid
# - 98% of Apps are Free
# 

# In[ ]:





# In[ ]:




