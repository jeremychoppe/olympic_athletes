#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("../data/raw/athlete_events.csv")


# In[3]:


df.head(10)


# In[4]:


df = df.drop(["ID", "Name", "Team", "Games", "City", "Event"], axis=1)


# In[5]:


df = df[df["Year"] > 1959]


# In[6]:


df['Medal'] = df['Medal'].replace(np.nan, 0)
df['Medal'].unique()
print(f'shape is {df.shape}')


# In[7]:


df = df.dropna()
print(f'shape is {df.shape}')


# ### Selecting sport categories

# ### Converting medal names into medal (1) or no medal (0)

# In[8]:


df = df.replace(["Gold", "Silver", "Bronze"], 1)

df.Medal.value_counts()


# In[9]:


# check most represented sports
df.value_counts('Sport')


# ## Winter Olympics datasets:

# In[10]:


# set Event of interst

sport_list = ["Athletics", "Swimming", "Gymnastics"]
# rename_mapper = {
#     "Figure Skating Men's Singles" : "figure_men", 
#     "Figure Skating Women's Singles"  : "figure_wom", 
#     "Speed Skating Men's 1,500 metres" : "speed_man", 
#     "Speed Skating Women's 1,500 metres" : "speed_wom"
# }


# In[11]:


setattr# filter main dataset and retrieve only categories of interest
wd_ag = df[df.Sport.isin(sport_list)].copy().reset_index()
# # rename wd_ag events
# wd_ag.replace(rename_mapper, inplace=True)
# create one hot encoding for NOC adn Event cols
noc_one = pd.get_dummies(wd_ag[["NOC", "Sport", "Sex", "Season"]])
#noc_one = pd.get_dummies(wd_ag[["Event"]])
# concat wd_ag and tmp 
tmp = pd.concat([wd_ag, noc_one], axis=1)
# calc BMI
tmp["BMI"] = tmp["Weight"] / ((tmp["Height"] / 100)** 2)
# print examples
tmp.sample(n=10)


# In[12]:


# men_skating = tmp[tmp.Event =="Figure Skating Men's Singles"]
# print(f'men_skating shape is\t{men_skating.shape}')
# women_skating = tmp[tmp.Event =="Figure Skating Women's Singles"]
# print(f'women_skating shape is\t{women_skating.shape}')
# men_speed_skating = tmp[tmp.Event =="Speed Skating Men's 1,500 metres"]
# print(f'men_speed_skating shape is\t{men_speed_skating.shape}')
# women_speed_skating = tmp[tmp.Event =="Speed Skating Women's 1,500 metres"]
# print(f'women_speed_skating shape is\t{women_speed_skating.shape}')


# ### AutoML

# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder


# ## Random forest feature importance:

# ### 1. graph from the presentation based on age, year, country, sport, sex (sex_m, sex_f), season, and bmi: 

# Note: sex and season are categorical variables, therefore they were one hot encoded into new columns (Sex_F,	Sex_M, and	Season_Summer). For this reason, we need to drop the original categorical sex and season columns.

# In[14]:


X = tmp.drop(["Medal", "Sport", "NOC", "Height", "Weight", "index", "Sex", "Season"], axis=1)
print(f'X shape is {X.shape}\tcols are { ";".join(X.columns)}')
# create labels
y = tmp['Medal']
print(f'y shape is {y.shape}')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
print(f'X_train shape is {X_train.shape}')
print(f'y_train shape is {y_train.shape}')
print(f'X_test shape is {X_test.shape}')
print(f'y_test shape is {y_test.shape}')


# In[15]:


X.head()


# In[16]:


from sklearn.ensemble import RandomForestClassifier
clf_final = RandomForestClassifier(max_depth=200, random_state=42)
clf_final.fit(X_train, y_train)


# In[17]:


predictions_all_countr = clf_final.predict(X_test)


# In[21]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
cm_final = confusion_matrix(y_test, predictions_all_countr)
#tn, fp, fn, tp = confusion_matrix(wdf_dict['y_test'], wdf_dict['predictions_sk']).ravel()
#print(tn, fp, fn, tp )
print(accuracy_score(y_test, predictions_all_countr))
print(precision_score(y_test, predictions_all_countr))
import seaborn as sns
sns.heatmap(cm_final, annot=True, fmt='d')


# In[22]:


from sklearn.metrics import f1_score
f1_score(y_test, predictions_all_countr, average="macro")


# BMI is the most important feature:

# In[24]:


feat_importances = pd.Series(clf_final.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')


# ### 2. graph from the presentation based on age, bmi, and sports: 

# Removing features results in lower performace and lower success in discovering medal winners:

# In[25]:


X = tmp[["Age","BMI","Sport_Athletics",	"Sport_Gymnastics",	"Sport_Swimming"]]
print(f'X shape is {X.shape}\tcols are { ";".join(X.columns)}')
# create labels
y = tmp['Medal']
print(f'y shape is {y.shape}')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
print(f'X_train shape is {X_train.shape}')
print(f'y_train shape is {y_train.shape}')
print(f'X_test shape is {X_test.shape}')
print(f'y_test shape is {y_test.shape}')


# In[26]:


X.head()


# In[27]:


from sklearn.ensemble import RandomForestClassifier
clf_final = RandomForestClassifier(max_depth=200, random_state=42)
clf_final.fit(X_train, y_train)


# In[28]:


predictions_all_countr = clf_final.predict(X_test)


# In[29]:


cm_final = confusion_matrix(y_test, predictions_all_countr)
#tn, fp, fn, tp = confusion_matrix(wdf_dict['y_test'], wdf_dict['predictions_sk']).ravel()
#print(tn, fp, fn, tp )
print(accuracy_score(y_test, predictions_all_countr))
print(precision_score(y_test, predictions_all_countr))

sns.heatmap(cm_final, annot=True, fmt='d')


# In[30]:


from sklearn.metrics import f1_score
f1_score(y_test, predictions_all_countr, average="macro")


# We can see that sport category has very low feature importance:

# In[32]:


feat_importances = pd.Series(clf_final.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')

