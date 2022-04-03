#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[6]:


df = pd.read_csv("../data/raw/athlete_events.csv")


# In[7]:


df.head(10)


# In[8]:


df = df.drop(["ID", "Name", "Team", "Games", "City", "Event"], axis=1)


# In[9]:


df = df[df["Year"] > 1959]


# In[10]:


df['Medal'] = df['Medal'].replace(np.nan, 0)
df['Medal'].unique()
print(f'shape is {df.shape}')


# In[11]:


df = df.dropna()
print(f'shape is {df.shape}')


# ### Selecting sport categories

# ### Converting medal names into medal (1) or no medal (0)

# In[12]:


df = df.replace(["Gold", "Silver", "Bronze"], 1)

df.Medal.value_counts()


# In[13]:


# check most represented sports
df.value_counts('Sport')


# ## Winter Olympics datasets:

# In[14]:


# set Event of interst

sport_list = ["Athletics", "Swimming", "Gymnastics"]
# rename_mapper = {
#     "Figure Skating Men's Singles" : "figure_men", 
#     "Figure Skating Women's Singles"  : "figure_wom", 
#     "Speed Skating Men's 1,500 metres" : "speed_man", 
#     "Speed Skating Women's 1,500 metres" : "speed_wom"
# }


# In[15]:


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


# In[16]:


# men_skating = tmp[tmp.Event =="Figure Skating Men's Singles"]
# print(f'men_skating shape is\t{men_skating.shape}')
# women_skating = tmp[tmp.Event =="Figure Skating Women's Singles"]
# print(f'women_skating shape is\t{women_skating.shape}')
# men_speed_skating = tmp[tmp.Event =="Speed Skating Men's 1,500 metres"]
# print(f'men_speed_skating shape is\t{men_speed_skating.shape}')
# women_speed_skating = tmp[tmp.Event =="Speed Skating Women's 1,500 metres"]
# print(f'women_speed_skating shape is\t{women_speed_skating.shape}')


# ### AutoML

# In[17]:


!pip install auto-sklearn


# In[18]:


import autosklearn.classification
import autosklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder


# In[ ]:


# setting parameters for autosklearn
# generate a function to create an automl
# object
def make_automl(name_id):
  automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=300, # limiting possible model combinations and selections to 300s == 5 mins
        per_run_time_limit=30,
        memory_limit=None,
        seed=1,
        resampling_strategy='holdout',
        resampling_strategy_arguments={
            'train_size': 0.8,
            'shuffle': True
        },
        metric=autosklearn.metrics.f1_weighted,
        scoring_functions=[
                           autosklearn.metrics.f1_weighted, 
                           autosklearn.metrics.balanced_accuracy,
                          autosklearn.metrics.precision_weighted, 
                           autosklearn.metrics.recall_weighted],
        tmp_folder=name_id
    )
  return(automl)

# get statistics
def get_metric_result(cv_results):
    results = pd.DataFrame.from_dict(cv_results)
    results = results[results['status'] == "Success"]
    cols = ['rank_test_scores', 'param_classifier:__choice__', 'mean_test_score']
    cols.extend([key for key in cv_results.keys() if key.startswith('metric_')])
    return results[cols]


# In[ ]:


wdf_dict = {}
# drop cols for X
X = tmp.drop(['Medal', 'Sport', 'NOC', 'Height', 'Weight', 'index', "Sex", "Season"], axis=1).copy()
print(f'X shape is {X.shape}\tcols are { ";".join(X.columns)}')
# create labels
y = tmp['Medal']
print(f'y shape is {y.shape}')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
print(f'X_train shape is {X_train.shape}')
print(f'y_train shape is {y_train.shape}')
print(f'X_test shape is {X_test.shape}')
print(f'y_test shape is {y_test.shape}')
# create automl object
automl_event = make_automl(f'final_sport_model_new')
# train model
automl_event.fit(X_train, y_train)
# stats
print(automl_event.sprint_statistics())
# get stats
metrics = get_metric_result(
        automl_event.cv_results_
        )
# predict to test set
predictions_sk = automl_event.predict(X_test)
# save prediction to wdf_dict
wdf_dict['predictions_sk'] = predictions_sk
# save y_test to wdf_dict
wdf_dict['y_test'] = y_test
# save train metrics
wdf_dict['metrics'] = metrics
# save model
wdf_dict['model'] = automl_event


# In[ ]:


wdf_dict['model'].show_models()


# In[ ]:


wdf_dict['metrics'].sort_values("metric_f1_weighted", ascending=False)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
import seaborn as sns

classification_report(wdf_dict['y_test'], wdf_dict['predictions_sk'], output_dict=True)
cm = confusion_matrix(wdf_dict['y_test'], wdf_dict['predictions_sk'])
#tn, fp, fn, tp = confusion_matrix(wdf_dict['y_test'], wdf_dict['predictions_sk']).ravel()
#print(tn, fp, fn, tp )
print(accuracy_score(wdf_dict['y_test'], wdf_dict['predictions_sk']))
print(precision_score(wdf_dict['y_test'], wdf_dict['predictions_sk']))

sns.heatmap(cm, annot=True, fmt='d')


# ## Random forest feature importance:

# ### 1. graph from the presentation based on age, year, country, sport, sex (sex_m, sex_f), season, and bmi: 

# Note: sex and season are categorical variables, therefore they were one hot encoded into new columns (Sex_F,	Sex_M, and	Season_Summer). For this reason, we need to drop the original categorical sex and season columns.

# In[ ]:


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


# In[ ]:


X.head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf_final = RandomForestClassifier(max_depth=200, random_state=42)
clf_final.fit(X_train, y_train)


# In[ ]:


predictions_all_countr = clf_final.predict(X_test)


# In[ ]:


cm_final = confusion_matrix(y_test, predictions_all_countr)
#tn, fp, fn, tp = confusion_matrix(wdf_dict['y_test'], wdf_dict['predictions_sk']).ravel()
#print(tn, fp, fn, tp )
print(accuracy_score(y_test, predictions_all_countr))
print(precision_score(y_test, predictions_all_countr))

sns.heatmap(cm_final, annot=True, fmt='d')


# In[ ]:


from sklearn.metrics import f1_score
f1_score(y_test, predictions_all_countr, average="macro")


# In[ ]:


labels = ["True_Neg","False_Pos","False_Neg","True_Pos"]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm_final, annot=labels, fmt='', cmap='Blues')


# BMI is the most important feature:

# In[ ]:


feat_importances = pd.Series(clf_final.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')


# ### 2. graph from the presentation based on age, bmi, and sports: 

# Removing features results in lower performace and lower success in discovering medal winners:

# In[ ]:


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


# In[ ]:


X.head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf_final = RandomForestClassifier(max_depth=200, random_state=42)
clf_final.fit(X_train, y_train)


# In[ ]:


predictions_all_countr = clf_final.predict(X_test)


# In[ ]:


cm_final = confusion_matrix(y_test, predictions_all_countr)
#tn, fp, fn, tp = confusion_matrix(wdf_dict['y_test'], wdf_dict['predictions_sk']).ravel()
#print(tn, fp, fn, tp )
print(accuracy_score(y_test, predictions_all_countr))
print(precision_score(y_test, predictions_all_countr))

sns.heatmap(cm_final, annot=True, fmt='d')


# In[ ]:


from sklearn.metrics import f1_score
f1_score(y_test, predictions_all_countr, average="macro")


# In[ ]:


labels = ["True_Neg","False_Pos","False_Neg","True_Pos"]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm_final, annot=labels, fmt='', cmap='Blues')


# We can see that sport category has very low feature importance:

# In[ ]:


feat_importances = pd.Series(clf_final.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')

