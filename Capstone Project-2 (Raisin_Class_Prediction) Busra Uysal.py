#!/usr/bin/env python
# coding: utf-8

# # Raisin Class Prediction

# Data Set Information:
# 
# Images of Kecimen and Besni raisin varieties grown in Turkey were obtained with CVS. A total of 900 raisin grains were used, including 450 pieces from both varieties. These images were subjected to various stages of pre-processing and 7 morphological features were extracted. These features have been classified using three different artificial intelligence techniques.
# 
# 
# Attribute Information:
# 
# 1. Area: Gives the number of pixels within the boundaries of the raisin.
# 2. Perimeter: It measures the environment by calculating the distance between the boundaries of the raisin and the pixels around it.
# 3. MajorAxisLength: Gives the length of the main axis, which is the longest line that can be drawn on the raisin.
# 4. MinorAxisLength: Gives the length of the small axis, which is the shortest line that can be drawn on the raisin.
# 5. Eccentricity: It gives a measure of the eccentricity of the ellipse, which has the same moments as raisins.
# 6. ConvexArea: Gives the number of pixels of the smallest convex shell of the region formed by the raisin.
# 7. Extent: Gives the ratio of the region formed by the raisin to the total pixels in the bounding box.
# 8. Class: Kecimen and Besni raisin.

# # Import libraries

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%matplotlib inline
#%matplotlib notebook
plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#pd.options.display.float_format = '{:.3f}'.format


# ## Exploratory Data Analysis and Visualization

# In[4]:


df = pd.read_excel('Raisin_Dataset.xlsx')


# In[5]:


df


# In[6]:


df.head()


# In[7]:


df.columns


# In[8]:


df.shape


# In[9]:


df.info()


# In[10]:


df["Class"] = df["Class"].map({"Kecimen": 0, "Besni": 1})


# In[11]:


df.head()


# In[12]:


df.Class.value_counts()


# In[13]:


df.describe().T


# In[14]:


df = df.copy()


# In[15]:


index = 0
plt.figure(figsize=(20,20))
for feature in df.columns:
    if feature != "Class":
        index += 1
        plt.subplot(3,3,index)
        sns.boxplot(x='Class',y=feature,data=df)
plt.show()


# In[16]:


df.corr()


# In[17]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[18]:


sns.pairplot(df, hue = "Class");


# In[ ]:





# In[ ]:





# In[ ]:





# ## Train | Test Split and Scaling

# In[19]:


X = df.drop(["Class"], axis = 1)
y = df["Class"]


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)


# # Scalling

# In[22]:


from sklearn.preprocessing import StandardScaler


# In[23]:


scaler = StandardScaler()


# In[24]:


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# # Logistic Regression

# In[25]:


from sklearn.linear_model import LogisticRegression


# In[26]:


log_model=LogisticRegression()


# In[27]:


log_model.fit(X_train_scaled, y_train)


# In[28]:


y_pred = log_model.predict(X_test_scaled)
y_pred


# In[29]:


y_pred_proba = log_model.predict_proba(X_test_scaled)
y_pred_proba


# In[30]:


test_data = pd.concat([X_test, y_test], axis=1)
test_data["pred"] = y_pred
test_data["pred_proba"] = y_pred_proba[:,1]#1 olma olasılığı
test_data.head(10)


# # Model Performance

# In[47]:


from sklearn.metrics import confusion_matrix, classification_report


# In[48]:


def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))


# In[49]:


eval_metric(log_model, X_train_scaled, y_train, X_test_scaled, y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# ### Cross Validate

# In[50]:


from sklearn.model_selection import cross_validate


# In[51]:


model = LogisticRegression()

scores = cross_validate(model, X_train_scaled, y_train, scoring = ['precision','recall','f1','accuracy'], cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores


# In[52]:


df_scores.mean()[2:]


# ### GridSearchCV

# In[53]:


from sklearn.model_selection import GridSearchCV


# In[54]:


model = LogisticRegression(random_state=42, max_iter=100000)

penalty = ["l1", "l2"]
C = np.logspace(-1, 5, 20)
class_weight=['balanced', None]
solver = ['lbfgs', 'liblinear', 'sag', 'saga']

param_grid = {"penalty" : penalty,
             "C" : C,
             "class_weight": class_weight,
             "solver" : solver}

grid_model = GridSearchCV(estimator= model,
                          param_grid= param_grid,
                         cv = 10,
                         scoring = "recall",
                         n_jobs = -1) 


# In[55]:


grid_model.fit(X_train_scaled,y_train)


# In[56]:


grid_model.best_params_


# In[57]:


grid_model.best_score_


# In[58]:


eval_metric(grid_model, X_train_scaled, y_train, X_test_scaled, y_test)


# ## ROC (Receiver Operating Curve) and AUC (Area Under Curve)

# In[59]:


from sklearn.metrics import roc_auc_score, auc, roc_curve


# In[60]:


roc_auc_score(y_test, y_pred_proba[:,1])


# In[61]:


from yellowbrick.classifier import ROCAUC


# In[65]:


import matplotlib.pyplot as plt
from yellowbrick.classifier import ROCAUC, PrecisionRecallCurve


roc_visualizer = ROCAUC(model, micro=False, macro=False, per_class=False)
roc_visualizer.fit(X_train_scaled, y_train)
roc_visualizer.score(X_test_scaled, y_test)
roc_visualizer.show()


prc_visualizer = PrecisionRecallCurve(model, micro=False, per_class=False)
prc_visualizer.fit(X_train_scaled, y_train)
prc_visualizer.score(X_test_scaled, y_test)
prc_visualizer.show()


# In[ ]:





# In[ ]:





# In[ ]:





# # Conclusion 

# In[90]:


scaler = StandardScaler().fit(X)


# In[91]:


import pickle
pickle.dump(scaler, open("scaler_raisin", 'wb'))


# In[93]:


X_scaled = scaler.transform(X)


# In[94]:


final_model = LogisticRegression().fit(X_scaled, y)


# In[95]:


pickle.dump(final_model, open("final_model_raisin", "wb"))


# In[96]:


X.describe().T


# In[97]:


my_dict = {
    "Area": [110804.128, 50804.128, 80804.128],
    "MajorAxisLength": [400.930, 200.930, 500.930],
    "MinorAxisLength": [200.488, 280.488, 225.488],
    "Eccentricity": [0.482, 0.682, 0.882],
    "ConvexArea": [80186.090, 100186.090, 12086.090],
    "Extent": [0.482, 0.682, 0.882], 
    "Perimeter": [1065.907, 1865.907, 2265.907], 
}

sample = pd.DataFrame(my_dict)
print(sample)


# In[98]:


sample = pd.DataFrame(my_dict)
sample


# In[101]:


scaler_raisin = pickle.load(open("scaler_raisin", "rb"))


# In[102]:


sample_scaled = scaler_raisin.transform(sample)
sample_scaled


# In[103]:


final_model = pickle.load(open("final_model_raisin", "rb"))


# In[115]:


predictions = final_model.predict(sample_scaled)
predictions_proba = final_model.predict_proba(sample_scaled)

print(predictions_proba)


# In[118]:


sample["pred"] = predictions
sample["pred_proba_keçimen"] = predictions_proba[:, 0]
sample["pred_proba_besni"] = predictions_proba[:, 1]
sample


# In[ ]:




