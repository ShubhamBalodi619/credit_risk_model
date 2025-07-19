#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[684]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score


# # Getting the data

# In[686]:


credit_risk_data= pd.read_csv(r"C:\Users\balod\Quant Finance Projects\Credit Risk Modelling\credit_risk_dataset.csv")
credit_risk_data.head()


# # Exploring & cleaning the data

# In[688]:


credit_risk_data.shape


# In[689]:


credit_risk_data.describe()


# ## Visualizing the data

# In[691]:


credit_risk_data.hist(bins= 50, figsize= (20, 15))


# ## Dealing with the outliers

# In[693]:


credit_risk_data_copy= credit_risk_data.copy()


# In[694]:


credit_risk_data_copy.pivot_table(index="person_age", columns="loan_status", values="person_income", aggfunc="count").sort_values(by="person_age", ascending=0)


# In[695]:


credit_risk_data_cleaned= credit_risk_data_copy[credit_risk_data_copy["person_age"]<=80]
credit_risk_data_cleaned= credit_risk_data_cleaned.reset_index(drop=1)
# credit_risk_data_cleaned.head()


# In[696]:


credit_risk_data_cleaned.shape


# In[697]:


credit_risk_data_cleaned.describe()


# In[698]:


credit_risk_data_cleaned.pivot_table(index="person_emp_length", columns="loan_status", values="person_income", aggfunc="count").sort_values(by="person_emp_length", ascending=0)


# In[699]:


credit_risk_data_cleaned= credit_risk_data_cleaned[credit_risk_data_cleaned["person_emp_length"]<=40]
credit_risk_data_cleaned= credit_risk_data_cleaned.reset_index(drop=1)
# credit_risk_data_cleaned.head()


# In[700]:


credit_risk_data_cleaned.shape


# In[701]:


credit_risk_data_cleaned.describe()


# In[702]:


credit_risk_data_cleaned_copy= credit_risk_data_cleaned.copy()


# In[703]:


credit_risk_data_cleaned_copy.isna().sum()


# In[704]:


credit_risk_data_cleaned_copy= credit_risk_data_cleaned_copy.fillna({"loan_int_rate":credit_risk_data_cleaned_copy["loan_int_rate"].median()})


# In[705]:


credit_risk_data_cleaned_copy.isna().sum()


# In[706]:


credit_risk_data_cleaned_copy.describe()


# In[707]:


credit_risk_data_cleaned_copy.shape


# # Preparing the data

# ## Encoding the categorical variables

# In[710]:


credit_risk_data_cleaned_copy.groupby("person_home_ownership").count()["person_income"]


# In[711]:


credit_risk_data_cleaned_copy.groupby("loan_grade").count()["person_income"]


# In[712]:


credit_risk_data_cleaned_copy.groupby("cb_person_default_on_file").count()["person_income"]


# In[713]:


credit_risk_data_cleaned_copy.groupby("loan_intent").count()["person_income"]


# In[714]:


credit_risk_data_cleaned_copy["cb_person_default_on_file"]= np.where(credit_risk_data_cleaned_copy["cb_person_default_on_file"]=="N",0,1)
credit_risk_data_cleaned_copy.head()


# In[715]:


person_home_ownership_df= pd.get_dummies(credit_risk_data_cleaned_copy["person_home_ownership"]).astype(int)
loan_intent_df= pd.get_dummies(credit_risk_data_cleaned_copy["loan_intent"]).astype(int)
loan_grade_df= pd.get_dummies(credit_risk_data_cleaned_copy["loan_grade"]).astype(int)


# ## Scaling the numerical variables

# In[717]:


data_to_scale= credit_risk_data_cleaned_copy[['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']]
data_to_scale.head()


# In[718]:


scaler= StandardScaler()
scaled_data= scaler.fit_transform(data_to_scale)
scaled_data= pd.DataFrame(scaled_data, columns=["person_age",	"person_income",	"person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"])
scaled_data.head()


# In[719]:


credit_risk_data_scaled_encoded= pd.concat([scaled_data, person_home_ownership_df, loan_intent_df, loan_grade_df], axis=1)
credit_risk_data_scaled_encoded["cb_person_default_on_file"]= credit_risk_data_cleaned_copy["cb_person_default_on_file"]
credit_risk_data_scaled_encoded["loan_status"]= credit_risk_data_cleaned_copy["loan_status"]
credit_risk_data_scaled_encoded.head()


# ## Data balancing

# In[721]:


credit_risk_data_scaled_encoded.groupby("loan_status").count()["person_age"]


# In[722]:


features= credit_risk_data_scaled_encoded.drop(columns= ["loan_status"])
features.head()


# In[723]:


target= credit_risk_data_scaled_encoded["loan_status"]
target.head()


# In[724]:


smote= SMOTE()
balanced_features, balanced_target= smote.fit_resample(features, target)
balanced_features.shape


# In[725]:


balanced_target.shape


# In[726]:


balanced_features_df= pd.DataFrame(balanced_features)
balanced_features_df.head()


# In[727]:


balanced_target_df= pd.DataFrame(balanced_target)
balanced_target_df.head()


# In[728]:


balanced_target_df.groupby("loan_status").size()


# # Model Training- Logistic Model

# ## Splitting the data into training and testing sets

# In[731]:


x_train, x_test, y_train, y_test= train_test_split(balanced_features, balanced_target, test_size= 0.2, random_state= 42)


# In[732]:


logit= LogisticRegression()
logit.fit(x_train, y_train)
y_predict= logit.predict(x_test)


# In[733]:


training_accuracy= logit.score(x_train, y_train)
test_accuracy= logit.score(x_test, y_test)
print(f"Training accuracy: {round(training_accuracy, 3)}")
print(f"Test accuracy: {round(test_accuracy, 3)}")


# In[734]:


pred_probs= logit.predict_proba(x_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, pred_probs)

plt.plot([0,1], [0,1], '--')
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title('Logistic Regression ROC Curve')


# In[735]:


auc_score= roc_auc_score(y_test, pred_probs)
print(f"AUC score: {round(auc_score, 3)}")


# In[736]:


sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[737]:


print("Classification report:")
print(classification_report(y_test, y_predict))


# In[738]:


logit_feature_importance= pd.DataFrame({"features":balanced_features.columns, "feature_coefficient": logit.coef_[0]})
logit_feature_importance_sorted= logit_feature_importance.sort_values(by= "feature_coefficient", ascending= False)

logit_feature_importance_sorted.plot(kind='barh', x='features', y='feature_coefficient', legend=False, figsize=(10, 8), color=logit_feature_importance_sorted['feature_coefficient'].apply(lambda x: 'green' if x > 0 else 'red'))

plt.title('Logistic Regression Feature Importances')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.tight_layout()
plt.gca().invert_yaxis()  # Optional: largest on top
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


# In[739]:


logit_prediction_df= pd.DataFrame({"test_indices": x_test.index, "logit_prediction": y_predict})
logit_prediction_df.head()


# In[740]:


credit_risk_data_merged= credit_risk_data_cleaned.merge(logit_prediction_df, left_index= True, right_on= "test_indices", how="left")
credit_risk_data_merged.head()


# In[741]:


final_data_with_prediction= credit_risk_data_merged.dropna()
final_data_with_prediction= final_data_with_prediction.reset_index()
final_data_with_prediction= final_data_with_prediction.drop(columns=["index", "test_indices"])
final_data_with_prediction.shape


# In[742]:


final_data_with_prediction.head()


# In[743]:


# final_data_with_prediction.to_excel(r"C:\Users\balod\Quant Finance Projects\Credit Risk Modelling\pd_prediction.xlsx", index=False)


# In[ ]:





# In[ ]:





# In[ ]:




