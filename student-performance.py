#!/usr/bin/env python
# coding: utf-8

# # Student Performance Prediction

# In[1]:


#import library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics


# # Data Extraction 

# In[2]:


training1 = pd.read_csv('../input/student-performance-data-set-competition-form/X_train.csv')
training2 = pd.read_csv('../input/student-performance-data-set-competition-form/y_train.csv')


# In[3]:


training1.shape


# In[4]:


training2.shape


# In[5]:


training1.head()


# In[6]:


training2.head()


# In[7]:


#union of data
join = (training1, training2['G3'])
df = pd.concat(join, axis = True)
df.head()


# In[8]:


#indexing student ID
df = df.set_index('StudentID')
df.head()


# In[9]:


df.describe()


# In[10]:


df.info()


# In[11]:


#heatmap correlation
plt.figure(figsize = (8,6))
sns.heatmap(df.corr(), vmax = 0.9, cmap = 'viridis')
plt.title("Pearson Correlation")
plt.show()


# # Visualization

# In[12]:


#visualize school type
plt.figure(figsize = (8,6))
sns.countplot(df['school'])
plt.title("School Type of Student Performance")
plt.xlabel("School")
plt.show()


# In[13]:


#visualize school type ~ G3
plt.figure(figsize = (8,6))
sns.boxplot(data = df, x = 'school', y ='G3', palette = 'Set1')
plt.title("School Type of Student Performance ~ G3")
plt.xlabel("School")
plt.ylabel("G3")
plt.show()


# In[14]:


#visualize gender
plt.figure(figsize = (8,6))
sns.countplot(df['sex'], palette = 'Set2')
plt.title("Gender of Student Performance")
plt.xlabel("Gender")
plt.show()


# In[15]:


#visualize gender ~ G3
plt.figure(figsize = (8,6))
sns.boxplot(data = df, x = 'sex', y ='G3')
plt.title("Gender of Student Performance ~ G3")
plt.xlabel("Gender")
plt.ylabel("G3")
plt.show()


# In[16]:


#visualize address
plt.figure(figsize = (8,6))
sns.countplot(df['address'], palette = 'Set1')
plt.title("Address of Student Performance")
plt.xlabel("Address")
plt.show()


# In[17]:


#visualize address ~ G3
plt.figure(figsize = (8,6))
sns.boxplot(data = df, x = 'address', y ='G3', palette = 'Set2')
plt.title("Address of Student Performance ~ G3")
plt.xlabel("Address")
plt.ylabel("G3")
plt.show()


# In[18]:


#visualize familly size
plt.figure(figsize = (8,6))
sns.countplot(df['famsize'])
plt.title("Family Size of Student Performance")
plt.xlabel("Family Size")
plt.show()


# In[19]:


#visualize family size ~ G3
plt.figure(figsize = (8,6))
sns.boxplot(data = df, x = 'famsize', y ='G3', palette = 'Set1')
plt.title("Family Size of Student Performance ~ G3")
plt.xlabel("Family Size")
plt.ylabel("G3")
plt.show()


# In[20]:


#visualize performance status
plt.figure(figsize = (8,6))
sns.countplot(df['Pstatus'], palette = 'Set2')
plt.title("Performance Status of Student Performance")
plt.xlabel("Performance Status")
plt.show()


# In[21]:


#visualize performance status ~ G3
plt.figure(figsize = (8,6))
sns.boxplot(data = df, x = 'Pstatus', y ='G3')
plt.title("Performance Status of Student Performance ~ G3")
plt.xlabel("Performance Status")
plt.ylabel("G3")
plt.show()


# In[22]:


#visualize mother job
plt.figure(figsize = (8,6))
sns.countplot(df['Mjob'], palette = 'Set1')
plt.title("Mother Job of Student Performance")
plt.xlabel("Mother Job")
plt.show()


# In[23]:


#visualize mother job ~ G3
plt.figure(figsize = (8,6))
sns.boxplot(data = df, x = 'Mjob', y ='G3', palette = 'Set2')
plt.title("Mother Job of Student Performance ~ G3")
plt.xlabel("Mother Job")
plt.ylabel("G3")
plt.show()


# In[24]:


#visualize mother job
plt.figure(figsize = (8,6))
sns.countplot(df['Fjob'])
plt.title("Father Job of Student Performance")
plt.xlabel("Father Job")
plt.show()


# In[25]:


#visualize father job ~ G3
plt.figure(figsize = (8,6))
sns.boxplot(data = df, x = 'Fjob', y ='G3', palette = 'Set1')
plt.title("Father Job of Student Performance ~ G3")
plt.xlabel("Father Job")
plt.ylabel("G3")
plt.show()


# In[26]:


#visualize reason
plt.figure(figsize = (8,6))
sns.countplot(df['reason'], palette = 'Set2')
plt.title("Reason of Student Performance")
plt.xlabel("Reason")
plt.show()


# In[27]:


#visualize reason ~ G3
plt.figure(figsize = (8,6))
sns.boxplot(data = df, x = 'reason', y ='G3')
plt.title("Reason of Student Performance ~ G3")
plt.xlabel("Reason")
plt.ylabel("G3")
plt.show()


# In[28]:


#visualize guardian
plt.figure(figsize = (8,6))
sns.countplot(df['guardian'], palette = 'Set1')
plt.title("Guardian of Student Performance")
plt.xlabel("Guardian")
plt.show()


# In[29]:


#visualize guardian ~ G3
plt.figure(figsize = (8,6))
sns.boxplot(data = df, x = 'guardian', y ='G3', palette = 'Set2')
plt.title("Guardian of Student Performance ~ G3")
plt.xlabel("Guardian")
plt.ylabel("G3")
plt.show()


# In[30]:


#group gender & school
gender_school = df.groupby(['sex', 'school']).size().reset_index(name = 'Count')

#visualize gender ~ school
plt.figure(figsize = (8,6))
sns.barplot(data = gender_school, x = 'sex', y = 'Count', hue = 'school')
plt.title("Gender ~ School")
plt.show()


# In[31]:


#group gender & address
gender_address = df.groupby(['sex', 'address']).size().reset_index(name = 'Count')

#visualize gender ~ address
plt.figure(figsize = (8,6))
sns.barplot(data = gender_address, x = 'sex', y = 'Count', hue = 'address', palette = 'Set1')
plt.title("Gender ~ Address")
plt.show()


# In[32]:


#group gender & family size
gender_family = df.groupby(['sex', 'famsize']).size().reset_index(name = 'Count')

#visualize gender ~ family size
plt.figure(figsize = (8,6))
sns.barplot(data = gender_family, x = 'sex', y = 'Count', hue = 'famsize', palette = 'Set2')
plt.title("Gender ~ Family Size")
plt.show()


# In[33]:


#group gender & performance status
gender_performance = df.groupby(['sex', 'Pstatus']).size().reset_index(name = 'Count')

#visualize gender ~ performance status
plt.figure(figsize = (8,6))
sns.barplot(data = gender_performance, x = 'sex', y = 'Count', hue = 'Pstatus')
plt.title("Gender ~ Performance Status")
plt.show()


# In[34]:


#group gender & mother job
gender_mother = df.groupby(['sex', 'Mjob']).size().reset_index(name = 'Count')

#visualize gender ~ mother job
plt.figure(figsize = (8,6))
sns.barplot(data = gender_mother, x = 'sex', y = 'Count', hue = 'Mjob', palette = 'Set1')
plt.title("Gender ~ Mother Job")
plt.show()


# In[35]:


#group gender & father job
gender_father = df.groupby(['sex', 'Fjob']).size().reset_index(name = 'Count')

#visualize gender ~ father job
plt.figure(figsize = (8,6))
sns.barplot(data = gender_father, x = 'sex', y = 'Count', hue = 'Fjob', palette = 'Set2')
plt.title("Gender ~ Father Job")
plt.show()


# In[36]:


#group gender & reason
gender_reason = df.groupby(['sex', 'reason']).size().reset_index(name = 'Count')

#visualize gender ~ reason
plt.figure(figsize = (8,6))
sns.barplot(data = gender_reason, x = 'sex', y = 'Count', hue = 'reason')
plt.title("Gender ~ Reason")
plt.show()


# In[37]:


#group gender & guardian
gender_guardian = df.groupby(['sex', 'guardian']).size().reset_index(name = 'Count')

#visualize gender ~ guardian
plt.figure(figsize = (8,6))
sns.barplot(data = gender_guardian, x = 'sex', y = 'Count', hue = 'guardian', palette = 'Set1')
plt.title("Gender ~ Guardian")
plt.show()


# In[38]:


#group gender & school supply
gender_school = df.groupby(['sex', 'schoolsup']).size().reset_index(name = 'Count')

#visualize gender ~ school supply
plt.figure(figsize = (8,6))
sns.barplot(data = gender_school, x = 'sex', y = 'Count', hue = 'schoolsup', palette = 'Set2')
plt.title("Gender ~ School Supply")
plt.show()


# In[39]:


#group gender & family supply
gender_famliy = df.groupby(['sex', 'famsup']).size().reset_index(name = 'Count')

#visualize gender ~ family supply
plt.figure(figsize = (8,6))
sns.barplot(data = gender_famliy, x = 'sex', y = 'Count', hue = 'famsup')
plt.title("Gender ~ Family Supply")
plt.show()


# In[40]:


#group gender & paid by student
gender_paid = df.groupby(['sex', 'paid']).size().reset_index(name = 'Count')

#visualize gender ~ paid by student
plt.figure(figsize = (8,6))
sns.barplot(data = gender_paid, x = 'sex', y = 'Count', hue = 'paid', palette = 'Set1')
plt.title("Gender ~ Paid by Student")
plt.show()


# In[41]:


#group gender & activities
gender_activities = df.groupby(['sex', 'activities']).size().reset_index(name = 'Count')

#visualize gender ~ activities
plt.figure(figsize = (8,6))
sns.barplot(data = gender_activities, x = 'sex', y = 'Count', hue = 'activities', palette = 'Set2')
plt.title("Gender ~ Activities")
plt.show()


# In[42]:


#group gender & nursery
gender_nursery = df.groupby(['sex', 'nursery']).size().reset_index(name = 'Count')

#visualize gender ~ nursery
plt.figure(figsize = (8,6))
sns.barplot(data = gender_nursery, x = 'sex', y = 'Count', hue = 'nursery')
plt.title("Gender ~ Nursery")
plt.show()


# In[43]:


#group gender & higher
gender_higher = df.groupby(['sex', 'higher']).size().reset_index(name = 'Count')

#visualize gender ~ higher
plt.figure(figsize = (8,6))
sns.barplot(data = gender_higher, x = 'sex', y = 'Count', hue = 'higher', palette = 'Set1')
plt.title("Gender ~ Higher")
plt.show()


# In[44]:


#group gender & internet
gender_internet = df.groupby(['sex', 'internet']).size().reset_index(name = 'Count')

#visualize gender ~ internet
plt.figure(figsize = (8,6))
sns.barplot(data = gender_internet, x = 'sex', y = 'Count', hue = 'internet', palette = 'Set2')
plt.title("Gender ~ Internet")
plt.show()


# In[45]:


#group gender & romantic
gender_romantic = df.groupby(['sex', 'romantic']).size().reset_index(name = 'Count')

#visualize gender ~ romantic
plt.figure(figsize = (8,6))
sns.barplot(data = gender_romantic, x = 'sex', y = 'Count', hue = 'romantic')
plt.title("Gender ~ Romantic")
plt.show()


# In[46]:


#visualize histogram of each attribute
df.hist(figsize = (12,12), color = 'purple')
plt.show()


# # Regression Model

# In[47]:


#handling categorical data
df = pd.get_dummies(df, drop_first = True)
df.head()


# In[48]:


#split data 
X = df.drop('G3', axis = 1)
y = df['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[49]:


#Linear Regression
lr = LinearRegression()
get_ipython().run_line_magic('time', 'lr.fit(X_train, y_train)')
lr.score(X_test, y_test)


# In[50]:


#Lasso Regression
lasso = Lasso(alpha = 0.5)
get_ipython().run_line_magic('time', 'lasso.fit(X_train, y_train)')
lasso.score(X_test, y_test)


# ## Linear Regression Model is the best accuracy score result on 89%

# In[51]:


#prediction
y_pred = lr.predict(X_test)
print(y_pred)


# In[52]:


#check MAE, MSE & RMSE
print('Mean Absolute Error : ', metrics.mean_absolute_error(y_test, y_pred).round(2))
print('Mean Squared Error : ', metrics.mean_squared_error(y_test, y_pred).round(2))
print('Root Mean Squared Error : ', np.sqrt(metrics.mean_absolute_error(y_test, y_pred).round(2)))


# In[53]:


#visualize model
x = y_test
y = y_pred

plt.figure(figsize = (8,6))
plt.title("Linear Regression Model")
plt.plot(x, y, 'o', color = 'r')

m, b = np.polyfit(x, y, 1)
plt.plot(x, m * x + b, color = 'darkblue')


# In[54]:


#distribution
plt.figure(figsize = (8,6))
sns.distplot(df['G3'], color = 'darkorange')
plt.title("Distribution of G3")
plt.show()


# # Check Feature Importance

# In[55]:


#defining feature
coef = pd.Series(lr.coef_, index = X.columns)

imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])


# In[56]:


#visualize feature
plt.figure(figsize = (10,6))
imp_coef.plot(kind = 'barh', color = 'lightseagreen')
plt.title("Feature Importance")
plt.xlabel('Score')
plt.ylabel('Features')
plt.show()

