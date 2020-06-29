#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Display plots inline
get_ipython().run_line_magic('matplotlib', 'inline')

# Data libraries
import pandas as pd
import numpy as np

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Plotting defaults
plt.rcParams['figure.figsize'] = (8,5)
plt.rcParams['figure.dpi'] = 80

# sklearn modules
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


# In[2]:


#load the data
data = pd.read_csv("forpython.csv")


# In[4]:


data.info()


# In[ ]:


"BloodStool","MucoidStool","AbdominalPain","ThreeDaysFever","unclass_virus",
"single_common","single_uncommon","between_virus","common_count","uncommon_count"


# In[17]:


sns.pairplot(data, vars = ["stay","age","NumberDiarEpi",],kind = 'reg', diag_kind = 'kde',
             plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})


# In[3]:


y = data['stay']
x = data.drop('stay',axis = 1)


# In[4]:


def preprocess(data,scale):
    """Process the data before feeding into models.
       It's a combination of log-transfromation, dummy coding and scaling.
       
    Args:
        data: dataframe value, data sets to be preprocessed
        scale: boolean value, should data value be scaled
        
    """
    ##log_transformation
    #data['log_sale_price'] = np.log(data['sale_price'])
    #data['log_lot_area'] = np.log(data['lot_area'])
    #data['house_age'] = data['year_sold']- data['year_built']
    
    y = data['stay']
    
    #sales['log_sale_price'] = np.log(sales['sale_price'])
    #sales['log_lot_area'] = np.log(sales['lot_area'])
    #sales['house_age'] = sales['year_sold']- sales['year_built']
    data_dummy = data.copy()
    
    #dummy coding
    data_scale = pd.get_dummies(data_dummy).drop(columns = ['stay'])

    
    #scale the value
    if scale == True:
        S = StandardScaler().fit(data_scale)
        data_scale = S.transform(data_scale)
   
    return y, data_scale


# In[5]:


y, x = preprocess(data, False)
y, x_scaled = preprocess(data, True)


# # kernel ridge

# In[8]:


m = GridSearchCV(KernelRidge(kernel='rbf'),
                 param_grid={"alpha": np.logspace(-5, 0, num=10),
                             "gamma": np.logspace(-5, 0, num=10)},
                 scoring = 'neg_mean_squared_error', cv=5)
m = m.fit(x_scaled,y)
m.best_params_


# In[34]:


m = KernelRidge(kernel='rbf', alpha=1.0, gamma=1e-05).fit(x_scaled, y)


# In[35]:


var_contribution=np.dot(x.transpose(),m.dual_coef_)
var_contribution = pd.DataFrame(np.round(var_contribution, 4),  
                          x.columns, columns = ["Kernel ridge Coefficients"])
var_contribution = var_contribution.sort_values(by = "Kernel ridge Coefficients", 
ascending = False)
var_contribution

