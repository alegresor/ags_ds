import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="white")
# sklearn
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing as prep
#   regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
#   cross validation
from sklearn.model_selection import GridSearchCV

one_hot = True
scale = 'uniform'#'uniform','normal','none'
norm = 'l1'#'l1','l2','max'

# load data
df = pd.read_csv('car_prices/data/cars_clean.csv')
df['doors'] = df['doors'].astype(str)

# preprocessing
y = df['msrp'].values
df.drop('msrp',axis=1,inplace=True)
x_num = df.select_dtypes([int,float]).values
x_bool = df.select_dtypes([bool]).values
x_obj = df.select_dtypes([object]).values
#   scale
if scale == 'uniform':
    x_num = prep.MinMaxScaler().fit_transform(x_num)    
elif scale == 'normal':
    x_num = prep.scale(x_num)
#   normalize
if norm:
    x_num = prep.normalize(x_num,norm)
#   encode
for i in range(x_obj.shape[1]):
    x_obj[:,i] = prep.LabelEncoder().fit_transform(x_obj[:,i])
if one_hot:
    x_obj = prep.OneHotEncoder(drop='first',sparse=False).fit_transform(x_obj)

# train test split
x = np.concatenate([x_num,x_bool,x_obj],axis=1)
x_f,x_p,y_f,y_p = train_test_split(x,y,test_size=.2,random_state=7)
print('x_train shape:',x_f.shape)
print('x_test.shape',x_p.shape)

# models
def print_metrics(y,y_hat,name):
    print(name)
    print('\tmae: %.3f'%mean_absolute_error(y,y_hat))
    print('\trmse: %.3f'%np.sqrt(mean_squared_error(y,y_hat)))
    print('\tr2: %.3f'%r2_score(y,y_hat))

#   linear regression
lr_model = LinearRegression()
lr_model.fit(x_f,y_f)
y_hat = lr_model.predict(x_p)
print_metrics(y_p,y_hat,'linar regression')
#   decision tree
dt_model = DecisionTreeRegressor(random_state=7)
dt_grid_cv = GridSearchCV(dt_model,
    param_grid = {
        'max_depth': [5, 10, 15]},
    cv = 3)
dt_grid_cv.fit(x_f,y_f)
y_hat = dt_grid_cv.predict(x_p)
print_metrics(y_p,y_hat,'decision tree')
#   random forest
rf_model = RandomForestRegressor(random_state=7)
rf_grid_cv = GridSearchCV(rf_model,
    param_grid = {
        'max_depth': [5, 10, 15],
        'n_estimators': [10, 15, 20]},
    cv = 3)
rf_grid_cv.fit(x_f,y_f)
y_hat = rf_grid_cv.predict(x_p)
print_metrics(y_p,y_hat,'random forest')
#   gradient boosted trees
gbt_model = GradientBoostingRegressor(random_state=7)
gbt_model.fit(x_f,y_f)
predictions = gbt_model.predict(x_p)
print_metrics(y_p,y_hat,'gradient boosted trees')
