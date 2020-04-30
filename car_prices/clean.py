import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

drop_small_counts = True
bin_cylinders = True

df = pd.read_csv('car_prices/data/cars.csv')
df.columns = ['make','model','year','fuel','hp','cylinders','transmission',
            'wd','doors','category','size','style','highway mpg',
            'city mpg','popularity','msrp']

# drop rows with any null values
df.dropna(how='any',axis=0,inplace=True)

# make
df['make'] = df['make'].str.lower()
small_make = df['make'].value_counts()
small_make = small_make.loc[small_make<10]
if drop_small_counts:
    df = df.loc[~df['make'].isin(small_make.keys())]
else:
    df.loc[df['make'].isin(small_make.keys()),'make'] = 'other'

# year
bins = np.arange(1989,2024,5)
df['year'] = pd.cut(df['year'],bins,labels=np.arange(len(bins)-1))

# Engine Fuel Type
df['fuel'] = df['fuel'].str.lower()
small_fuel = df['fuel'].value_counts()
small_fuel = small_fuel.loc[small_fuel<10]
if drop_small_counts:
    df = df.loc[~df['fuel'].isin(small_fuel.keys())]
else:
    df.loc[df['fuel'].isin(small_fuel.keys()),'fuel'] = 'other'

# cylinders
if bin_cylinders:
    bins = sorted(df['cylinders'].unique())
    bins[0] = bins[0]-1
    df['cylinders'] = pd.cut(df['cylinders'],bins,labels=np.arange(len(bins)-1))

# transmission
df = df.loc[df['transmission']!='UNKNOWN']

# category
cats = set([val for vals in df['category'].tolist() for val in vals.split(',')])
for cat in cats: df['is_%s'%cat.lower()] = df['category'].str.contains(cat)
df.drop('category',axis=1,inplace=True)

# style
df['style'] = df['style'].str.lower()
styles = ['suv','sedan','coupe','hatchback','convertible','pickup','van']
for style in styles: df.loc[df['style'].str.contains(style),'style'] = style

# highway mpg
df.loc[(df['make']=='audi')&(df['model']=='A6'),'highway mpg'] = 35.4

# doors
df['doors'] = df['doors'].astype(str)

# model
df_wo_model = df.drop('model',axis=1)


# outputs
for col in df_wo_model.columns:
    series = df[col]
    vals = series.values
    if series.dtype in ['int64','float64']:
        print(series.describe())
        zscores = (vals-vals.mean())/vals.std()
        outliers = df.loc[abs(zscores)>3,['make','model',col]]
        if len(outliers) > 0:
            outliers = outliers.iloc[(-np.abs(outliers[col].values)).argsort()]
        else:
            outliers = 'None'
        print('\nOutliers:\n'+str(outliers))
    else:
        print(series.value_counts())
    print('\nNull Values',series.isnull().sum())
    print('\n%s\n'%('~'*50))
print('Records:',len(df))

df_wo_model.to_csv('car_prices/data/cars_clean.csv',index=False)