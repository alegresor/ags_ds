import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="white")

# load data
df = pd.read_csv('car_prices/data/cars_clean.csv')
df['doors'] = df['doors'].astype(str)
print(df.dtypes)

# pairs plot
plt.figure()
pp = sns.pairplot(df.select_dtypes([int,float]),
    corner=True)
pp.savefig('car_prices/figs/pairs.png',bbox_inches='tight',dpi=100)

# correlation heatmap
plt.figure()
corr = df.corr()
hm = sns.heatmap(corr,
    mask = np.triu(np.ones_like(corr, dtype=np.bool)),
    cmap = sns.diverging_palette(220, 10,as_cmap=True),
    center = 0,
    square = True,
    linewidths = .5,
    cbar_kws = {"shrink": .5})
hm.figure.savefig('car_prices/figs/corr.png',bbox_inches='tight',dpi=100)
