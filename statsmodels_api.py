# %%
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas
from patsy import dmatrices

import polars as pl
from funcs import *
# %%
df = sm.datasets.get_rdataset("Guerry", "HistData").data
vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']
df = df[vars]
df[-5:]
df = df.dropna()
df[-5:]

# df_pl = pl.DataFrame(df)
# df_pl = to_ordered_enum(df_pl, ["Department", "Region"])
y, X = dmatrices('Lottery ~ Literacy + Wealth + Region', 
                 data=df, return_type='dataframe')
y
X

### model fit and summary:
# 1. Describe model using a model class
mod = sm.OLS(y, X)

# 2. Fit model using a class method
res = mod.fit()

# 3. Summarize model
print(res.summary())
dir(res)

# diagnostics/specification tests
sm.stats.linear_rainbow(res)
print(sm.stats.linear_rainbow.__doc__)

sm.graphics.plot_partregress('Lottery', 'Wealth', ['Region', 'Literacy'],
                           data=df, obs_labels=False)

# %%    ### Using formula api ###
dat = sm.datasets.get_rdataset("Guerry", "HistData").data
results = results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', 
                            data=dat).fit()
print(results.summary())

nobs = 100
X = np.random.random((nobs, 2))
X = sm.add_constant(X)
X
beta = [1, .1, .5]
e = np.random.random(nobs)
y = np.dot(X, beta) + e
results = sm.OLS(y, X).fit()
print(results.summary())

sm.webdoc(sm.OLS, stable=True)
# %%
