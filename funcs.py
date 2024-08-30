# %%
# Library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import statsmodels as sm
import pyarrow as pa
pa.__file__
pa.__version__

import polars as pl
import polars.selectors as cs

from datetime import datetime, timedelta, date
from pandas.tseries.offsets import Hour, Minute, Day, MonthEnd
from pandas.tseries.frequencies import to_offset
from scipy.stats import percentileofscore

import ISLP as islp

import xlsxwriter

plt.style.use('ggplot')
ax1 = sns.set_style(style=None, rc=None )

# sns.axes_style()
# sns.set_style()
# sns.plotting_context()
# sns.set_context()
# sns.despine()


# %%
def pairs(
    data: pl.DataFrame,
    x_vars, y_vars, hue=None,
    numerical_numerical_1 = sns.scatterplot,
    numerical_numerical_2 = sns.histplot,
    discrete_numerical_1 = sns.boxplot,
    discrete_numerical_2 = sns.violinplot,
    discrete_discrete_1 = sns.histplot,
    discrete_discrete_2 = None,
    diag_numerical = sns.histplot,
    diag_discrete = sns.countplot,
    
    numerical_numerical_1_kwargs = {},
    numerical_numerical_2_kwargs = {},
    discrete_numerical_1_kwargs = {},
    discrete_numerical_2_kwargs = {},
    discrete_discrete_1_kwargs = {},
    discrete_discrete_2_kwargs = {},
    diag_numerical_kwargs = {},
    diag_discrete_kwargs = {},
    
    **subplots_kwargs
    ):
    
    # g = sns.PairGrid(data=data, x_vars=x_vars, y_vars=y_vars, 
    #              **kwargs)
    _, g = plt.subplots(len(y_vars),len(x_vars), **subplots_kwargs)

    # for ax in g.axes.flatten():
    for i in range(len(x_vars)):
        for j in range(len(y_vars)):
            x = x_vars[i]
            y = y_vars[j]
            ax = g[j,i]
            x_dtype = data.dtypes[data.get_column_index(x)]
            y_dtype = data.dtypes[data.get_column_index(y)]
            
            if x_dtype in [pl.Categorical, pl.Enum, pl.String]:
                x_dtype = "discrete"
            else:
                x_dtype = "numerical"
            if y_dtype in [pl.Categorical, pl.Enum, pl.String]:
                y_dtype = "discrete"
            else:
                y_dtype = "numerical"
            
            # diagonal
            # changed from if i == j?
            if x == y:
                y = None
                if x_dtype == "discrete":        
                    func, func_kwargs = (diag_discrete, diag_discrete_kwargs)
                else:
                    func, func_kwargs = (diag_numerical, diag_numerical_kwargs)
            # lower triangle
            elif i < j:
                if x_dtype == "discrete" and y_dtype == "discrete":
                    func, func_kwargs = (discrete_discrete_1, discrete_discrete_1_kwargs)
                if x_dtype == "numerical" and y_dtype == "numerical":
                    func, func_kwargs = (numerical_numerical_1, numerical_numerical_1_kwargs)
                else:
                    func, func_kwargs = (discrete_numerical_1, discrete_numerical_1_kwargs)
            # upper triangle
            else:
                if x_dtype == "discrete" and y_dtype == "discrete":
                    func, func_kwargs = (discrete_discrete_2, discrete_discrete_2_kwargs)
                if x_dtype == "numerical" and y_dtype == "numerical":
                    func, func_kwargs = (numerical_numerical_2, numerical_numerical_2_kwargs)
                else:
                    func, func_kwargs = (discrete_numerical_2, discrete_numerical_2_kwargs)
            # upper triangle
            
            # draw the graph on the axis
            func(data=data, x=x, y=y, ax=ax, hue=hue, **func_kwargs)
            
    return g

# %%    ### Fix for -inf/inf categories from qcut/cut function in polars ###
def fix_cut(data: pl.DataFrame, colnames, col_mins, col_maxs):
    exprs = []
    
    ### For each column, replace -infs/infs with the supplied column mins/maxs
    ### Also converts the column to an Enum
    for col, col_min, col_max in zip(colnames, col_mins, col_maxs):
        min_format = "[{}".format(col_min)
        max_format = "{}]".format(col_max)
        
        enum_list = (
            data[col]
            .unique().cast(pl.String).str.replace_many(
            ["(-inf",   "[-inf",     "inf)",     "inf]" ], 
            [min_format, min_format, max_format, max_format])
            .to_list()
        )

        exprs.append(
            pl.col(col).cast(pl.String)
            .str.replace_many(
                ["(-inf",   "[-inf",     "inf)",     "inf]" ], 
                [min_format, min_format, max_format, max_format])
            .cast(pl.Enum(enum_list))
        )
        
    return data.with_columns(exprs)

test_sql = pl.DataFrame([pl.Series("some_name", [1,4,5,1]), pl.Series("some_name2", [2,6,1,7])])
fix_cut(test_sql.select(pl.col("some_name").qcut(2), pl.col("some_name2").qcut(3)), 
        ["some_name", "some_name2"], [1, 1], [5, 7]).dtypes

# %% Test-Train split
def train_test_split_lazy(
    df: pl.LazyFrame, train_fraction: float = 0.75
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Split polars dataframe into two sets.
    Args:
        df (pl.DataFrame): Dataframe to split
        train_fraction (float, optional): Fraction that goes to train. Defaults to 0.75.
    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: Tuple of train and test dataframes
    """
    df = df.with_columns(pl.all().shuffle(seed=1)).with_row_index()
    # df = df.with_columns(pl.all().shuffle(seed=1))
    
    # df_train = df.filter(pl.col("index") < pl.col("index").max() * train_fraction)
    # df_test = df.filter(pl.col("index") >= pl.col("index").max() * train_fraction)
    
    # df_train = df.filter(pl.col("index") < pl.len() * train_fraction)
    # df_test = df.filter(pl.col("index") >= pl.len() * train_fraction)

    # this is better and faster than above
    df_height = df.select(pl.len()).collect().item()
    train_num = round(df_height * train_fraction)
    test_num = df_height - train_num
    df_train = df.head( train_num )
    df_test = df.tail( test_num )
    
    return df_train, df_test