# %%
# potential solutions: https://www.lackos.xyz/itsl/
# good package explanation https://stackoverflow.com/questions/9048518/importing-packages-in-python

# Library imports
from funcs import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
# import statsmodels as sm
import statsmodels.api as sm
from statsmodels.stats.outliers_influence \
import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm

import pyarrow as pa

import polars as pl
import polars.selectors as cs

from datetime import datetime, timedelta, date
from pandas.tseries.offsets import Hour, Minute, Day, MonthEnd
from pandas.tseries.frequencies import to_offset
from scipy.stats import percentileofscore
import scipy.stats as stats

import xlsxwriter



from functools import partial

import sklearn.metrics as sklm
import sklearn as skl
import sklearn.discriminant_analysis as skl_da
from sklearn.model_selection import \
     (cross_validate,
      KFold,
      ShuffleSplit)
from sklearn.base import clone

from sklearn.discriminant_analysis import \
     (LinearDiscriminantAnalysis as LDA,
      QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as skm
from sklearn.model_selection import train_test_split
import sklearn.linear_model as skl
from sklearn.linear_model import \
     (LinearRegression,
      LogisticRegression,
      Lasso)

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

from sklearn.tree import (DecisionTreeClassifier as DTC,
                          DecisionTreeRegressor as DTR,
                          plot_tree,
                          export_text)
from sklearn.metrics import (accuracy_score,
                             log_loss)
from sklearn.ensemble import \
     (RandomForestRegressor as RF,
      GradientBoostingRegressor as GBR)
from sklearn.svm import SVC
from sklearn.metrics import RocCurveDisplay

import ISLP as islp
from ISLP import load_data
from ISLP.models import (ModelSpec as MS ,
summarize ,
poly)

from ISLP.models import sklearn_sm

from ISLP import confusion_table
from ISLP.models import contrast

from ISLP.models import \
     (Stepwise,
      sklearn_selected,
      sklearn_selection_path)

from pygam import (s as s_gam,
                   l as l_gam,
                   f as f_gam,
                   LinearGAM,
                   LogisticGAM)

from ISLP.transforms import (BSpline,
                             NaturalSpline)
from ISLP.models import bs, ns
from ISLP.pygam import (approx_lam,
                        degrees_of_freedom,
                        plot as plot_gam,
                        anova as anova_gam)

from ISLP.svm import plot as plot_svm

from ISLP.torch import (SimpleDataModule,
                        SimpleModule,
                        ErrorTracker,
                        rec_num_workers)
from ISLP.torch.imdb import (load_lookup,
                             load_tensor,
                             load_sparse,
                             load_sequential)

# from ISLP.bart import BART

import torch
from torch import nn
from torch.optim import RMSprop
from torch.utils.data import TensorDataset
from torchmetrics import (MeanAbsoluteError,
                          R2Score)
from torchinfo import summary
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import seed_everything
seed_everything(0, workers=True)
torch.use_deterministic_algorithms(True, warn_only=True)

from torchvision.io import read_image
from torchvision.datasets import MNIST, CIFAR100
from torchvision.models import (resnet50,
                                ResNet50_Weights)
from torchvision.transforms import (Resize,
                                    Normalize,
                                    CenterCrop,
                                    ToTensor)
from glob import glob
import json

import torchvision


plt.style.use('ggplot')
ax1 = sns.set_style(style=None, rc=None )

# sns.axes_style()
# sns.set_style()
# sns.plotting_context()
# sns.set_context()
# sns.despine()

# %%
data = pl.scan_csv("freMTPL2freq.csv", 
                   schema_overrides={"IDpol": pl.Float64}
                   ).collect()
data = data.with_columns(pl.col("IDpol").cast(pl.Int64))
data.describe()
# %%
import hvplot.polars
import hvplot.pandas
import panel as pn
pn.extension('bokeh')

data.plot.scatter(x="VehPower",y="Exposure",by="Area")
explorer = data.to_pandas().hvplot.explorer()
pn.serve(explorer) 


df = pl.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [5, 4, 3, 2, 1],
    'C': [2, 3, 4, 5, 6]
})
explorer = df.to_pandas().hvplot.explorer()
pn.serve(explorer)
# %%
data.schema
data.null_count()
data.filter(data.is_duplicated())

# %% ### some hvplots

data.sort("Area").hvplot(
    kind='box',
    by=['Area'],
    # x='Area',
    y='Exposure',
    legend='bottom_right',
    widget_location='bottom',
)


data.plot.box(
    by=['Area', 'VehGas'],
    # x='Area',
    y='Exposure',
    legend='bottom_right',
    groupby = 'DrivAge'
    # widget_location='bottom',
)

data.group_by("Area").agg(
    pl.all().median()
).sort("Area")
# %%
w_latitude = pn.widgets.DiscreteSlider(
    name='Latitude', options=list(data["VehAge"].unique()))
w_latitude

# %%

# %%
college = pl.scan_csv("data/college.csv").collect()
college
# var_overview(college, "var_overview.xlsx")
college_to_plot = college.select("Top10perc", "Apps", "Enroll")
hvplot.scatter_matrix(college_to_plot.to_pandas())
college.plot.scatter(x="Top10perc", y="Enroll")
from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap
(
    ggplot(college) +  geom_point(aes(x="Top10perc", y="Enroll"))
)
plot = hvplot.scatter_matrix(college_to_plot.to_pandas())

from bokeh.plotting import figure, show

# Create a figure with fixed sizing mode and set width and height
p = figure(sizing_mode='fixed', width=1200, height=600)
hvplot.save(plot, 'test.html')
f, ax = plt.subplots(figsize=(7, 5))

plt.figure()
sns.displot(college.select("Enroll"), kde=True)
plt.show()
plt.close('all')
# %%
test_series = pl.Series(name='test',values=["A","B","C","D"])
test = pl.DataFrame(test_series)

mapping_from = test["test"].unique().sort()
mapping_to = ["A","A","A","DD"]

mapping_1 = {}
for val in mapping_from:
    if val >= "D":
        mapping_1[val] = "DD"
    else:
        mapping_1[val] = "A"
mapping_1

# testing different methods for replacing/mapping
# map a single value (can also just use the list method below)
test.with_columns(pl.col("test").replace("B","A"))
# map multiple - 2 lists vs a dictionary
test.with_columns(pl.col("test").replace(mapping_from,mapping_to))
test.with_columns(pl.col("test").replace(mapping_1))
# map using is_in and pl.when etc.
test.with_columns(
    pl.when(pl.col("test").is_in(["A","B"])).then(pl.lit("A")).otherwise(
    pl.when(pl.col("test") == "C").then(pl.lit("A")).otherwise(pl.lit("D")))
    .alias("test2")
)

# %% ################ SEABORN #########################
dots = sns.load_dataset("dots")
tips = sns.load_dataset("tips")
penguins = sns.load_dataset("penguins")
titanic = sns.load_dataset("titanic")
iris = sns.load_dataset("iris")

dots_pl = pl.DataFrame(dots)
dots_pl = dots_pl.with_columns(cs.string().cast(pl.Categorical("lexical")))
dots_pl

### RELATIONSHIPS - MANY VARIABLES ###
fmri = sns.load_dataset('fmri')
pl_fmri = pl.DataFrame(fmri)
fmri_gb = pl_fmri.group_by("timepoint","region","event").agg(pl.col("signal").mean())
sns.relplot(
    data=fmri, kind="line", col="region", row="event",
    x="timepoint", y="signal", hue="region", style="event", size="region",
    dashes=False, markers=True, estimator=None
)
sns.relplot(
    data=fmri_gb, kind="line", col="region", row="event", height=3, aspect=1.5,
    linewidth=2.5, # col_wrap=?, row_wrap=? col_order=[..], row_order=[..]
    x="timepoint", y="signal", hue="region", style="event", size="region",
    # hue/size order/norm, style order
    dashes=False, markers=True, estimator=None,
    facet_kws=dict(sharex=False, sharey=False),
)

plt.show()
plt.close('all')

### DISTRIBUTIONAL ###
sns.displot(data=tips, kind="hist",x="total_bill", col="time", kde=True, rug=False)
sns.displot(data=tips, kind="ecdf", x="total_bill", col="time", hue="smoker", rug=True)
sns.displot(data=tips, kind="hist", x="time", y="smoker")

# BIVARIATE DISTRIBUTIONAL
# bin-width for x and y respectively
sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", hue="species",
            binwidth=(2, .5), cbar=False)
penguins

plt.show()
plt.close('all')

### CATEGORICAL ###
sns.catplot(data=tips, kind="swarm", x="day", y="total_bill", hue="smoker")
sns.catplot(data=tips, kind="violin", x="day", y="total_bill", hue="smoker", split=True)
sns.catplot(data=tips, kind="box", x="day", y="total_bill", hue="smoker")
sns.catplot(data=tips, kind="boxen", x="day", y="total_bill", hue="smoker")
sns.catplot(data=tips, kind="bar", x="day", y="total_bill", hue="smoker")
sns.catplot(data=tips, kind="count", x="smoker", hue="day",order=["No", "Yes"])
sns.catplot(
    data=tips, x="day", y="total_bill", hue="sex",
    kind="violin", split=True, bw_adjust=.5, cut=0,
)
sns.catplot(
    data=titanic, x="class", y="survived", hue="sex",
    palette={"male": "g", "female": "m"},
    markers=["^", "o"], linestyles=["-", "--"],
    kind="point"
)
g = sns.catplot(data=tips, x="day", y="total_bill", kind="violin", inner=None)
sns.swarmplot(data=tips, x="day", y="total_bill", color="k", size=3, ax=g.ax)
sns.catplot(
    data=tips, x="day", y="total_bill", hue="sex",
    kind="violin", inner="stick", split=True, palette="pastel",
    ax=g.ax
)
g = sns.catplot(
    data=titanic,
    x="fare", y="embark_town", row="class",
    kind="box", orient="h",
    sharex=False, margin_titles=True,
    height=1.5, aspect=4,
)
g.set(xlabel="Fare", ylabel="")
g.set_titles(row_template="{row_name} class")
for ax in g.axes.flat:
    ax.xaxis.set_major_formatter('${x:.0f}')

plt.show()
plt.close('all')

### MULTI VARIATE ###
sns.jointplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", hue="species")
sns.pairplot(tips, x_vars=["total_bill", "size"], y_vars=["tip"],
             hue="smoker", height=5, aspect=.8, kind="reg")
g = sns.PairGrid(iris, x_vars=["total_bill", "size"], y_vars=["tip"], hue="species")
g.map_diag(sns.histplot, kde=True)
g.map_offdiag(sns.scatterplot)
g.add_legend()

from scipy import stats
def quantile_plot(x, **kwargs):
    quantiles, xr = stats.probplot(x, fit=False)
    plt.scatter(xr, quantiles, **kwargs)
def qqplot(x, y, **kwargs):
    _, xr = stats.probplot(x, fit=False)
    _, yr = stats.probplot(y, fit=False)
    plt.scatter(xr, yr, **kwargs)
with sns.axes_style("white"):
    g = sns.FacetGrid(tips, row="sex", col="smoker", hue="time",
                      margin_titles=True, height=2.5, aspect=1,
                      row_order = ["Male", "Female"], col_order = ["No","Yes"],
                      palette={"Lunch": "seagreen", "Dinner":".7"}
                      )
g.map(sns.scatterplot, "total_bill", "tip", #   color="#334488"
      )
for ax in g.axes_dict.values():
    ax.axline((0, 0), slope=.2, c=".2", ls="--", zorder=0)
# g.set_axis_labels(x_var= , y_var=)
g.set(xlim=(0,60), ylim=(0, 15), xticks=[10, 30, 50], yticks=[2, 6, 10],
      xlabel="Total bill (US Dollars)", ylabel="Tip",
    #   title="My TITLE"
      )
g.figure.subplots_adjust(wspace=.02, hspace=.02)
g.add_legend()
g.map(quantile_plot, "total_bill")
g.map(qqplot, "total_bill", "tip")


def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=15, cmap=cmap, **kwargs)
with sns.axes_style("dark"):
    g = sns.FacetGrid(tips, hue="time", col="time", height=4)
g.map(hexbin, "total_bill", "tip", extent=[0, 50, 0, 10])


g = sns.PairGrid(penguins, hue="species", corner=True
                #  palette=.., x_vars=.. , y_vars=.. , height=., aspect=.
                 )
g.map_lower(sns.kdeplot, fill=True,hue=None, levels=5, color=".2")
g.map_lower(sns.scatterplot, marker="+")
g.map_diag(sns.histplot, element="step", linewidth=0, kde=True)
g.add_legend(frameon=True)
g.legend.set_bbox_to_anchor((.61, .6))

# two methods, PairGrid more flexible
g = sns.pairplot(iris, hue="species", palette="Set2", diag_kind="kde", height=2.5)
g = sns.PairGrid(iris, hue="species", palette="Set2", height=2.5)
g.map_diag(sns.histplot, kde=True)
g.map_offdiag(sns.scatterplot)

plt.show()
plt.close('all')

### CUSTOMIZATIONS? ###
g = sns.relplot(
    data=penguins,
    x="bill_length_mm", y="bill_depth_mm", hue="body_mass_g",
    palette=sns.color_palette("crest", as_cmap=True), 
    marker='o', 
    s=100
)
g.set_axis_labels("Bill length (mm)", "Bill depth (mm)", labelpad=10)
g.legend.set_title("Body mass (g)")
g.figure.set_size_inches(6.5, 4.5)
# g.ax.margins(.15)
g.despine(trim=True)
# g.legend.remove()
# g.add_legend()

plt.show()
plt.close('all')
# %%
import seaborn.objects as so
test1 = data.group_by("Area","VehGas").agg(
    pl.col("Exposure").sum()
).sort("Area")

sns.barplot(test1, x="Area",y="Exposure",estimator='mean')
# %%
sns.barplot(test1, x="Area",y="Exposure",estimator='median')
# %% Multiple plots overlayed together
sns.set_style(style=None, rc=None )
sns.set_theme(palette="deep")
(
    so.Plot(test1, x="Area",y="Exposure")
    .add(so.Bar())
    .add(so.Bar(), so.Agg(func='median'))
    .add(so.Dot(color='red'), so.Agg(func='mean'))
    .add(so.Dot(pointsize=10, color='green'))
)
# %%
(
    so.Plot(data, x="Area",y="Exposure",color="VehGas")
    # .add(so.Bar())
    .add(so.Bar(), so.Agg(func='median'), so.Dodge())
    .add(so.Dot(), so.Agg(func='mean'))
    # .add(so.Dot(pointsize=10, color='green'))
)
# %%
summarised = data.group_by("Area","VehGas").agg(
    pl.col("Exposure").mean().alias("mean_exposure"),
    pl.col("Exposure").median().alias("median_exposure"),
).sort("Area")

# %%
fig, ax1 = plt.subplots( figsize=(15,6) )
ax2 = ax1.twinx()
bp = sns.barplot(summarised, x="Area",y="mean_exposure", hue="VehGas", ax=ax1)
sp = sns.scatterplot(summarised, x="Area",y="mean_exposure", hue="VehGas",ax=ax2)
plt.show()
# bp.legend.set_title("blah")
# bp.get_legend_handles_labels()
# bp.get_legend()
sns.move_legend(sp, (1.1, 0.5))
# sp.legend(labels="scatter_mean_Exposure")
  # %%
df = pd.DataFrame(
    {"x": [1, 2, 3, 4, 5], "y1": [5, 2, 1, 6, 2], "y2": [1000, 240, 1300, 570, 120]}
)
fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
ax2 = ax1.twinx()
ax1.tick_params(axis="x", labelrotation=45)
p1 = (
    so.Plot(df, x="x", y="y1")
    .add(so.Bar(width=0.7))
    .label(x="Month", y="Data 1", title="TITLE")
    .on(ax1)
    .plot()
)
p2 = (
    so.Plot(df, x="x", y="y2")
    .add(so.Line(color="orange", linewidth=3))
    .label(y="Data 2", title="TITLE")
    .scale(y=so.Continuous().label(like="{x:,.0f}"))
    .on(ax2)
    .plot()
)

# %%  seaborn.objects interface
diamonds = sns.load_dataset('diamonds')

### Faceting/legend/scaling/axis customizations
(
    so.Plot(diamonds, x="carat", y="price", color="carat", marker="cut")
    # .facet(col="...", wrap="...", row="...", 
    #    order={"col": ["Gentoo", "Adelie"], "row": ["Female", "Male"]})
    .add(so.Dots())
    .scale(
        x=so.Continuous().tick(every=0.5),
        y=so.Continuous().tick(every=2500).label(like="${x:.0f}"),
        color=so.Continuous().tick(at=[1, 2, 3, 4]),
        marker=so.Nominal(list(range(5)), order=diamonds["cut"].unique()),
    )
)

### Customizing limits, labels, and titles
(
    so.Plot(penguins, x="body_mass_g", y="species", color="island")
    .facet(col="sex")
    .add(so.Dot(), so.Jitter(.5))
    .share(x=False)
    .limit(y=(2.5, -.5))
    .label(
        x="Body mass (g)", y="",
        color=str.capitalize,
        title="{} penguins".format,
    )
)
### Theme customization
from seaborn import axes_style
theme_dict = {**axes_style("whitegrid"), "grid.linestyle": ":"}
so.Plot().theme(theme_dict)

### OR to change theme for all Plot instances
# so.Plot.config.theme.update(theme_dict)
# %%
# Create a plot with text annotations and grouping
p = so.Plot(diamonds, x="carat", y="price", color="cut")
p.add(so.Dot(), so.Text(text="cut"), group="cut")

# Display the plot
p.show()
# %%
data = np.random.normal(size=(20, 6)) + np.arange(6) / 2
f, ax = plt.subplots()
sns.violinplot(data=data)
sns.despine(offset=10, trim=True, left=True)
# sns.despine(offset=10, trim=False)

# %%
def sinplot(n=10, flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, n + 1):
        plt.plot(x, np.sin(x + i * .5) * (n + 2 - i) * flip)
f = plt.figure(figsize=(6, 6))
gs = f.add_gridspec(2, 2)

with sns.axes_style("darkgrid"):
    ax = f.add_subplot(gs[0, 0])
    sinplot(6)

with sns.axes_style("white"):
    ax = f.add_subplot(gs[0, 1])
    sinplot(6)

with sns.axes_style("ticks"):
    ax = f.add_subplot(gs[1, 0])
    sinplot(6)

with sns.axes_style("whitegrid"):
    ax = f.add_subplot(gs[1, 1])
    sinplot(6)
    
f.tight_layout()
# %%
# CAN also set context, style, fontSize, colours/palettes, rcParams 
sns.set_theme()
sns.set_context("paper")
sinplot()
plt.show()
sns.set_context("talk")
sinplot()
plt.show()
sns.set_context("poster")
sinplot()
plt.show()
sns.set_context("notebook")
sinplot()
plt.show()
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
sinplot()
plt.show()
# %%
cmap = sns.color_palette("viridis", as_cmap=True)
# Use the colormap in a plot
sns.scatterplot(data=penguins, x="flipper_length_mm", y="body_mass_g", hue="body_mass_g", palette=cmap)
plt.legend().set(bbox_to_anchor=(1.1, 1.1, 0.25, 0.25), title="BLAH")
# plt.legend().set_bbox_to_anchor((1.1, 1.1, 0.25, 0.25))
plt.show()

# %% ### LOOPING THROUGH  VARIABLES MANUALLY FOR PairGrid
iris = sns.load_dataset("iris")
iris_pl = pl.DataFrame(iris)
iris_pl.schema
iris_pl.select(cs.numeric()).columns

x_vars = ["sepal_width","species"]
y_vars = ["petal_length","petal_width"]
g = sns.PairGrid(data=iris, x_vars=x_vars, y_vars=y_vars, 
                 height=4,aspect=1.5)

# g.map_diag(sns.scatterplot)
# g.map_offdiag(sns.scatterplot)
# g.map_lower(sns.scatterplot)
temp = iris_pl.group_by("species").agg(pl.col("petal_length").mean())
temp
# sns.barplot(temp, x="species",y="petal_length")
def testpairgrid(x, y, **kwargs):
    temp = iris_pl.group_by(x).agg(pl.col(y).mean())
    sns.barplot(x=temp.select(x), y=temp.select(y), **kwargs)

# for ax in g.axes.flatten():
for i in range(len(x_vars)):
    for j in range(len(y_vars)):
        x = x_vars[i]
        y = y_vars[j]
        
        ax = g.axes[j,i]
        
        if x == "species":
            sns.boxplot(iris, x=x, y=y, ax=ax)
        else:
            sns.scatterplot(iris, x=x, y=y, ax=ax)

# %%  ### MULTIPLE GRAPHS ON ONE PLOT, CUSTOM DATA AGGREGATIONS
sns.catplot(data=tips, kind="count", x="smoker", hue="day",order=["No", "Yes"])
plt.show()
plt.close()

fig, ax = plt.subplots()
# start with a countplot
sns.countplot(data=tips, x="smoker", hue="day",order=["No", "Yes"])
# add secondary point plot to graph
tips_pl = pl.DataFrame(tips)

tips_agg = tips_pl.group_by("smoker","day").agg(
    pl.len().alias("len")
).group_by("smoker").agg(
    pl.col("len").sum().alias("len"),
    (pl.col("len").sum() / pl.len()).alias("avg_across_day")
)

sns.scatterplot(data=tips_agg, x="smoker",y="len", color="black")
sns.lineplot(data=tips_agg, x="smoker",y="avg_across_day", 
                                    color="0.5",marker='o',linestyle='--' )
plt.legend().set_bbox_to_anchor(((1,0.5)))
plt.legend().set(bbox_to_anchor=(1,0.5), title="DAY LEGEND")
plt.suptitle("MAIN TITLE")
plt.title("SUB TITLE",fontsize=6,color='red')
# calling this suppresses the axes printing as some weird list
plt.show()

# %%
titanic = sns.load_dataset("titanic")
sns.swarmplot(titanic, x="class",y="sex")

# %%
sort_cats = pl.DataFrame(
    [pl.Series("blah",["1","15","152"]),
     pl.Series("blah2",[1, 152, 15]),
     pl.Series("blah3",["a","dd","b"])
    ]
    )

test_enum = to_ordered_enum(sort_cats, ["blah", "blah2","blah3"])
test_enum.dtypes
# %%    ### 2.4  page 65 ###

####### 8.)
### a)
college = pl.scan_csv("data/College.csv").collect()
college.head()
college.schema

### c)
college.describe()

### d)
first_cols = ["Top10perc", "Apps", "Enroll"]
g = sns.PairGrid(college.select(first_cols).filter(pl.col("Apps")<40000), 
                 x_vars=first_cols, y_vars=first_cols)
g.map_lower(sns.histplot, bins=20)
# map_upper doesn't seem to be working
g.map_upper(sns.scatterplot)
g.map_diag(sns.histplot, kde=True, bins=20)
g.axes[0,0].set_xlim(0, 50)
g.axes[0,0].set_ylim(0, 0.04)
plt.show()
plt.close()

test1 = college["Top10perc"].value_counts()
sns.barplot(test1, x="Top10perc", y="count")
plt.show()
plt.close()

# custom pairs function testing
from funcs import *
x_vars = ["Top10perc", "Apps", "Enroll"]
x_vars = college.select(cs.numeric()).columns[:10]
x_vars
college.shape
g = pairs(college, x_vars=x_vars, y_vars=x_vars, 
          sharex=False, sharey=False, # hue="Private", 
          figsize=(60, 60))
# plt.gcf().subplots_adjust(bottom=0.05, left=0.1, top=0.95, right=0.95)
plt.suptitle('College Scatter Matrix', fontsize=35, y=0.90)
plt.show()
plt.close()

### e)
plt.figure(figsize=(4,6))
# fig = plt.figure()
# fig.set_size_inches(4, 6)
sns.boxplot(college, x="Elite", y="Outstate")
plt.show()
plt.close()

### f)
college = college.with_columns(
    pl.col("Top10perc").cut([50],labels=["No","Yes"]).alias("Elite"),
    pl.col("Top10perc").qcut([0.25, 0.75], labels=["Low","Med","High"]).alias("Elite_q"),
)
college["Elite"].value_counts()

### g)
fig, ax = plt.subplots(3, 3, figsize=(16,16))
college.schema
bins = [5, 15, 25]

for i in range(len(x_vars)):
    for j in range(len(bins)):
        bin = bins[j]
        var = x_vars[i]
        # var by row, bin by col
        axes = ax[i,j]
        sns.histplot(data=college, x=var, 
                     bins=bin, ax=axes)
        
### h) ...

        

# %%  ##### 9.) page 66

### a)
auto = pl.scan_csv("data/Auto.csv").collect()
auto.schema
auto.head()
auto["cylinders"].value_counts()
auto["origin"].value_counts()
auto["name"].value_counts().sort("count", descending=True)
auto = to_ordered_enum(auto, ["cylinders", "origin", "name"])
auto.head()

auto.null_count()
var_overview(auto, "auto_overview.xlsx")

### b) and c)
### adding a new statistic to a summary
d = auto.describe()
non_num = d.select(cs.exclude(cs.numeric(), "statistic")).columns
non_num_dtypes = [(d.dtypes)[d.get_column_index(name)] for name in non_num]
exprs = [pl.lit(None).cast(dtype).alias(name) 
         for (name, dtype) in zip(non_num, non_num_dtypes)]
exprs
e = auto.select(
    (cs.numeric().max() - cs.numeric().min()).cast(pl.Float64),
    *exprs,
    pl.lit("range").alias("statistic")
)
e.select(d.columns)
d.extend(e.select(d.columns))
# pl.concat([d, e], how="diagonal")

auto.select(
    cs.numeric().max().name.suffix("_max"),
    cs.numeric().min().name.suffix("_min"),
    (cs.numeric().max() - cs.numeric().min()).name.suffix("_range")
).unpivot()

### d)
obs_to_remove = np.arange(9, 85)
auto_2 = (
    auto.with_row_index().filter(~pl.col("index").is_in(obs_to_remove))
    .select(pl.exclude("index"))
)
auto_2
auto_2.describe()

### e)
auto.schema
# name has too many categories to visualize well
test_cols = auto.select(cs.exclude("name")).columns
test_cols
pairs(auto, x_vars=test_cols, y_vars=test_cols, figsize=(48,48))

# %%  ##### 2.4 page 67 #####

### 10a) 10b)
boston = pl.scan_csv("data/Boston.csv").collect()
boston.schema
boston.shape
boston.head()
boston.describe()

### 10c)
var_overview(boston, "boston_overview.xlsx")
g = pairs(boston, figsize=(48,48))
plt.suptitle('Boston Scatter Matrix', fontsize=35, y=0.9)
plt.show()
plt.close()

### 10e)
boston.top_k(10, by=["crim"])
boston.top_k(10, by=["tax"])
boston.top_k(10, by=["ptratio"])
# %%    ##### Lab 3.6 page 117 #####

boston = load_data("Boston")
boston_pl = pl.DataFrame(boston)
boston_pl.columns

X = pd.DataFrame (
    {'intercept ': np.ones(boston.shape[0]),
     'lstat ': boston['lstat']}
    )
X
X_pl = boston_pl.select(
    pl.lit(1).alias("intercept"),
    pl.col("lstat")    
)
X_pl.to_pandas()
Y = pd.DataFrame( boston_pl["medv"], columns=["medv"])
Y
Y_pl = boston_pl["medv"]
Y_pl.to_pandas()

### fit linear model
model = sm.OLS(Y, X)
results = model.fit()
results.params
model_pl = sm.OLS(Y_pl.to_pandas(), X_pl.to_pandas())
results_pl = model_pl.fit()
summarize(results)
summarize(results_pl)

# fit and transform method
design = MS(["lstat"])
design = design.fit(boston)
X = design.transform(boston)
X[:4]

## predict on new data
new_df = pd.DataFrame ({'lstat':[5, 10, 15]})
newX = design.transform(new_df)
newX

new_predictions = results.get_prediction(newX)
# predicted mean, confidence/prediction intervals
new_predictions.predicted_mean
new_predictions.conf_int(alpha=0.05)
new_predictions.conf_int(alpha=0.05, obs=True)

# graph the fit
def abline(ax, b, m, *args, **kwargs):
    "Add a line with slope m and intercept b to ax"
    xlim = ax.get_xlim()
    ylim = [m * xlim[0] + b, m * xlim[1] + b]
    ax.plot(xlim, ylim, *args, **kwargs)
ax = boston.plot.scatter('lstat', 'medv')
abline(ax,
       results.params[0],
       results.params[1],
       'r--',
       linewidth=3)
pairs(boston_pl, x_vars=["lstat"], y_vars=["medv"])

# residual plot
ax = plt.subplots(figsize=(8,8))[1]
ax.scatter(results.fittedvalues, results.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--')

# leverage statistics
infl = results.get_influence()
ax = plt.subplots(figsize=(8,8))[1]
ax.scatter(np.arange(X.shape[0]), infl.hat_matrix_diag)
ax.set_xlabel('Index')
ax.set_ylabel('Leverage')
np.argmax(infl.hat_matrix_diag)
infl.hat_matrix_diag.shape
pl.DataFrame(dir(infl)).filter(
    pl.col("column_0").str.starts_with("__").not_()
    ).sort("column_0")

### Multiple linear regression
X = MS(['lstat', 'age']).fit_transform(boston)
model1 = sm.OLS(Y, X)
results1 = model1.fit()
summarize(results1)

### X without the response 'medv'
terms = boston.columns.drop("medv")
X = MS(terms).fit_transform(boston)
model = sm.OLS(Y, X)
results = model.fit()
summarize(results)

### X without response and one of the predictors 'age'
minus_age = boston.columns.drop(['medv', 'age']) 
Xma = MS(minus_age).fit_transform(boston)
model1 = sm.OLS(Y, Xma)
summarize(model1.fit())

### VIFs
### matrix of X, but skip intercept column
vals = [VIF(X, i) for i in range(1, X.shape[1])]
vif = pd.DataFrame({'vif':vals},
                   index=X.columns[1:])
vif

### interaction terms
X = MS(['lstat',
        'age',
        ('lstat', 'age')]).fit_transform(boston)
model2 = sm.OLS(Y, X)
summarize(model2.fit())

### non-linear transformations of predictors
X = MS([poly('lstat', degree=2), 'age']).fit_transform(boston)
model3 = sm.OLS(Y, X)
results3 = model3.fit()
summarize(results3)

boston.shape
anova_lm(results1, results3)

### plot residuals
ax = plt.subplots(figsize=(8,8))[1]
ax.scatter(results3.fittedvalues, results3.resid)

ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--')

### Qualitative predictors
Carseats = load_data('Carseats')
Carseats.columns

# fit a multiple regression model w/interactions, qualititative
allvars = list(Carseats.columns.drop('Sales'))
y = Carseats['Sales']
final = allvars + [('Income', 'Advertising'),
                   ('Price', 'Age')]
X = MS(final).fit_transform(Carseats)
model = sm.OLS(y, X)
summarize(model.fit())


# %% ##### Logistic Regression, LDA, QDA, KNN
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
                         summarize)

from ISLP import confusion_table
from ISLP.models import contrast
from sklearn.discriminant_analysis import \
     (LinearDiscriminantAnalysis as LDA,
      QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

Smarket = load_data('Smarket')
Smarket
Smarket.columns
Smarket.corr(numeric_only=True)

sns.lineplot(Smarket, x="Year",y="Volume", 
             errorbar=('ci',False))
g = sns.lineplot(Smarket, x=np.arange(Smarket.shape[0]),y="Volume", 
             errorbar=('ci',False))
plt.legend(['Volume'])
Smarket.plot(y="Volume")

## Logistic regression
allvars = Smarket.columns.drop(['Today', 'Direction', 'Year'])
design = MS(allvars)
X = design.fit_transform(Smarket)
y = Smarket.Direction == 'Up'
glm = sm.GLM(y,
             X,
             family=sm.families.Binomial())
results = glm.fit()
summarize(results)
## alternative using sm.Logit
summarize(sm.Logit(y,X).fit())

# parameters, pvalues
results.params
results.pvalues
pl.DataFrame(dir(results)).filter(
    pl.all().str.starts_with('_').not_()
)

# predictions/confusion matrix
probs = results.predict()
probs[:10]
labels = np.array(['Down']*1250)
labels[probs>0.5] = "Up"
labels
confusion_table(labels, Smarket.Direction)
sklm.confusion_matrix(
    y_true=Smarket.Direction, 
    y_pred=labels,
    labels=["Down","Up"])

# training error rate on whole data
np.mean(labels == Smarket.Direction)

### test error rate
train = (Smarket.Year < 2005)
Smarket_train = Smarket.loc[train]
Smarket_test = Smarket.loc[~train]
Smarket_test.shape

# train test split
# note y train/tests vectors are True/False
train
X_train, X_test = X.loc[train], X.loc[~train]
y_train, y_test = y.loc[train], y.loc[~train]
glm_train = sm.GLM(y_train,
                   X_train,
                   family=sm.families.Binomial())
results = glm_train.fit()
probs = results.predict(exog=X_test)

# get labels of predictions on test y data
D = Smarket.Direction
L_train, L_test = D.loc[train], D.loc[~train]
labels = np.array(['Down']*L_test.shape[0])
labels[probs>0.5] = 'Up'
confusion_table(labels, L_test)
np.mean(labels == L_test), np.mean(labels != L_test)

## refitting with only few of the more significant
model = MS(['Lag1', 'Lag2']).fit(Smarket)
X = model.transform(Smarket)
X_train, X_test = X.loc[train], X.loc[~train]
glm_train = sm.GLM(y_train,
                   X_train,
                   family=sm.families.Binomial())
results = glm_train.fit()
probs = results.predict(exog=X_test)
labels = np.array(['Down']*L_test.shape[0])
labels[probs>0.5] = 'Up'
confusion_table(labels, L_test)
np.mean(labels == L_test), np.mean(labels != L_test)
# out of predicted labels == True, 
# how many are actually true?
( np.sum((labels == "Up") & (L_test == "Up")) / 
 np.sum(labels == "Up")
 )

# predict on new data
newdata = pd.DataFrame({'Lag1':[1.2, 1.5],
                        'Lag2':[1.1, -0.8]});
newX = model.transform(newdata)
newX
results.predict(newX)

# %%    ##### LDA #####
lda = LDA(store_covariance=True)
lda_2 = skl_da.LinearDiscriminantAnalysis()
lda_2

X_train, X_test = [M.drop(columns=['intercept'])
                   for M in [X_train, X_test]]
lda.fit(X_train, L_train)
lda.means_
lda.classes_
# prior(down) = 0.492, prior(up) = 0.508
lda.priors_
np.mean(L_train == "Up")

# linear discriminant vectors
lda.scalings_
X_train
np.dot(X_train, lda.scalings_)
lda_pred = lda.predict(X_test)
lda_pred
confusion_table(lda_pred, L_test)

# estimate probability /apply threshold
lda_prob = lda.predict_proba(X_test)
lda_prob
np.all(
       np.where(lda_prob[:,1] >= 0.5, 'Up','Down') == lda_pred
       )
# for more than two classes --> assign to highest class probability
np.all(
       [lda.classes_[i] for i in np.argmax(lda_prob, 1)] == lda_pred
       )

# %%    ##### QDA #####
qda = QDA(store_covariance=True)
qda.fit(X_train, L_train)
qda.means_, qda.priors_
# validate qda.means
np.mean(X_train.loc[L_train == "Up", "Lag1"])
np.mean(X_train.loc[L_train == "Up", "Lag2"])
np.mean(X_train.loc[L_train == "Down", "Lag1"])
np.mean(X_train.loc[L_train == "Down", "Lag2"])

# QDA estimates one covariance matrix per class
qda.covariance_
# covariance matrix estimate for first class ("Down"?)
qda.covariance_[0]

# prediction/confusion matrix
qda_pred = qda.predict(X_test)
confusion_table(qda_pred, L_test)
np.mean(qda_pred == L_test)


# %%     ##### Naive Bayes #####
        ### within the kth class, the p predictors are independent ###
NB = GaussianNB()
NB.fit(X_train, L_train)

# classes and prior probabilities
NB.classes_
NB.class_prior_

# parameters of the features
# rows are classes, cols are features
NB.theta_
NB.var_

# verify the parameters manually
X_train[L_train == 'Down'].mean()
X_train[L_train == 'Down'].var(ddof=0)

# making predictions
nb_labels = NB.predict(X_test)
confusion_table(nb_labels, L_test)
NB.predict_proba(X_test)[:5]

# %%    #### KNN #####
knn1 = KNeighborsClassifier(n_neighbors=1)
X_train, X_test = [np.asarray(X) for X in [X_train, X_test]]
knn1.fit(X_train, L_train)
knn1_pred = knn1.predict(X_test)
confusion_table(knn1_pred, L_test)
np.mean(knn1_pred == L_test)

# distribution of responses
Caravan = load_data('Caravan')
Purchase = Caravan.Purchase
Purchase.value_counts() / Purchase.size

# pre-processing, centering/scaling
feature_df = Caravan.drop(columns=['Purchase'])
scaler = StandardScaler(with_mean=True,
                        with_std=True,
                        copy=True)
scaler.fit(feature_df)
X_std = scaler.transform(feature_df)
X_std
feature_std = pd.DataFrame(
                 X_std,
                 columns=feature_df.columns)
feature_std
feature_std.std()

# train-test split
(X_train, X_test, y_train, y_test) = (
    train_test_split(
        feature_std,
        Purchase,
        test_size=1000,
        random_state=0)
)
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1_pred = knn1.fit(X_train, y_train).predict(X_test)
np.mean(y_test != knn1_pred), np.mean(y_test != "No")
confusion_table(knn1_pred, y_test)

# tuning hyper parameter 'k'
# care about % of predicted = Yes actually being Yes in this scenario
for K in range(1,6):
    knn = KNeighborsClassifier(n_neighbors=K)
    knn_pred = knn.fit(X_train, y_train).predict(X_test)
    C = confusion_table(knn_pred, y_test)
    templ = ('K={0:d}: # predicted to rent: {1:>2},' +
            '  # who did rent {2:d}, accuracy {3:.1%}')
    pred = C.loc['Yes'].sum()
    did_rent = C.loc['Yes','Yes']
    print(templ.format(
          K,
          pred,
          did_rent,
          did_rent / pred))

# %%    ##### Logistic Regression - sklearn #####

# set C argument very high to mimic no regularization
# solver='liblinear' suppresses a non-converging algorithm message
logit = LogisticRegression(C=1e10, solver='liblinear')
logit.fit(X_train, y_train)
logit_pred = logit.predict_proba(X_test)

# TRY DIFFERENT THRESHOLDS
logit_labels = np.where(logit_pred[:,1] > .5, 'Yes', 'No')
confusion_table(logit_labels, y_test)
logit_labels = np.where(logit_pred[:,1]>0.25, 'Yes', 'No')
confusion_table(logit_labels, y_test)

# %%    ##### Linear and Poisson regression #####
Bike = load_data('Bikeshare')
Bike.shape, Bike.columns

### Linear regression
X = MS(['mnth',
        'hr',
        'workingday',
        'temp',
        'weathersit']).fit_transform(Bike)
Y = Bike['bikers']
M_lm = sm.OLS(Y, X).fit()
summarize(M_lm)

## different encoding for some variables -> same results/model,
## slightly different interpretation
hr_encode = contrast('hr', 'sum')
mnth_encode = contrast('mnth', 'sum')
X2 = MS([mnth_encode,
         hr_encode,
        'workingday',
        'temp',
        'weathersit']).fit_transform(Bike)
X2
M2_lm = sm.OLS(Y, X2).fit()
S2 = summarize(M2_lm)
S2
np.sum((M_lm.fittedvalues - M2_lm.fittedvalues)**2)
np.allclose(M_lm.fittedvalues, M2_lm.fittedvalues)

# extra coefficients relating to mnth
coef_month = S2[S2.index.str.contains('mnth')]['coef']
coef_month
months = Bike['mnth'].dtype.categories
coef_month = pd.concat([
                       coef_month,
                       pd.Series([-coef_month.sum()],
                                  index=['mnth[Dec]'
                                 ])
                       ])
coef_month

# plot coefficients for mnth
fig_month, ax_month = subplots(figsize=(8,8))
x_month = np.arange(coef_month.shape[0])
ax_month.plot(x_month, coef_month, marker='o', ms=10)
ax_month.set_xticks(x_month)
ax_month.set_xticklabels([l[5] for l in coef_month.index], fontsize=20)
ax_month.set_xlabel('Month', fontsize=20)
ax_month.set_ylabel('Coefficient', fontsize=20)

# do same for hour
coef_hr = S2[S2.index.str.contains('hr')]['coef']
coef_hr = coef_hr.reindex(['hr[{0}]'.format(h) for h in range(23)])
coef_hr = pd.concat([coef_hr,
                     pd.Series([-coef_hr.sum()], index=['hr[23]'])
                    ])
fig_hr, ax_hr = subplots(figsize=(8,8))
x_hr = np.arange(coef_hr.shape[0])
ax_hr.plot(x_hr, coef_hr, marker='o', ms=10)
ax_hr.set_xticks(x_hr[::2])
ax_hr.set_xticklabels(range(24)[::2], fontsize=20)
ax_hr.set_xlabel('Hour', fontsize=20)
ax_hr.set_ylabel('Coefficient', fontsize=20)

# %%    ##### Poison Regression #####
M_pois = sm.GLM(Y, X2, family=sm.families.Poisson()).fit()
S_pois = summarize(M_pois)
coef_month = S_pois[S_pois.index.str.contains('mnth')]['coef']
coef_month = pd.concat([coef_month,
                        pd.Series([-coef_month.sum()],
                                   index=['mnth[Dec]'])])
coef_hr = S_pois[S_pois.index.str.contains('hr')]['coef']
coef_hr = pd.concat([coef_hr,
                     pd.Series([-coef_hr.sum()],
                     index=['hr[23]'])])

# plot coefficients for month hr for Poisson regression
fig_pois, (ax_month, ax_hr) = subplots(1, 2, figsize=(16,8))
ax_month.plot(x_month, coef_month, marker='o', ms=10)
ax_month.set_xticks(x_month)
ax_month.set_xticklabels([l[5] for l in coef_month.index], fontsize=20)
ax_month.set_xlabel('Month', fontsize=20)
ax_month.set_ylabel('Coefficient', fontsize=20)
ax_hr.plot(x_hr, coef_hr, marker='o', ms=10)
ax_hr.set_xticklabels(range(24)[::2], fontsize=20)
ax_hr.set_xlabel('Hour', fontsize=20)
ax_hr.set_ylabel('Coefficient', fontsize=20)

# compare fitted values for linear vs Poisson regression
fig, ax = subplots(figsize=(8, 8))
ax.scatter(M2_lm.fittedvalues,
           M_pois.fittedvalues,
           s=20)
ax.set_xlabel('Linear Regression Fit', fontsize=20)
ax.set_ylabel('Poisson Regression Fit', fontsize=20)
ax.axline([0,0], c='black', linewidth=3,
          linestyle='--', slope=1)


# %%    ##### Cross-Validation and the Bootstrap #####

# Load data and split into train/test
Auto = load_data('Auto')
Auto_train, Auto_valid = train_test_split(Auto,
                                         test_size=196,
                                         random_state=0)

# make and fit model to training
hp_mm = MS(['horsepower'])
X_train = hp_mm.fit_transform(Auto_train)
y_train = Auto_train['mpg']
model = sm.OLS(y_train, X_train)
results = model.fit()

# fit on test data and get MSE
X_valid = hp_mm.transform(Auto_valid)
y_valid = Auto_valid['mpg']
valid_pred = results.predict(X_valid)
np.mean((y_valid - valid_pred)**2)

# function for determining MSE
def evalMSE(terms,
            response,
            train,
            test):

   mm = MS(terms)
   X_train = mm.fit_transform(train)
   y_train = train[response]

   X_test = mm.transform(test)
   y_test = test[response]

   results = sm.OLS(y_train, X_train).fit()
   test_pred = results.predict(X_test)

   return np.mean((y_test - test_pred)**2)

# try linear, quad, cubic fits
MSE = np.zeros(3)
for idx, degree in enumerate(range(1, 4)):
    MSE[idx] = evalMSE([poly('horsepower', degree)],
                       'mpg',
                       Auto_train,
                       Auto_valid)
MSE

# retry on different test set
Auto_train, Auto_valid = train_test_split(Auto,
                                          test_size=196,
                                          random_state=3)
MSE = np.zeros(3)
for idx, degree in enumerate(range(1, 4)):
    MSE[idx] = evalMSE([poly('horsepower', degree)],
                       'mpg',
                       Auto_train,
                       Auto_valid)
MSE

### Cross-validation

# wrapper function from ISLP, also accepts a
# model_args = {..} parameter for additional params
hp_model = sklearn_sm(sm.OLS,
                      MS(['horsepower']))
X, Y = Auto.drop(columns=['mpg']), Auto['mpg']
# fit a LOOCV
cv_results = cross_validate(hp_model,
                            X,
                            Y,
                            cv=Auto.shape[0])
cv_err = np.mean(cv_results['test_score'])
cv_err

# fit for polynomial degrees 1 to 5
cv_error = np.zeros(5)
H = np.array(Auto['horsepower'])
M = sklearn_sm(sm.OLS)
for i, d in enumerate(range(1,6)):
    X = np.power.outer(H, np.arange(d+1))
    M_CV = cross_validate(M,
                          X,
                          Y,
                          cv=Auto.shape[0])
    cv_error[i] = np.mean(M_CV['test_score'])
cv_error

# example of outer generating numpy arrays
A = np.array([3, 5, 9])
B = np.array([2, 4])
np.add.outer(A, B)

# cv using K folds, k = 10
cv_error = np.zeros(5)
cv = KFold(n_splits=10,
           shuffle=True,
           random_state=0) # use same splits for each degree
cv
for i, d in enumerate(range(1,6)):
    X = np.power.outer(H, np.arange(d+1))
    M_CV = cross_validate(M,
                          X,
                          Y,
                          cv=cv)
    cv_error[i] = np.mean(M_CV['test_score'])
cv_error

# cross_validate function is flexible and can take other split functions
# different to KFold since in each iteration of ShuffleSplit, the entire
# dataset is sampled for a test and training set, versus the fold method
# which guarantees all test sets in the K folds are mutually exclusive
validation = ShuffleSplit(n_splits=1,
                          test_size=196,
                          random_state=0)
results = cross_validate(hp_model,
                         Auto.drop(['mpg'], axis=1),
                         Auto['mpg'],
                         cv=validation)
results['test_score']

# estimate variability in test error
# note that it's not a valid estimate since 
# training/test sets overlap with ShuffleSplit method
validation = ShuffleSplit(n_splits=10,
                          test_size=196,
                          random_state=0)
results = cross_validate(hp_model,
                         Auto.drop(['mpg'], axis=1),
                         Auto['mpg'],
                         cv=validation)
results
results['test_score'].mean(), results['test_score'].std()


### Bootstrap
Portfolio = load_data('Portfolio')

# covariance of first 100 observations
def alpha_func(D, idx):
   cov_ = np.cov(D[['X','Y']].loc[idx], rowvar=False)
#    print(cov_)
   return ((cov_[1,1] - cov_[0,1]) /
           (cov_[0,0]+cov_[1,1]-2*cov_[0,1]))
alpha_func(Portfolio, range(100))

# covariance of bootstrapped 100 observations with replacement
rng = np.random.default_rng(0)
alpha_func(Portfolio,
           rng.choice(100,
                      100,
                      replace=True))

# generic function to computing the bootstrap standard error for 
# arbitrary functions that take A WHOLE DATAFRAME as an argument.
def boot_SE(func,
            D,
            n=None,
            B=1000,
            seed=0):
    rng = np.random.default_rng(seed)
    first_, second_ = 0, 0
    # how many to sample out of the data index per bootstrap sample
    n = n or D.shape[0]
    for _ in range(B):
        # get index values corresponding to bootstrap sample
        idx = rng.choice(D.index,
                         n,
                         replace=True)
        # calculate function metric based on data, index values
        value = func(D, idx)
        # sum of X and X^2
        first_ += value
        second_ += value**2
    # E(X^2) - E(X)^2 --> variance, sqrt to get se
    return np.sqrt(second_ / B - (first_ / B)**2)

alpha_SE = boot_SE(alpha_func,
                   Portfolio,
                   B=1000,
                   seed=0)
# SE(alpha_hat)
alpha_SE


### Estimating accuracy of Linear Regression Model

# define function to get parameters
def boot_OLS(model_matrix, response, D, idx):
    D_ = D.loc[idx]
    Y_ = D_[response]
    X_ = clone(model_matrix).fit_transform(D_)
    return sm.OLS(Y_, X_).fit().params

# partially define the function boot_OLS
hp_func = partial(boot_OLS, MS(['horsepower']), 'mpg')
hp_func
Auto
# test on a few bootstrap samples
rng = np.random.default_rng(0)
np.array([hp_func(Auto,
          rng.choice(Auto.index,
                     392,
                     replace=True)) for _ in range(10)])

# apply the hp_func 'partialised' func as arg to boot_SE
hp_se = boot_SE(hp_func,
                Auto,
                B=1000,
                seed=10)
# SE(B0) and SE(B1)
hp_se

# compare bootstrap se estimates versus model se estimates
# note bootstrap may be more accurate as it does not assume that
# xi's are fixed unlike the formulas to compute se in the model,
# so may provide a better estimate of sigma^2
hp_model.fit(Auto, Auto['mpg'])
model_se = summarize(hp_model.results_)['std err']
model_se


## quadratic fit boot SE and model SE for parameters
quad_model = MS([poly('horsepower', 2, raw=True)])
quad_func = partial(boot_OLS,
                    quad_model,
                    'mpg')
boot_SE(quad_func, Auto, B=1000)

M = sm.OLS(Auto['mpg'],
           quad_model.fit_transform(Auto))
summarize(M.fit())['std err']





# %%    ##### Linear Models and Regularization Methods #####

# Forward selection
Hitters = load_data('Hitters')
np.isnan(Hitters['Salary']).sum()
Hitters = Hitters.dropna()
Hitters.shape

# use Cp as a scorer.
# by default, sklearn tries to maximize a scorer, so
# compute the negative Cp statistic
def nCp(sigma2, estimator, X, Y):
    "Negative Cp statistic"
    n, p = X.shape
    Yhat = estimator.predict(X)
    RSS = np.sum((Y - Yhat)**2)
    return -(RSS + 2 * p * sigma2) / n

# fit model using all variables, to estimate sigma^2
design = MS(Hitters.columns.drop('Salary')).fit(Hitters)
Y = np.array(Hitters['Salary'])
X = design.transform(Hitters)
sigma2 = sm.OLS(Y,X).fit().scale

# sklearn_selected() requires only the estimator, X, Y
# variables of nCp function, so define partial function, this
# is then used as a scorer for model selection
neg_Cp = partial(nCp, sigma2)
# also need to specify a search strategy, the one below is defined
# in ISLP.models package, which stops adding to model once there is 
# no improvement in the scoring function provided
strategy = Stepwise.first_peak(design,
                               direction='forward',
                               max_terms=len(design.terms))

strategy

## now fit linear regression using forward selection, which takes
# a model and a search strategy, if no score is provided, then default MSE
hitters_MSE = sklearn_selected(sm.OLS,
                               strategy)
hitters_MSE.fit(Hitters, Y)
hitters_MSE.selected_state_

# only select 10 variables with using scoring=neg_CP
hitters_Cp = sklearn_selected(sm.OLS,
                               strategy,
                               scoring=neg_Cp)
hitters_Cp.fit(Hitters, Y)
hitters_Cp.selected_state_

#### alternative model selection than Cp: CV/Validation
strategy = Stepwise.fixed_steps(design,
                                len(design.terms),
                                direction='forward')
full_path = sklearn_selection_path(sm.OLS, strategy)
full_path.fit(Hitters, Y)
Yhat_in = full_path.predict(Hitters)
# training dataset is predicted on each observation (each row) 
# and also each model (the 20 columns)
Yhat_in.shape

## plot training MSE for each model with increasing # predictors
mse_fig, ax = plt.subplots(figsize=(8,8))
# need to get the shape into (263, 1)
insample_mse = ((Yhat_in - Y[:,None])**2).mean(0)
n_steps = insample_mse.shape[0]
ax.plot(np.arange(n_steps),
        insample_mse,
        'k', # color black
        label='In-sample')
ax.set_ylabel('MSE',
              fontsize=20)
ax.set_xlabel('# steps of forward stepwise',
              fontsize=20)
ax.set_xticks(np.arange(n_steps)[::2])
ax.legend()
ax.set_ylim([50000,250000])

##### calculate cross-validated predicted values using 5-fold CV
K = 5
kfold = skm.KFold(K,
                  random_state=0,
                  shuffle=True)
# each prediction on a row belongs to a 'hold-out' fold and is 
# predicted from a model trained on one training fold. 
# this is done per each of the 20 models, giving (n, n_models) shape
Yhat_cv = skm.cross_val_predict(full_path,
                                Hitters,
                                Y,
                                cv=kfold)
Yhat_cv.shape

## evaluate CV error per fold, per each of the 20 models, and plot
cv_mse = []
# kfold.split(Y) --> length of 5, each correspond to a train/hold out fold tuple indices
for train_idx, test_idx in kfold.split(Y):
    # again, require shape of (263, 1) (to match the (263, 20))
    errors = (Yhat_cv[test_idx] - Y[test_idx,None])**2
    cv_mse.append(errors.mean(0)) # column means
cv_mse = np.array(cv_mse).T
# one per model (with increasing # predictors) per fold
cv_mse.shape

ax.errorbar(np.arange(n_steps), 
            cv_mse.mean(1),
            cv_mse.std(1) / np.sqrt(K),
            label='Cross-validated',
            c='r') # color red
ax.set_ylim([50000,250000])
ax.legend()
mse_fig

### Implement validation set approach over cv approach, and plot
validation = skm.ShuffleSplit(n_splits=1, 
                              test_size=0.2,
                              random_state=0)
# list(validation.split(Y))
for train_idx, test_idx in validation.split(Y):
    full_path.fit(Hitters.iloc[train_idx],
                  Y[train_idx])
    Yhat_val = full_path.predict(Hitters.iloc[test_idx])
    errors = (Yhat_val - Y[test_idx,None])**2
    validation_mse = errors.mean(0)
    
ax.plot(np.arange(n_steps), 
    validation_mse,
    'b--', # color blue, broken line
    label='Validation')
ax.set_xticks(np.arange(n_steps)[::2])
ax.set_ylim([50000,250000])
ax.legend()
mse_fig




# %%    ########## RIDGE AND LASSO REGRESSION ############

X = design.transform(Hitters)
X
X = X.drop(columns=["intercept"])
# Xs = X - X.mean(0)[None,:]
Xs = X - X.mean(0)
X_scale = X.std(0).replace(0, np.nan)
# Xs = Xs / X_scale[None,:]
Xs = Xs.div(X_scale).fillna(0)
Xs
lambdas = 10**np.linspace(8, -2, 100) / Y.std()
lambdas
# ridge correspond to l1_ratio = 0
soln_array = skl.ElasticNet.path(Xs,
                                 Y,
                                 l1_ratio=0.,
                                 alphas=lambdas)[1]
soln_array.shape
soln_array
soln_array.T

# plot coefficients as lambda changes
soln_path = pd.DataFrame(soln_array.T,
                         columns=Xs.columns,
                         index=-np.log(lambdas))
soln_path.index.name = 'negative log(lambda)'
soln_path

path_fig, ax = plt.subplots(figsize=(8,8))
soln_path.plot(ax=ax, legend=False)
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20)
ax.legend(loc='upper left')

# 40th lambda, coefficients at 40th lambda
beta_hat = soln_path.loc[soln_path.index[39]] # soln_path.iloc[39]
lambdas[39], beta_hat
# l2 norm
np.linalg.norm(beta_hat)
# l2 norm of a much smaller lambda = higher l2 norm
beta_hat = soln_path.loc[soln_path.index[59]]
# np.linalg.norm(soln_path.iloc[59])
lambdas[59], np.linalg.norm(beta_hat)

# Above we normalized X upfront, and fit the ridge model using Xs. 
# The Pipeline() object in sklearn provides a clear way to separate feature 
# normalization from the fitting of the ridge model itself.
ridge = skl.ElasticNet(alpha=lambdas[59], l1_ratio=0)
scaler = StandardScaler(with_mean=True,  with_std=True)
pipe = Pipeline(steps=[('scaler', scaler), ('ridge', ridge)])
pipe.fit(X, Y)

# very close to the one provided by ElasticNet.path
np.linalg.norm(ridge.coef_)




##### Estimating test error of Ridge

# fix random state of splitter to ensure reproducability
validation = skm.ShuffleSplit(n_splits=1,
                              test_size=0.5,
                              random_state=0)

# test MSE with alpha = 0.01
ridge.alpha = 0.01
results = skm.cross_validate(ridge,
                             X,
                             Y,
                             scoring='neg_mean_squared_error',
                             cv=validation)
-results['test_score']

# test MSE with alpha = 1e10
ridge.alpha = 1e10
results = skm.cross_validate(ridge,
                             X,
                             Y,
                             scoring='neg_mean_squared_error',
                             cv=validation)
-results['test_score']

## using validation set approach to choose lambda
param_grid = {'ridge__alpha': lambdas}
grid = skm.GridSearchCV(pipe,
                        param_grid,
                        cv=validation,
                        scoring='neg_mean_squared_error')
grid.fit(X, Y)
grid.best_params_['ridge__alpha']
grid.best_estimator_

### using 5-fold CV approach, NOTE definition of kfold
kfold = skm.KFold(K,
                  random_state=0,
                  shuffle=True)
grid = skm.GridSearchCV(pipe, 
                        param_grid,
                        cv=kfold,
                        scoring='neg_mean_squared_error')
grid.fit(X, Y)
grid.best_params_['ridge__alpha']
grid.best_estimator_

# plot CV-MSE of lambdas
ridge_fig, ax = plt.subplots(figsize=(8,8))
# -log(lambda) means graph starts from very high lambda values to low
# meaning that starts from high bias low var to low bias high var
ax.errorbar(-np.log(lambdas),
            -grid.cv_results_['mean_test_score'],
            yerr=grid.cv_results_['std_test_score'] / np.sqrt(K))
ax.set_ylim([50000,250000])
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Cross-validated MSE', fontsize=20)

# instead of using 'neg_mean_squared_error' as score, use the default
# of R^2
grid_r2 = skm.GridSearchCV(pipe, 
                           param_grid,
                           cv=kfold)
grid_r2.fit(X, Y)
# 
grid_r2.cv_results_['mean_test_score']
grid_r2.cv_results_['std_test_score']

r2_fig, ax = plt.subplots(figsize=(8,8))
ax.errorbar(-np.log(lambdas),
            grid_r2.cv_results_['mean_test_score'],
            # https://www.mdpi.com/2571-905X/4/4/51
            # this is the standard error in the K-folds MSE
            yerr=grid_r2.cv_results_['std_test_score'] / np.sqrt(K))
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Cross-validated $R^2$', fontsize=20)



##### Fast CV for Solution Paths
# here the normalization done once across data rather than per fold, but 
# shouild be similar
ridgeCV = skl.ElasticNetCV(alphas=lambdas, 
                           l1_ratio=0,
                           cv=kfold)
pipeCV = Pipeline(steps=[('scaler', scaler),
                         ('ridge', ridgeCV)])
pipeCV.fit(X, Y)

### see CV errors by lambda values
tuned_ridge = pipeCV.named_steps['ridge']
# two different ways to get min lambda
-np.log(tuned_ridge.alpha_)
-np.log(lambdas[np.argmin(tuned_ridge.mse_path_.mean(axis=1))])
ridgeCV_fig, ax = plt.subplots(figsize=(8,8))
ax.errorbar(-np.log(lambdas),
            tuned_ridge.mse_path_.mean(axis=1),
            yerr=tuned_ridge.mse_path_.std(axis=1) / np.sqrt(K))
ax.axvline(-np.log(tuned_ridge.alpha_), c='k', ls='--')
ax.set_ylim([50000,250000])
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Cross-validated MSE', fontsize=20)

# each row is a lambda, columns are the folds
tuned_ridge.mse_path_.shape
# get mean sq-error across folds per each lambda, then take minimum MSE
np.min(tuned_ridge.mse_path_.mean(axis=1))
# coefficients of lowest lambda fit
tuned_ridge.coef_



######## However, we have used all the data in the CV-fitting of lambda
# instead, split into training/test fit, and the training is used for fitting
outer_valid = skm.ShuffleSplit(n_splits=1, 
                               test_size=0.25,
                               random_state=1)
inner_cv = skm.KFold(n_splits=5,
                     shuffle=True,
                     random_state=2)
ridgeCV = skl.ElasticNetCV(alphas=lambdas,
                           l1_ratio=0,
                           cv=inner_cv)
pipeCV = Pipeline(steps=[('scaler', scaler),
                         ('ridge', ridgeCV)])
# split, fit and evaluate on test error
results = skm.cross_validate(pipeCV, 
                             X,
                             Y,
                             cv=outer_valid,
                             scoring='neg_mean_squared_error')
-results['test_score']



########## LASSO Regression
# note l1_ratio = 1
lassoCV = skl.ElasticNetCV(n_alphas=100, 
                           l1_ratio=1,
                           cv=kfold)
pipeCV = Pipeline(steps=[('scaler', scaler),
                         ('lasso', lassoCV)])
pipeCV.fit(X, Y)
tuned_lasso = pipeCV.named_steps['lasso']
tuned_lasso.alpha_

# EQUIVALENT TO skl.ElasticNet.path( ...)[:2]
lambdas, soln_array = skl.Lasso.path(Xs, 
                                    Y,
                                    l1_ratio=1,
                                    n_alphas=100)[:2]

soln_path = pd.DataFrame(soln_array.T,
                         columns=Xs.columns,
                         index=-np.log(lambdas))

# plot paths of coefficients as lambda changes
path_fig, ax = plt.subplots(figsize=(8,8))
soln_path.plot(ax=ax, legend=False)
ax.legend(loc='upper left')
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficiients', fontsize=20)

# lowest CV-MSE
np.min(tuned_lasso.mse_path_.mean(1))

# plot of CV-error by lambdas
lassoCV_fig, ax = plt.subplots(figsize=(8,8))
ax.errorbar(-np.log(tuned_lasso.alphas_),
            tuned_lasso.mse_path_.mean(1),
            yerr=tuned_lasso.mse_path_.std(1) / np.sqrt(K))
ax.axvline(-np.log(tuned_lasso.alpha_), c='k', ls='--')
ax.set_ylim([50000,250000])
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Cross-validated MSE', fontsize=20)

# coefficients corresponding to lambda with best CV-MSE
tuned_lasso.coef_




# %%    ########## PCR and PLS regression #############
# LinearRegression() fits an intercept by default unlike OLS
pca = PCA(n_components=2)
linreg = skl.LinearRegression()
pipe = Pipeline([('pca', pca),
                 ('linreg', linreg)])
pipe.fit(X, Y)
pipe.named_steps['linreg'].coef_

# PCA with standardization of variables - recommended
pipe = Pipeline([('scaler', scaler), 
                 ('pca', pca),
                 ('linreg', linreg)])
pipe.fit(X, Y)
pipe.named_steps['linreg'].coef_

# use CV to choose # components
# note naming of 'pca__n_components' as must adhere to the format:
# step_name__parameter_name in the pipeline
param_grid = {'pca__n_components': range(1, 20)}
grid = skm.GridSearchCV(pipe,
                        param_grid,
                        cv=kfold,
                        scoring='neg_mean_squared_error')
grid.fit(X, Y)

# plot PCA components against CV-error
pcr_fig, ax = plt.subplots(figsize=(8,8))
n_comp = param_grid['pca__n_components']
ax.errorbar(n_comp,
            -grid.cv_results_['mean_test_score'],
            grid.cv_results_['std_test_score'] / np.sqrt(K))
ax.set_ylabel('Cross-validated MSE', fontsize=20)
ax.set_xlabel('# principal components', fontsize=20)
ax.set_xticks(n_comp[::2])
# ax.set_ylim([50000,250000])

# PCA() method complains if n_components = 0, so fit
# intercept model manually
Xn = np.zeros((X.shape[0], 1))
cv_null = skm.cross_validate(linreg,
                             Xn,
                             Y,
                             cv=kfold,
                             scoring='neg_mean_squared_error')
# MSE per fold, then take mean across the MSE of all the folds
-cv_null['test_score'].mean()
# % of variance explained by each PCA component
pipe.named_steps['pca'].explained_variance_ratio_


##### Partial Least Squares (PLS)
pls = PLSRegression(n_components=2, 
                    scale=True)
pls.fit(X, Y)

# Use CV to choose # components
param_grid = {'n_components':range(1, 20)}
grid = skm.GridSearchCV(pls,
                        param_grid,
                        cv=kfold,
                        scoring='neg_mean_squared_error')
grid.fit(X, Y)

# Plot CV-MSE by # components
pls_fig, ax = plt.subplots(figsize=(8,8))
n_comp = param_grid['n_components']
ax.errorbar(n_comp,
            -grid.cv_results_['mean_test_score'],
            grid.cv_results_['std_test_score'] / np.sqrt(K))
ax.set_ylabel('Cross-validated MSE', fontsize=20)
ax.set_xlabel('# principal components', fontsize=20)
ax.set_xticks(n_comp[::2])
# ax.set_ylim([50000,250000])






# %%    ################ NON-LINEAR MODELLING ##################

# Load data
Wage = load_data('Wage')
y = Wage['wage']
age = Wage['age']

###### Polynomial Regression / Step Functions
poly_age = MS([poly('age', degree=4)]).fit(Wage)
M = sm.OLS(y, poly_age.transform(Wage)).fit()
summarize(M)

# create grid of ages for which we want predictions
age_grid = np.linspace(age.min(),
                       age.max(),
                       100)
age_df = pd.DataFrame({'age': age_grid})

# define function to visualize several model specs
def plot_wage_fit(age_df, 
                  basis,
                  title):

    X = basis.transform(Wage)
    Xnew = basis.transform(age_df)
    M = sm.OLS(y, X).fit()
    preds = M.get_prediction(Xnew)
    bands = preds.conf_int(alpha=0.05)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(age,
               y,
               facecolor='gray',
               alpha=0.5)
    for val, ls in zip([preds.predicted_mean,
                      bands[:,0],
                      bands[:,1]],
                     ['b','r--','r--']):
        ax.plot(age_df.values, val, ls, linewidth=3)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel('Age', fontsize=20)
    ax.set_ylabel('Wage', fontsize=20)
    return ax

plot_wage_fit(age_df, 
              poly_age,
              'Degree-4 Polynomial')

# selection of polynomial degree based on ANOVA
# NOTE that ANOVA works for non-orthogonal polynomials
# summarize(M) also works here since orthogonal poly's
models = [MS([poly('age', degree=d)]) 
          for d in range(1, 6)]
Xs = [model.fit_transform(Wage) for model in models]
anova_lm(*[sm.OLS(y, X_).fit()
           for X_ in Xs])
summarize(M)

models = [MS(['education', poly('age', degree=d)])
          for d in range(1, 4)]
XEs = [model.fit_transform(Wage)
       for model in models]
anova_lm(*[sm.OLS(y, X_).fit() for X_ in XEs])
# alternative is using CV to choose degree of polynomial ...


### predicting whether individual earns > $250k/yr
X = poly_age.transform(Wage)
high_earn = Wage['high_earn'] = y > 250 # shorthand
glm = sm.GLM(y > 250,
             X,
             family=sm.families.Binomial())
B = glm.fit()
summarize(B)

newX = poly_age.transform(age_df)
newX
age_df.iloc[60]
B.predict(newX, which="linear").iloc[60]
preds = B.get_prediction(newX)
bands = preds.conf_int(alpha=0.05)

# plot estimated relationship
fig, ax = plt.subplots(figsize=(8,8))
rng = np.random.default_rng(0)
ax.scatter(age +
           0.2 * rng.uniform(size=y.shape[0]),
           np.where(high_earn, 0.198, 0.002),
           fc='gray',
           marker='|')
for val, ls in zip([preds.predicted_mean,
                  bands[:,0],
                  bands[:,1]],
                 ['b','r--','r--']):
    ax.plot(age_df.values, val, ls, linewidth=3)
ax.set_title('Degree-4 Polynomial', fontsize=20)
ax.set_xlabel('Age', fontsize=20)
ax.set_ylim([0,0.2])
ax.set_ylabel('P(Wage > 250)', fontsize=20)

# replicate calculation of upper bound
se = np.sqrt(
    np.diag(
        np.dot(newX, 
               np.dot(B.cov_params(), 
                      newX.T))))
quantile_975 = stats.norm.ppf(0.975)
upper_bound = B.predict(newX, which="linear") + quantile_975* se
upper_bound_prob = 1 / (1 + np.exp(-upper_bound))
upper_bound_prob
bands[:,1]

### fit step function on categorical
# no perfect collinearity here even with all 4, since
# there is no intercept term in model?
cut_age = pd.qcut(age, 4)
cut_age
pd.get_dummies(cut_age)
summarize(sm.OLS(y, pd.get_dummies(cut_age)).fit())

#### Splines
# actual spine evaluation functions in scipy.interpolate,
# currently wrapped them as transforms in ISLP

# BSpline generates entire matrix of basis functions,
# default cubic splines, use degree= to change
# number of parameters expected = #knots + degree + 1
bs_ = BSpline(internal_knots=[25,40,60], intercept=True).fit(age)
bs_age = bs_.transform(age)
bs_age
bs_age.shape

# use name= to change output name of variables
bs_age = MS([bs('age', internal_knots=[25,40,60]
                , name='bs(age)')])
Xbs = bs_age.fit_transform(Wage)
M = sm.OLS(y, Xbs).fit()
summarize(M)

# df=6 here correspond to 3 knots,
# this function chooses knots at uniform quantiles
BSpline(df=6).fit(age).internal_knots_

# degree=0 --> piecewise constants,
# df=3 when degree=0 --> 3 knots
# VERY SIMILAR to qcut results above, but different
# inequality logic for bins lead to slightly different results
bs_age0 = MS([bs('age',
                 df=3, 
                 degree=0)]).fit(Wage)
Xbs0 = bs_age0.transform(Wage)
summarize(sm.OLS(y, Xbs0).fit())

### fit natural spline with 5 df and no intercept
# difference between B-splines (efficient regression splines)
# versus Natural splines: Natural has linear boundary constraints
ns_age = MS([ns('age', df=5)]).fit(Wage)
M_ns = sm.OLS(y, ns_age.transform(Wage)).fit()
summarize(M_ns)

# plot natural spline fit
plot_wage_fit(age_df,
              ns_age,
              'Natural spline, df=5')


##### Smoothing Splines and GAMs
# A smoothing spline is a special case of a GAM with 
# squared-error loss and a single feature

#### Smoothing operations...
# s=smoothing spline
# l=linear
# f=factor/categorical

#.reshape((-1,1)) --> doesnt seem to be needed
# np.asarray(age) also doesn't seem to be needed
X_age = age
# 0 means smoother applies to first column of a
# feature matrix, lam is the penalty parameter for non-smoothness
gam = LinearGAM(s_gam(0, lam=0.6))
gam.fit(X_age, y)

# investigate how fit changes with lam
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(age, y, facecolor='gray', alpha=0.5)
for lam in np.logspace(-2, 6, 5):
    gam = LinearGAM(s_gam(0, lam=lam)).fit(X_age, y)
    ax.plot(age_grid,
            gam.predict(age_grid),
            label='{:.1e}'.format(lam),
            linewidth=3)
ax.set_xlabel('Age', fontsize=20)
ax.set_ylabel('Wage', fontsize=20)
ax.legend(title='$\lambda$')

# let pygam package/algorithm find optimal lam parameter
gam_opt = gam.gridsearch(X_age, y)
gam_opt.predict(age_grid)
gam_opt.lam
# pl.DataFrame(dir(gam_opt)).filter(pl.all().str.starts_with("_").not_())
ax.plot(age_grid,
        gam_opt.predict(age_grid),
        label='Grid search',
        linewidth=4)
ax.legend()
fig

### alternatively, fix df using ISLP.pygam
# below finds lam that gives us ~4 df, noting that
# the 4 df includes the 2 unpenalized intercept/linear terms
age_term = gam.terms[0]
# smooth term associated with age in your GAM.
age_term
# approx_lam is estimating a smoothing parameter (lam_4) 
# for this term.
lam_4 = approx_lam(np.asarray(X_age).reshape(-1,1), 
                   age_term, 4)
age_term.lam = lam_4
# calculate how many effective degrees of freedom are
# associated with this smooth term given the estimated smoothing parameter.
degrees_of_freedom(X_age, age_term)

### vary df similar to an above plot, choose df as desired
# df+1 to account for smoothing splines always having intercept term
# df=1 in the loop means we want a linear fit
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(X_age,
           y,
           facecolor='gray',
           alpha=0.3)
for df in [1,3,4,8,15]:
    lam = approx_lam(X_age, age_term, df+1)
    age_term.lam = lam
    gam.fit(X_age, y)
    ax.plot(age_grid,
            gam.predict(age_grid),
            label='{:d}'.format(df),
            linewidth=4)
ax.set_xlabel('Age', fontsize=20)
ax.set_ylabel('Wage', fontsize=20)
ax.legend(title='Degrees of freedom')



##### Additive Models with Several Terms

# Build model matrix in manual fashion, 
# lets us construct partial dependence plots more easily
ns_age = NaturalSpline(df=4).fit(age)
ns_year = NaturalSpline(df=5).fit(Wage['year'])
Xs = [ns_age.transform(age),
      ns_year.transform(Wage['year']),
      # note excludes intercept, includes all cat levels
      pd.get_dummies(Wage['education']).values]
X_bh = np.hstack(Xs)
gam_bh = sm.OLS(y, X_bh).fit()

### Partial dependence plot - Age vs Wage 
# create range of age values to predict
age_grid = np.linspace(age.min(),
                       age.max(),
                       100)
# first 100 rows of X_bh to match dim of age_grid
X_age_bh = X_bh.copy()[:100]
X_age_bh
# set each feature to mean of each feature
X_age_bh[:] = X_bh[:].mean(0)[None,:]
X_age_bh[:,:4]
X_age_bh.shape
# replace just the first four columns representing age 
# with the natural spline basis computed at the values in age_grid
X_age_bh[:,:4] = ns_age.transform(age_grid)
X_age_bh.shape
# plot
preds = gam_bh.get_prediction(X_age_bh)
bounds_age = preds.conf_int(alpha=0.05)
partial_age = preds.predicted_mean
center = partial_age.mean()
partial_age -= center
bounds_age -= center
fig, ax = plt.subplots(figsize=(8,8))
ax.plot(age_grid, partial_age, 'b', linewidth=3)
ax.plot(age_grid, bounds_age[:,0], 'r--', linewidth=3)
ax.plot(age_grid, bounds_age[:,1], 'r--', linewidth=3)
ax.set_xlabel('Age')
ax.set_ylabel('Effect on wage')
ax.set_title('Partial dependence of age on wage', fontsize=20)

### Partial dependence plot for year vs wage
year_grid = np.linspace(2003, 2009, 100)
year_grid = np.linspace(Wage['year'].min(),
                        Wage['year'].max(),
                        100)
X_year_bh = X_bh.copy()[:100]
X_year_bh[:] = X_bh[:].mean(0)[None,:]
X_year_bh[:,4:9] = ns_year.transform(year_grid)
preds = gam_bh.get_prediction(X_year_bh)
bounds_year = preds.conf_int(alpha=0.05)
partial_year = preds.predicted_mean
center = partial_year.mean()
partial_year -= center
bounds_year -= center
fig, ax = plt.subplots(figsize=(8,8))
ax.plot(year_grid, partial_year, 'b', linewidth=3)
ax.plot(year_grid, bounds_year[:,0], 'r--', linewidth=3)
ax.plot(year_grid, bounds_year[:,1], 'r--', linewidth=3)
ax.set_xlabel('Year')
ax.set_ylabel('Effect on wage')
ax.set_title('Partial dependence of year on wage', fontsize=20)



# fit model using smoothing splines over natural splines
# require matrices when using pygam, must use cat.codes below

# default lam=0.6 (arbitrary)
gam_full = LinearGAM(s_gam(0) +
                     # more n_splines = more flexibility but more variance
                     s_gam(1, n_splines=7) +
                     # lam=0 for no shrinkage
                     f_gam(2, lam=0))
Xgam = np.column_stack([age,
                        Wage['year'],
                        Wage['education'].cat.codes])
gam_full = gam_full.fit(Xgam, y)

# plot partial dependence of age on wage using lam=0.6
fig, ax = plt.subplots(figsize=(8,8))
# a function is now defined in contrasts to previously verbose section
plot_gam(gam_full, 0, ax=ax)
ax.set_xlabel('Age')
ax.set_ylabel('Effect on wage')
ax.set_title('Partial dependence of age on wage - default lam=0.6', fontsize=20)

# perhaps more natural to specify df (to get lam) over lam:
# refit using 4 df (+1 is for intercept)
age_term = gam_full.terms[0]
# updating age_term.lam also updates it in gam_full.terms[0]
age_term.lam = approx_lam(Xgam, age_term, df=4+1)
year_term = gam_full.terms[1]
# also updated for gam_full
year_term.lam = approx_lam(Xgam, year_term, df=4+1)
gam_full = gam_full.fit(Xgam, y)

# refit partial dependence plot for age --> much smoother
fig, ax = plt.subplots(figsize=(8,8))
# a function is now defined in contrasts to previously verbose section
plot_gam(gam_full, 0, ax=ax)
ax.set_xlabel('Age')
ax.set_ylabel('Effect on wage')
ax.set_title('Partial dependence of age on wage - default lam=0.6', fontsize=20)


# partial dependence for year using df specification over lam
fig, ax = plt.subplots(figsize=(8,8))
plot_gam(gam_full,
         1,
         ax=ax)
ax.set_xlabel('Year')
ax.set_ylabel('Effect on wage')
ax.set_title('Partial dependence of year on wage', fontsize=20)

# partial dependence for education vs Wage
fig, ax = subplots(figsize=(8, 8))
ax = plot_gam(gam_full, 2)
ax.set_xlabel('Education')
ax.set_ylabel('Effect on wage')
ax.set_title('Partial dependence of wage on education',
             fontsize=20)
ax.set_xticklabels(Wage['education'].cat.categories, fontsize=8)



### ANOVA Tests for GAMs
# ANOVA test for year --> exclude, linear, or spline function?
# NOTE use of 'age_term' since we set the lam earlier based on df
gam_0 = LinearGAM(age_term + f_gam(2, lam=0))
gam_0.fit(Xgam, y)
gam_linear = LinearGAM(age_term +
                       l_gam(1, lam=0) +
                       f_gam(2, lam=0))
gam_linear.fit(Xgam, y)
anova_gam(gam_0, gam_linear, gam_full)

# ANOVA test for age --> see that non-linear term for age required
gam_0 = LinearGAM(year_term +
                  f_gam(2, lam=0))
gam_linear = LinearGAM(l_gam(0, lam=0) +
                       year_term +
                       f_gam(2, lam=0))
gam_0.fit(Xgam, y)
gam_linear.fit(Xgam, y)
anova_gam(gam_0, gam_linear, gam_full)

gam_full.summary()

## Make predictions on training set
Yhat = gam_full.predict(Xgam)

## Fit logistic regression using GAM, using pygam
gam_logit = LogisticGAM(age_term + 
                        l_gam(1, lam=0) +
                        f_gam(2, lam=0))
gam_logit.fit(Xgam, high_earn)

# plot partial dependence of logistic regression
fig, ax = plt.subplots(figsize=(8, 8))
ax = plot_gam(gam_logit, 2)
ax.set_xlabel('Education')
ax.set_ylabel('Effect on wage')
ax.set_title('Partial dependence of wage on education',
             fontsize=20)
ax.set_xticklabels(Wage['education'].cat.categories, fontsize=8)

# No high earners in first category --> misleading/wrong graph
pd.crosstab(Wage['high_earn'], Wage['education'])

# the '-1' included due to bug in pygam, just relabels education values
only_hs = Wage['education'] == '1. < HS Grad'
Wage_ = Wage.loc[~only_hs]
Xgam_ = np.column_stack([Wage_['age'],
                         Wage_['year'],
                         Wage_['education'].cat.codes-1])
high_earn_ = Wage_['high_earn']

# fit the model again
gam_logit_ = LogisticGAM(age_term +
                         year_term +
                         f_gam(2, lam=0))
gam_logit_.fit(Xgam_, high_earn_)

# partial dependence plot of education on higher earner status
fig, ax = plt.subplots(figsize=(8, 8))
ax = plot_gam(gam_logit_, 2)
ax.set_xlabel('Education')
ax.set_ylabel('Effect on wage')
ax.set_title('Partial dependence of high earner status on education', fontsize=20);
ax.set_xticklabels(Wage['education'].cat.categories[1:],
                   fontsize=8)

# partial - year vs high earn status
fig, ax = plt.subplots(figsize=(8, 8))
ax = plot_gam(gam_logit_, 1)
ax.set_xlabel('Year')
ax.set_ylabel('Effect on wage')
ax.set_title('Partial dependence of high earner status on year',
             fontsize=20)

# partial - age vs high earn status
fig, ax = plt.subplots(figsize=(8, 8))
ax = plot_gam(gam_logit_, 0)
ax.set_xlabel('Age')
ax.set_ylabel('Effect on wage')
ax.set_title('Partial dependence of high earner status on age', fontsize=20)

### Local Regression
lowess = sm.nonparametric.lowess
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(age, y, facecolor='gray', alpha=0.5)
for span in [0.2, 0.5]:
    fitted = lowess(y,
                    age,
                    frac=span,
                    xvals=age_grid)
    ax.plot(age_grid,
            fitted,
            label='{:.1f}'.format(span),
            linewidth=4)
ax.set_xlabel('Age', fontsize=20)
ax.set_ylabel('Wage', fontsize=20)
ax.legend(title='span', fontsize=15)




# %%        ######### TREE-BASED METHODS ##########

### Classification Trees

# data prep - response
Carseats = load_data('Carseats')
High = np.where(Carseats.Sales > 8,
                "Yes",
                "No")

# include all in the X model spec except response-related Sales
model = MS(Carseats.columns.drop('Sales'), intercept=False)
D = model.fit_transform(Carseats)
feature_names = list(D.columns)
X = np.asarray(D)

# there is an additional option 'min_samples_split',
# equal to min # of obs in node to be eligible to split
# criterion = Gini also possible
clf = DTC(criterion='entropy',
          max_depth=3,
          random_state=0)        
clf.fit(X, High)

# accuracy, or 1 - training error rate
accuracy_score(High, clf.predict(X))

# smaller deviance indicates tree provides good fit to training data
# close related to entropy
resid_dev = np.sum(log_loss(High, clf.predict_proba(X)))
resid_dev

# plot and print tree
ax = plt.subplots(figsize=(12,12))[1]
plot_tree(clf,
          feature_names=feature_names,
          ax=ax)

print(export_text(clf,
                  feature_names=feature_names,
                  show_weights=True))

# train on training data, score on test data
validation = skm.ShuffleSplit(n_splits=1,
                              test_size=200,
                              random_state=0)
results = skm.cross_validate(clf,
                             D,
                             High,
                             cv=validation)
results['test_score']



#### Pruning investigations --> better fit?

# split into train and test first
(X_train,
 X_test,
 High_train,
 High_test) = skm.train_test_split(X,
                                   High,
                                   test_size=0.5,
                                   random_state=0)

# fit tree on full training set, no max_depth as CV will inform
clf = DTC(criterion='entropy', random_state=0)
clf.fit(X_train, High_train)
accuracy_score(High_test, clf.predict(X_test))

# Extract cost complexity values using function
ccp_path = clf.cost_complexity_pruning_path(X_train, High_train)
kfold = skm.KFold(10,
                  random_state=1,
                  shuffle=True)

# yield set of impurities/alpha --> can extract optimal using CV
grid = skm.GridSearchCV(clf,
                        {'ccp_alpha': ccp_path.ccp_alphas},
                        # refit model using optimal ccp_alpha?
                        refit=True,
                        cv=kfold,
                        scoring='accuracy')

grid.fit(X_train, High_train)
grid.best_score_
grid.best_estimator_

# look at pruned tree
ax = plt.subplots(figsize=(12, 12))[1]
best_ = grid.best_estimator_
plot_tree(best_,
          feature_names=feature_names,
          ax=ax)

# leaves  
best_.tree_.n_leaves

# accuracy and confusion matrix
print(accuracy_score(High_test,
                     best_.predict(X_test)))
confusion = confusion_table(best_.predict(X_test),
                            High_test)
confusion



####### Fitting Regression Trees #####
Boston = load_data("Boston")
model = MS(Boston.columns.drop('medv'), intercept=False)
D = model.fit_transform(Boston)
feature_names = list(D.columns)
X = np.asarray(D)

## train test split - test gets 30%
(X_train,
 X_test,
 y_train,
 y_test) = skm.train_test_split(X,
                                Boston['medv'],
                                test_size=0.3,
                                random_state=0)

## fit tree and plot
reg = DTR(max_depth=3)
reg.fit(X_train, y_train)
np.mean((y_test - reg.predict(X_test))**2)

ax = plt.subplots(figsize=(12,12))[1]
plot_tree(reg,
          feature_names=feature_names,
          ax=ax)

# see if CV pruning will improve performance
ccp_path = reg.cost_complexity_pruning_path(X_train, y_train)
kfold = skm.KFold(5,
                  shuffle=True,
                  random_state=10)
grid = skm.GridSearchCV(reg,
                        {'ccp_alpha': ccp_path.ccp_alphas},
                        refit=True,
                        cv=kfold,
                        scoring='neg_mean_squared_error')
G = grid.fit(X_train, y_train)

# model fitted on the best alpha, and the MSE
best_ = grid.best_estimator_
np.mean((y_test - best_.predict(X_test))**2)

# plot best pruned tree
ax = plt.subplots(figsize=(12,12))[1]
plot_tree(G.best_estimator_,
          feature_names=feature_names,
          ax=ax)


#### Bagging and Random Forest #####

## Bagging - m = p special case of RF
bag_boston = RF(max_features=X_train.shape[1], 
                random_state=0)
bag_boston.fit(X_train, y_train)

ax = plt.subplots(figsize=(8,8))[1]
y_hat_bag = bag_boston.predict(X_test)
#   standardized residuals
# ax.scatter(y_hat_bag, (y_hat_bag - y_test)/
#            (y_hat_bag - y_test).std(ddof=1))
ax.scatter(y_hat_bag, y_test)
x = np.linspace(*ax.get_xlim(), 400)
ax.plot(x, x, label=f'y = x', color='blue')
np.mean((y_test - y_hat_bag)**2)

# change n_estimators (#trees) from default of 100
bag_boston = RF(max_features=X_train.shape[1],
                n_estimators=500,
                random_state=0).fit(X_train, y_train)
y_hat_bag = bag_boston.predict(X_test)
np.mean((y_test - y_hat_bag)**2)


#### Random Forest
RF_boston = RF(max_features=6,
               random_state=0).fit(X_train, y_train)
y_hat_RF = RF_boston.predict(X_test)
np.mean((y_test - y_hat_RF)**2)

# feature importance
feature_imp = pd.DataFrame(
    {'importance':RF_boston.feature_importances_},
    index=feature_names)
feature_imp.sort_values(by='importance', ascending=False).plot()

#### Boosting
# want 5000 trees with max_depth of 3 each
boost_boston = GBR(n_estimators=5000,
                   learning_rate=0.001,
                   max_depth=3,
                   random_state=0)
boost_boston.fit(X_train, y_train)
# see how training error decreases as more boosting trees added
boost_boston.train_score_

# get how test error decreases as more boosting trees added
test_error = np.zeros_like(boost_boston.train_score_)
for idx, y_ in enumerate(boost_boston.staged_predict(X_test)):
   test_error[idx] = np.mean((y_test - y_)**2)

# plot the progression of training/test error as boosting trees added
plot_idx = np.arange(boost_boston.train_score_.shape[0])
ax = plt.subplots(figsize=(8,8))[1]
ax.plot(plot_idx,
        boost_boston.train_score_,
        'b',
        label='Training')
ax.plot(plot_idx,
        test_error,
        'r',
        label='Test')
ax.legend()

# predict on test set
y_hat_boost = boost_boston.predict(X_test)
np.mean((y_test - y_hat_boost)**2)







# %%    ######### Support Vector Machines ##########

# generate some observations
rng = np.random.default_rng(1)
X = rng.standard_normal((50, 2))
y = np.array([-1]*25+[1]*25)
X[y==1] += 1
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(X[:,0],
           X[:,1],
           c=y,
           cmap=plt.cm.coolwarm)

# C is penalty for margin violations
# large C --> narrow margins
# fit the Support Vector Classifier
svm_linear = SVC(C=10, kernel='linear')
svm_linear.fit(X, y)

# visualize margins
fig, ax = plt.subplots(figsize=(8,8))
plot_svm(X,
         y,
         svm_linear,
         ax=ax)

# visualize margins using smaller cost of margin violations
svm_linear_small = SVC(C=0.1, kernel='linear')
svm_linear_small.fit(X, y)
fig, ax = plt.subplots(figsize=(8,8))
plot_svm(X,
         y,
         svm_linear_small,
         ax=ax)

# extract coefficients of linear boundary (linear only)
svm_linear.coef_


# SVM is an estimator in sklearn --> can tune via following method
kfold = skm.KFold(5, 
                  random_state=0,
                  shuffle=True)
grid = skm.GridSearchCV(svm_linear,
                        {'C':[0.001,0.01,0.1,1,5,10,100]},
                        refit=True,
                        cv=kfold,
                        scoring='accuracy')
grid.fit(X, y)
grid.best_params_

# extract CV score for each 'C' in grid
# C = 1 is best (lowest out of the best)
grid.cv_results_[('mean_test_score')]

# predict class on new data - use .best_estimator_
X_test = rng.standard_normal((20, 2))
y_test = np.array([-1]*10+[1]*10)
X_test[y_test==1] += 1

best_ = grid.best_estimator_
y_test_hat = best_.predict(X_test)
confusion_table(y_test_hat, y_test)

# use C=0.001 instead to classify new obs?
svm_ = SVC(C=0.001,
           kernel='linear').fit(X, y)
y_test_hat = svm_.predict(X_test)
confusion_table(y_test_hat, y_test)

# predict on linearly separable data?
X[y==1] += 1.9
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm)

# using large C
svm_ = SVC(C=1e5, kernel='linear').fit(X, y)
y_hat = svm_.predict(X)
confusion_table(y_hat, y)

fig, ax = plt.subplots(figsize=(8,8))
plot_svm(X,
         y,
         svm_,
         ax=ax)

# using small C
svm_ = SVC(C=0.1, kernel='linear').fit(X, y)
y_hat = svm_.predict(X)
confusion_table(y_hat, y)

fig, ax = plt.subplots(figsize=(8,8))
plot_svm(X,
         y,
         svm_,
         ax=ax)

############ Support Vector Machines ###########
# non-linear kernels

# generate non-linear boundary data
X = rng.standard_normal((200, 2))
X[:100] += 2
X[100:150] -= 2
y = np.array([1]*150+[2]*50)

fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(X[:,0],
           X[:,1],
           c=y,
           cmap=plt.cm.coolwarm)


# fit using radial kernal, gamma = 1, C = 1
(X_train, 
 X_test,
 y_train,
 y_test) = skm.train_test_split(X,
                                y,
                                test_size=0.5,
                                random_state=0)
svm_rbf = SVC(kernel="rbf", gamma=1, C=1)
svm_rbf.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(8,8))
plot_svm(X_train,
         y_train,
         svm_rbf,
         ax=ax)

# higher C reduces training error, but may overfit
svm_rbf = SVC(kernel="rbf", gamma=1, C=1e5, probability=True)
svm_rbf.fit(X_train, y_train)
fig, ax = plt.subplots(figsize=(8,8))
plot_svm(X_train,
         y_train,
         svm_rbf,
         ax=ax)

# use CV to choose C and gamma params for the radial kernel
kfold = skm.KFold(5, 
                  random_state=0,
                  shuffle=True)
grid = skm.GridSearchCV(svm_rbf,
                        {'C':[0.1,1,10,100,1000],
                         'gamma':[0.5,1,2,3,4]},
                        refit=True,
                        cv=kfold,
                        scoring='accuracy');
grid.fit(X_train, y_train)
grid.best_params_

best_svm = grid.best_estimator_
fig, ax = plt.subplots(figsize=(8,8))
plot_svm(X_train,
         y_train,
         best_svm,
         ax=ax)

y_hat_test = best_svm.predict(X_test)
confusion_table(y_hat_test, y_test)


### ROC Curves
roc_curve = RocCurveDisplay.from_estimator # shorthand
fig, ax = plt.subplots(figsize=(8,8))
# takes fitted estimator, X matrix and Y labels as args
roc_curve(best_svm,
          X_train,
          y_train,
          name='Training',
          color='r',
          ax=ax)

roc_curve(best_svm,
          X_test,
          y_test,
          name='Training',
          color='b',
          ax=ax)

from sklearn.metrics import roc_auc_score
# get predicted probabilities for test set
# note that order is determined lexicographically?
# e.g. labels 1 and 2 would get cast to 0 and 1?
y_prob = best_svm.predict_proba(X_test)[:, 1]
# Compute AUC
auc = roc_auc_score(y_test, y_prob)
print(f'AUC: {auc:.2f}')



###### SVM with multiple classes #####
## generate third class of observations
rng = np.random.default_rng(123)
X = np.vstack([X, rng.standard_normal((50, 2))])
y = np.hstack([y, [0]*50])
X[y==0,1] += 2
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm)

## fit SVM
# ovo - one-versus-one --> fit K(K+1)/2 SVMs, class i,j
#       works by tallying #times test obs assigned to a class
# ovr - one-versus-rest --> fit K SVMs (one per class, all others is the other category)
#       the Class SVM with highest probability (or score) assigned
svm_rbf_3 = SVC(kernel="rbf",
                C=10,
                gamma=1,
                decision_function_shape='ovo')
svm_rbf_3.fit(X, y)
fig, ax = plt.subplots(figsize=(8,8))
plot_svm(X,
         y,
         svm_rbf_3,
         scatter_cmap=plt.cm.tab10,
         ax=ax)


## Application to Gene Expression Data
Khan = load_data('Khan')
Khan['xtrain'].shape, Khan['xtest'].shape

# fit linear kernel, very large p so flexibility unnecessary
khan_linear = SVC(kernel='linear', C=10)
khan_linear.fit(Khan['xtrain'], Khan['ytrain'])
confusion_table(khan_linear.predict(Khan['xtrain']),
                Khan['ytrain'])

confusion_table(khan_linear.predict(Khan['xtest']),
                Khan['ytest'])






# %%    ############### DEEP LEARNING #################
# https://pytorch.org/tutorials/beginner/basics/intro.html
# https://www.kaggle.com/code/ddayanavincent/machine-learning-using-polars


##### Single Layer Network on Hitters data #####
Hitters = load_data('Hitters').dropna()
n = Hitters.shape[0]

# model matrix and response
model = MS(Hitters.columns.drop('Salary'), intercept=False)
X = model.fit_transform(Hitters).to_numpy()
Y = Hitters['Salary'].to_numpy()

# train test split
(X_train, 
 X_test,
 Y_train,
 Y_test) = train_test_split(X,
                            Y,
                            test_size=1/3,
                            random_state=1)
 
# linear models (to compare with NN)
hit_lm = LinearRegression().fit(X_train, Y_train)
Yhat_test = hit_lm.predict(X_test)
np.abs(Yhat_test - Y_test).mean()
# %%    ##### 3.7 page 136

# %%
import torch

print(f'CUDA version: {torch.version.cuda}')
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device:{torch.cuda.current_device()}")
print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")
print(f'Torch version: {torch.__version__}')
# %%
