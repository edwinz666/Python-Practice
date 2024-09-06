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
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

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

# %%    ##### 3.7 page 136
