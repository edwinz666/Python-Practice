# %%
# potential solutions: https://www.lackos.xyz/itsl/

# Library imports
from funcs import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import statsmodels as sm
import pyarrow as pa

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
auto.select(
    cs.numeric().max().name.suffix("_max"),
    cs.numeric().min().name.suffix("_min"),
    (cs.numeric().max() - cs.numeric().min()).name.suffix("_range")
).unpivot()
# %%

# %%

# %%
