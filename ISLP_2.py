# %%
# Library imports
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
def var_overview(data, filename):
    with xlsxwriter.Workbook(filename) as wb:  
        # create format for percent-formatted columns
        perc_format = wb.add_format({'num_format': '#,##0.00%'})  
        
        for col in data.columns:
            # create the worksheet for the variable
            ws = wb.add_worksheet(col)
            
            # 1. { ... }
            temp = (
                data.group_by(col).agg(
                    pl.len().alias("count"),
                    (pl.len() / data.height).alias("count_perc"),
                # pl.sum("Exposure").alias("Total_Exposure"),
                # pl.sum("ClaimNb").alias("Total_ClaimNb"),
                # )
                # .with_columns(
                #     (pl.col("Total_ClaimNb") / pl.col("Total_Exposure")).alias("Claim_Freq"),
                #     (pl.sum("Total_ClaimNb") / pl.sum("Total_Exposure")).alias("average_freq")
                ).sort(col)
            )
            # print(temp)
            
            # output this section only if lower than 100,000 categories
            max_height_1 = 100_000
            if temp.height <= max_height_1:
                temp.write_excel(
                    workbook=wb, 
                    worksheet=col,
                    position="A1",
                    table_name=col,
                    table_style="Table Style Medium 26",
                    hide_gridlines=True,
                    column_formats={'count_perc': '0.00%'
                                    # , 'Claim_Freq': '0.00%'
                                    # , 'Total_Exposure': '#,##0'
                                    },
                    autofit=True
                )
                
                
            # 2. { ... }
            summary = data.select(pl.col(col)).to_series().describe()
            additional = temp.select(
                pl.col(col).len().alias("distinct_count")
                ).unpivot(variable_name="statistic", value_name="value")
            
            summary.write_excel(
                workbook=wb, 
                worksheet=col,
                position=(0, temp.width + 1),
                table_name=col + "_summary",
                table_style="Table Style Medium 26",
                hide_gridlines=True,
                autofit=True
            )
            
            additional.write_excel(
                workbook=wb, 
                worksheet=col,
                position=(summary.height + 2, temp.width + 1),
                table_name=col + "_additional",
                table_style="Table Style Medium 26",
                hide_gridlines=True,
            )  

            # 3. { ... }
            # don't provide graphs for variables with a high # of categories
            max_height_2 = 1000
            if temp.height > max_height_2:
                continue
            
            # don't include data labels in graph if exceeding 10 unique values
            max_height_3 = 10
            data_labels = temp.height <= max_height_3
            
            # Row count chart
            chart = wb.add_chart({"type": "column"})
            chart.set_title({"name": col})
            chart.set_legend({"none": True})
            chart.set_style(38)
            chart.add_series(
                {  # note the use of structured references
                    "values": "={}[{}]".format(col, "count"),
                    "categories": "={}[{}]".format(col, col),
                    "data_labels": {"value": data_labels},
                }
            )
            
            # add chart to the worksheet
            ws.insert_chart(0, temp.width + 1 + summary.width + 1, chart)
        
            # # Exposure and Freq chart
            # column_chart = wb.add_chart({"type": "column"})
            # column_chart.set_title({"name": col})
            # column_chart.set_legend({"none": False, "position": "bottom"})
            # column_chart.set_style(38)
            # column_chart.add_series(
            #     {  # note the use of structured reference
            #         "name": "Total_Exposure",
            #         "values": "={}[{}]".format(col, "Total_Exposure"),
            #         "categories": "={}[{}]".format(col, col),
            #         "data_labels": {"value": False},
            #     }
            # )

            # # Create a new line chart. This will use this as the secondary chart.
            # line_chart = wb.add_chart({"type": "line"})

            # # Configure the data series for the secondary chart. We also set a
            # # secondary Y axis via (y2_axis).
            # line_chart.add_series(
            #     {
            #         "name": "Claim Frequency",
            #         "values": "={}[{}]".format(col, "Claim_Freq"),
            #         "categories": "={}[{}]".format(col, col),
            #         "y2_axis": True,
            #         "line": {'width': 3, 'color': '#770737'}
            #     }
            # )
            
            # line_chart.add_series(
            #     {
            #         "name": "Average Claim Frequency",
            #         "values": "={}[{}]".format(col, "average_freq"),
            #         "categories": "={}[{}]".format(col, col),
            #         "y2_axis": True,
            #         "line": {'width': 1.5, 'dash_type': 'dash'}
            #     }
            # )

            # # Combine the charts.
            # column_chart.combine(line_chart)

            # # Add a chart title and some axis labels.
            # column_chart.set_title({"name": "Exposure and Claim Frequency"})
            # column_chart.set_x_axis({"name": col})
            # column_chart.set_y_axis({"name": "Exposure"})

            # # Note: the y2 properties are on the secondary chart.
            # line_chart.set_y2_axis({"name": "Claim Frequency"})
            
            # ws.insert_chart(18, temp.width + 1 + summary.width + 1, column_chart, 
            #                 options={'x_scale': 1.5, 'y_scale': 1.5}
            # )
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











# %%    ###### CONVERT TO ENUM FUNCTION ########

# Converts selected columns of a dataframe to an ordered enum type
def to_ordered_enum(
    data: pl.DataFrame, 
    colnames: list[str],
    ):
    
    # initialise list of expressions to cast 'data' to
    exprs = []
    
    for col in colnames:
        dtype = data.dtypes[data.get_column_index(col)]
        
        # generate list for the Enum type to cast to
        if dtype.is_(pl.Categorical):
            sorted_values = data[col].unique().cat.get_categories().sort()
        elif dtype.is_(pl.Enum):
            sorted_values = data[col].unique().cast(pl.String).sort()
        elif dtype.is_numeric():
            max_digits = len(str(data[col].max()))
            sorted_values = data[col].unique().sort().cast(pl.String).str.zfill(max_digits) 
        else:
            sorted_values = data[col].unique().sort()

        
        # append an expression to list depending on if it is a numeric type
        if dtype.is_numeric():
            exprs.append(pl.col(col).cast(pl.String).str.zfill(max_digits).cast(pl.Enum(sorted_values)))
        else:
            exprs.append(pl.col(col).cast(pl.Enum(sorted_values)))

    # change data types of 'data' to Enum for the list of expressions provided
    data = data.with_columns(*exprs)
    
    return data


sort_cats = pl.DataFrame(
    [pl.Series("blah",["1","15","152"]),
     pl.Series("blah2",[1, 152, 15]),
     pl.Series("blah3",["a","dd","b"])
    ]
    , 
    # schema={"blah":pl.Categorical}
    )
sort_cats
sort_cats.with_columns(pl.col("blah").cast(pl.Enum(pl.col("blah").implode())))
sort_cats.with_columns(pl.col("blah").cast(pl.Enum(["1","15","152"])))
sort_cats.select((pl.col("blah2").max()-1).cast(pl.String).str.len_chars())
sort_cats.select( [pl.col("blah2").alias("alias") + 10, pl.col("blah3")])
testsss = to_ordered_enum(sort_cats, ["blah", "blah2","blah3"])
testsss.dtypes
to_ordered_enum(testsss, ["blah"])


# %%    ### 2.4  page 65 ###

### a)
college = pl.scan_csv("data/College.csv").collect()
college.head()
college.schema

### c)
college.describe()
### d)
first_cols = ["Top10perc", "Apps", "Enroll"]
college.select(first_cols)
sns.pairplot(college.select(first_cols).to_pandas())
g = sns.PairGrid(college.select(first_cols))
g.map_lower(sns.scatterplot)
g.map_upper(sns.kdeplot)
g.map_diag(sns.histplot)
sns.histplot(college["Top10perc"], kde=True)

test1 = college["Top10perc"].value_counts()
sns.barplot(test1, x="Top10perc", y="count")
sns.countplot(college, x="Top10perc")