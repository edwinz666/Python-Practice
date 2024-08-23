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
# %%
import hvplot.polars
import hvplot.pandas
import panel as pn
pn.extension('bokeh')

data = pl.scan_csv("freMTPL2freq.csv", 
                   schema_overrides={"IDpol": pl.Float64}
                   ).collect()
data = data.with_columns(pl.col("IDpol").cast(pl.Int64))
data.describe()

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

# %%
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

# data.filter(pl.col("VehAge") == w_latitude).hvplot(
#     kind='box',
#     by=['Area'],
#     # x='Area',
#     y='Exposure',
#     legend='bottom_right',
#     widget_location='bottom',
# )
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
var_overview(college, "var_overview.xlsx")
# %%
