# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

import polars as pl
import polars.selectors as cs

import xlsxwriter

plt.style.use('ggplot')
ax1 = sns.set_style(style=None, rc=None )

# %%
# error in parsing IDpol column due to scientific notation, 
# import as a Float first, then convert to Integer
data = pl.scan_csv("freMTPL2freq.csv", 
                   schema_overrides={"IDpol": pl.Float64}
                   ).collect()
data = data.with_columns(pl.col("IDpol").cast(pl.Int64))
 # %%
data.schema
data.null_count()
data.filter(data.is_duplicated())

# view unique values of designated factor columns
factor_cols = ["Area", "VehPower", "VehBrand", "VehGas", "Region"]

for c in factor_cols:
    print(data[c].unique().sort())

# set ordered factor values for several columns
area = data["Area"].unique().sort()

vehPower_1 = data.select(pl.col("VehPower").unique())
vehPower_2 = vehPower_1.select(
    pl.col("VehPower").cast(pl.String).sort_by(pl.col("VehPower"))
)

vehBrand_1 = data.select(pl.col("VehBrand").unique())
vehBrand_2 = vehBrand_1.select(
    pl.col("VehBrand").sort_by(
        pl.col("VehBrand").str.slice(1).cast(pl.Int64))
    )

vehGas = data["VehGas"].unique().sort()
region = data["Region"].unique().sort()

enums = [area, vehPower_2, vehBrand_2, vehGas, region]

# update factor columns in data with ordered factor types
for e, f in zip(enums, factor_cols):
    data = data.with_columns(
        pl.col(f).cast(pl.String).cast(pl.Enum(e))
    )
    
print(data.schema)

# %%

# For each variable in the dataset, group by that variable, and output a sheet detailing:
#   1. A summary per category, detailing row counts, proportion of
#      row counts, total exposure, total claim numbers, and claim frequency.
#      This output is only performed if the number of categories is below a certain threshold.
#   2. An overall summary across all categories in the variable, detailing counts, 
#      null_counts, distinct_counts, mean, std, and various percentiles
#   3. A graph illustrating the row counts per category
#   4. A graph illustrating the total exposure and claims frequency per category
with xlsxwriter.Workbook("variable_summary.xlsx") as wb:  
    # create format for percent-formatted columns
    perc_format = wb.add_format({'num_format': '#,##0.00%'})  
    
    for col in data.columns:
        # create the worksheet for the variable
        ws = wb.add_worksheet(col)
        
        # 1. { ... }
        temp = data.group_by(col).agg(
            pl.len().alias("count"),
            (pl.len() / data.height).alias("count_perc"),
            pl.sum("Exposure").alias("Total_Exposure"),
            pl.sum("ClaimNb").alias("Total_ClaimNb"),
        ).with_columns(
            (pl.col("Total_ClaimNb") / pl.col("Total_Exposure")).alias("Claim_Freq"),
            (pl.sum("Total_ClaimNb") / pl.sum("Total_Exposure")).alias("average_freq")
        ).sort(col)
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
                column_formats={'count_perc': '0.00%', 
                                'Claim_Freq': '0.00%',
                                'Total_Exposure': '#,##0'},
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
        
        
        
        # Exposure and Freq chart
        column_chart = wb.add_chart({"type": "column"})
        column_chart.set_title({"name": col})
        column_chart.set_legend({"none": False, "position": "bottom"})
        column_chart.set_style(38)
        column_chart.add_series(
            {  # note the use of structured reference
                "name": "Total_Exposure",
                "values": "={}[{}]".format(col, "Total_Exposure"),
                "categories": "={}[{}]".format(col, col),
                "data_labels": {"value": False},
            }
        )

        # Create a new line chart. This will use this as the secondary chart.
        line_chart = wb.add_chart({"type": "line"})

        # Configure the data series for the secondary chart. We also set a
        # secondary Y axis via (y2_axis).
        line_chart.add_series(
            {
                "name": "Claim Frequency",
                "values": "={}[{}]".format(col, "Claim_Freq"),
                "categories": "={}[{}]".format(col, col),
                "y2_axis": True,
                "line": {'width': 3, 'color': '#770737'}
            }
        )
        
        line_chart.add_series(
            {
                "name": "Average Claim Frequency",
                "values": "={}[{}]".format(col, "average_freq"),
                "categories": "={}[{}]".format(col, col),
                "y2_axis": True,
                "line": {'width': 1.5, 'dash_type': 'dash'}
            }
        )

        # Combine the charts.
        column_chart.combine(line_chart)

        # Add a chart title and some axis labels.
        column_chart.set_title({"name": "Exposure and Claim Frequency"})
        column_chart.set_x_axis({"name": col})
        column_chart.set_y_axis({"name": "Exposure"})

        # Note: the y2 properties are on the secondary chart.
        line_chart.set_y2_axis({"name": "Claim Frequency"})
        
        ws.insert_chart(18, temp.width + 1 + summary.width + 1, column_chart, 
                        options={'x_scale': 1.5, 'y_scale': 1.5}
        )   
        
        
# %%       
        
# From the summaries for each variable, the following may require some further investigations

# ClaimNb > 3 (total of 16 rows out of 678013), 

# Exposure = 1 has a very large mass, comprising 168,125 out of the 678,013 rows.
# There is potential for there to be a data issue here, for example the Exposure field 
# may be defaulted to 1 when there is no data available.

# VehAge > 90, 
# VehAge >= 99 (there is a step increase in the row count for
# VehAge's 99 and 100. In addition, there was also a step increase
# in the VehAge category itself, jumping from VehAge = 85 to VehAge = 99). 
# For reference, VehAge's 70-85 all had at most 3 rows in each category), 

# DrivAge = 99 had a large step increase in the row count 
# (from 5 rows for DrivAge = 98 to 70 rows for DrivAge = 99). Investigate DrivAges >= 99
# to determine if there were any data issues

# Large density for the Density column at 27000, which also corresponds to the largest value for this column.
# Investigate potential issues with the data when Density = 27000
var_to_check = [("ClaimNb", pl.col("ClaimNb") > 3), 
                ("VehAge", pl.col("VehAge") > 90),
                ("DrivAge", pl.col("DrivAge") >= 99), 
                ("Density", pl.col("Density") == 27000)
                ]

# output rows under investigation to an Excel file
with xlsxwriter.Workbook("rows_to_check.xlsx") as wb:
    # iterate over the variable and corresponding filter to generate
    # sheets for each variable which will contain the filtered data
    for v, f in var_to_check:
        rows_to_check = data.filter(f).sort(v)
        print(rows_to_check)

        ws = wb.add_worksheet(v)
    
        rows_to_check.write_excel(
            workbook=wb, 
            worksheet=v,
            position="A1",
            table_name=v,
            table_style="Table Style Medium 26",
            hide_gridlines=True,
            autofit=True
        )
    

# %%

# Breakpoints for density to get roughly equal exposure in each group
# density_breaks = data.sort("Density").select(
#     pl.col("Density"),
#     pl.col("Exposure").cum_sum().qcut(10)
# ).unique("Exposure",keep='last').select("Density").to_series()
# print(density_breaks)
# density_breaks.append(pl.Series([0]))
# density_breaks = density_breaks.sort()

# split Exposure into 10 equal breakpoints
exp_sum = data["Exposure"].sum()
exp_breaks = np.linspace(exp_sum / 10, np.floor(exp_sum), 10)
exp_breaks

# split Density into 10 equal bands by Exposure (roughly)
density_breaks = (
    data.group_by("Density").agg(
    pl.col("Exposure").sum()).sort("Density").select(
    pl.col("Density").gather(pl.cum_sum("Exposure").
        search_sorted(exp_breaks, side='left'))).to_series()
)
print(density_breaks)

# Grouping categories
data = data.with_columns(
    # Group VehPower 12+
    pl.when(pl.col('VehPower').cast(pl.Int64) >= 12).then(pl.lit('12+'))
                    .otherwise(pl.col('VehPower').cast(pl.String)).cast(pl.Enum(
                        [str(x) for x in range(4, 12)] + ['12+'])).alias("VehPower_New"),
    # Group VehAge 20+
    pl.when(pl.col('VehAge') >= 20).then(pl.lit('20+')).otherwise(pl.col('VehAge')
                    .cast(pl.String).str.zfill(2)).cast(pl.Categorical("lexical")).alias("VehAge_New"),
    # Group DrivAge into 5-year bands
    pl.col("DrivAge").cut(np.arange(15, 105, 5)).alias("DrivAge_Bands"),
    # Group BonusMalus into 50-75, 75-100, 100-125, and 125-230
    pl.col("BonusMalus").cut([75, 100, 125], left_closed=True).alias("BonusMalus_Bands"),
    # Group Density into 10 bands such that there is roughly equal exposure in each
    pl.col("Density").cut(density_breaks).alias("Density_Bands")
    )

# Re-create Excel workbook with additional grouped categories
with xlsxwriter.Workbook("variable_summary.xlsx") as wb:  
    # create format for percent-formatted columns
    perc_format = wb.add_format({'num_format': '#,##0.00%'})  
    
    for col in data.columns:
        # create the worksheet for the variable
        ws = wb.add_worksheet(col)
        
        # 1. { ... }
        temp = data.group_by(col).agg(
            pl.len().alias("count"),
            (pl.len() / data.height).alias("count_perc"),
            pl.sum("Exposure").alias("Total_Exposure"),
            pl.sum("ClaimNb").alias("Total_ClaimNb"),
        ).with_columns(
            (pl.col("Total_ClaimNb") / pl.col("Total_Exposure")).alias("Claim_Freq"),
            (pl.sum("Total_ClaimNb") / pl.sum("Total_Exposure")).alias("average_freq")
        ).sort(col)
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
                column_formats={'count_perc': '0.00%', 
                                'Claim_Freq': '0.00%',
                                'Total_Exposure': '#,##0'},
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
        
        
        
        # Exposure and Freq chart
        column_chart = wb.add_chart({"type": "column"})
        column_chart.set_title({"name": col})
        column_chart.set_legend({"none": False, "position": "bottom"})
        column_chart.set_style(38)
        column_chart.add_series(
            {  # note the use of structured reference
                "name": "Total_Exposure",
                "values": "={}[{}]".format(col, "Total_Exposure"),
                "categories": "={}[{}]".format(col, col),
                "data_labels": {"value": False},
            }
        )

        # Create a new line chart. This will use this as the secondary chart.
        line_chart = wb.add_chart({"type": "line"})

        # Configure the data series for the secondary chart. We also set a
        # secondary Y axis via (y2_axis).
        line_chart.add_series(
            {
                "name": "Claim Frequency",
                "values": "={}[{}]".format(col, "Claim_Freq"),
                "categories": "={}[{}]".format(col, col),
                "y2_axis": True,
                "line": {'width': 3, 'color': '#770737'}
            }
        )
        
        line_chart.add_series(
            {
                "name": "Average Claim Frequency",
                "values": "={}[{}]".format(col, "average_freq"),
                "categories": "={}[{}]".format(col, col),
                "y2_axis": True,
                "line": {'width': 1.5, 'dash_type': 'dash'}
            }
        )

        # Combine the charts.
        column_chart.combine(line_chart)

        # Add a chart title and some axis labels.
        column_chart.set_title({"name": "Exposure and Claim Frequency"})
        column_chart.set_x_axis({"name": col})
        column_chart.set_y_axis({"name": "Exposure"})

        # Note: the y2 properties are on the secondary chart.
        line_chart.set_y2_axis({"name": "Claim Frequency"})
        
        ws.insert_chart(18, temp.width + 1 + summary.width + 1, column_chart, 
                        options={'x_scale': 1.5, 'y_scale': 1.5}
        )  
        
        
         
# %%

# %%
cols=["VehAge_New", "DrivAge_Bands", "VehPower", "VehPower", "VehPower_New", "VehPower_New"] 
hues=["BonusMalus_Bands", "BonusMalus_Bands", "VehGas", "VehGas","VehGas","VehGas"]
filters=[None, None, None, pl.col("VehBrand") != "B12", None, pl.col("VehBrand") != "B12"]
filters_as_string=["", "", "", ", without Brand B12","", ", without Brand B12"]

for (c, h, f, fm) in zip(cols, hues, filters, filters_as_string):
    if f is None:
        temp = data
    else:
        temp = data.filter(f)
        
    temp = temp.group_by(c, h).agg(
                pl.len().alias("count"),
                (pl.len() / data.height).alias("count_perc"),
                pl.sum("Exposure").alias("Total_Exposure"),
                pl.sum("ClaimNb").alias("Total_ClaimNb"),
                ).with_columns(
                    (pl.col("Total_ClaimNb") / pl.col("Total_Exposure")).alias("Claim_Freq"),
                    (pl.sum("Total_ClaimNb") / pl.sum("Total_Exposure")).alias("average_freq")
                ).sort(c)
            
    fig, ax1 = plt.subplots(figsize=(12,8))
    sns.lineplot(data = temp, x = c, y = 'Claim_Freq', hue=h,
                marker='o', sort = False, ax=ax1)
    sns.lineplot(data = temp, x = c, y = 'average_freq',
                linestyle = '--', color = '#770737', sort = False, ax=ax1, 
                label="Average Claim Frequency")

    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.set(ylabel="Claim Frequency", title='Claim Frequency by {} and {}{}'.format(c, h, fm) )

# %%

# %%
