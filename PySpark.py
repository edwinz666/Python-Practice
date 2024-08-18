# %%
from pyspark.sql import SparkSession
from datetime import datetime, date
import pandas as pd
from pyspark.sql import Row

import os
import sys
from pyspark.sql import SparkSession

# https://stackoverflow.com/questions/48260412/environment-variables-pyspark-python-and-pyspark-driver-python 
# also set System variable PYSPARK_PYTHON to python ?
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# %%

spark = SparkSession.builder.getOrCreate()
# spark = (SparkSession.builder.appName("Datacamp Pyspark Tutorial")
# .config("spark.memory.offHeap.enabled","true").config("spark.memory.offHeap.size","10g").getOrCreate())

# spark = SparkSession.builder \
#     .appName("your-app-name") \
#     .config("spark.executor.heartbeatInterval", "60s") \
#     .config("spark.network.timeout", "600s") \
#     .config("spark.executor.pyspark.memory", "2g") \
#     .config("spark.driver.memory", "4g") \
#     .getOrCreate()
    

df = spark.createDataFrame([
    Row(a=1, b=2., c='string1', d=date(2000, 1, 1), e=datetime(2000, 1, 1, 12, 0)),
    Row(a=2, b=3., c='string2', d=date(2000, 2, 1), e=datetime(2000, 1, 2, 12, 0)),
    Row(a=4, b=5., c='string3', d=date(2000, 3, 1), e=datetime(2000, 1, 3, 12, 0))
])
df.show()
df.head(5)

df = spark.createDataFrame([
    ['red', 'banana', 1, 10], ['blue', 'banana', 2, 20], ['red', 'carrot', 3, 30],
    ['blue', 'grape', 4, 40], ['red', 'carrot', 5, 50], ['black', 'carrot', 6, 60],
    ['red', 'banana', 7, 70], ['red', 'grape', 8, 80]], schema=['color', 'fruit', 'v1', 'v2'])
df.show()
print(df.count())
# %%


spark = SparkSession.builder.master("local").appName("PySpark Installation Test").getOrCreate()
df = spark.createDataFrame([(1, "Hello"), (2, "World")], ["id", "message"])
df.printSchema()
df.show()
# %%


# Set the Python executable path

# Create Spark session
spark = SparkSession.builder \
    .appName("MyApp") \
    .getOrCreate()

# Sample code to show dataframe
df = spark.createDataFrame([(1, "foo"), (2, "bar")], ["id", "value"])
df.show()

# test some random commit