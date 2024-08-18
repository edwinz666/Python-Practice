# %%
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

# %%
df = pl.DataFrame({"b": [1, 2, 2, 2, 0 , 1, 2], "a": [3, 4, 8, 15, 31, 31, 15]})

df.pipe(lambda tdf: tdf.select(sorted(tdf.columns)))
df.select(sorted(df.columns))

# df.pipe(lambda df: df.with_columns(
#     get_ham("col_a"),
#     get_bar("col_b", df.schema),
#     get_foo("col_c", df.schema),
# )

df.pipe(lambda df: df.with_columns(
    # .over = groupby
    # .col().agg_Function() is column to aggregate over
    pl.col("a").sum().over("a").alias("a_sum"),
    pl.col("b").sum().over("b").alias("b_sum"),
    
    # 
    pl.col("b").sum().over("a").alias("b_over_a_sum"),
    (pl.col("b")/pl.col("a")).alias("row_context_div"),
    (pl.col("b")/pl.col("a")).sum().over("a").alias("div_over_groupby_a"),
    pl.col("b").sum().alias("b_sum_all"),
    
    pl.col("b").sum().over(["a","b"]).alias("min_b_over_ab")
    )
)
# %%
import polars as pl
from datetime import datetime

df = pl.DataFrame(
    {
        "integer": [1, 2, 3],
        "date": [
            datetime(2025, 1, 1),
            datetime(2025, 1, 2),
            datetime(2025, 1, 3),
        ],
        "float": [4.0, 5.0, 6.0],
        "string": ["a", "b", "c"],
    }
)

print(df)
# %%
df.write_csv("examples/output.csv")
df_csv = pl.read_csv("examples/output.csv")
print(df_csv)
# %%
cat_series = pl.Series( "cat",
    ["Polar", "Panda", "Brown", "Brown", "Polar"], dtype=pl.Categorical,
)
cat2_series = pl.Series( "cat2",
    ["Panda", "Brown", "Brown", "Polar", None], dtype=pl.Categorical, 
)
cat_series.rename("renamed_cat")
cat_series

# Triggers a CategoricalRemappingWarning: Local categoricals have different encodings, expensive re-encoding is done
# APPENDS IN PLACE and changes cat_series when run
cat_series.append(cat2_series)
# %%
cat_series
cat2_series
pl.DataFrame(cat2_series).hstack(pl.DataFrame(cat_series)).describe()

# %%
df = pl.DataFrame(
    {
        "nrs": [1, 2, 3, None, 5],
        "names": ["foo", "ham", "spam", "egg", None],
        "random": np.random.rand(5),
        "groups": ["A", "A", "B", "C", "B"],
    }
)
print(df)
print(df.select(pl.col("names").n_unique()))
# .count() excludes Null values for that column
print(df.select(pl.col("names").count()))

out = df.group_by("groups").agg(
    pl.sum("nrs").alias("sum_1"),  # sum nrs by groups
    pl.col("nrs").sum().alias("sum_2"), # another way to define it?
    pl.col("random").count().alias("count"),  # count group members
    # sum random where name != null
    pl.col("random").filter(pl.col("names").is_not_null()).sum().name.suffix("_sum"),
    pl.col("names").reverse().alias("reversed names"),
)
print(out)

df.select(
    pl.col("groups").sort().head(2), 
    # pl.col("groups").sort().head(3).alias("groups_head_3"),
    pl.col("nrs").filter(pl.col("names") == "ham").sum()
)

df.group_by(cs.string()).agg(cs.numeric().sum())
# %%
import re
print(re.search("<.*?>", "<a>" ))
print(re.search("<[^A]>", "<a>" ))
print(re.search("<[^a]>", "<a>" ))

print(re.search("<[^a]>", "<b>" ))
print(re.search("<[^a-c]>", "<b>" ))
print(re.search("<.+>", "<>" ))

print(re.search("(<)?(\w+@\w+(?:\.\w+)+)(?(1)>|$)", '<user@host.com'))
print(re.search("(<)?(\w+@\w+(?:\.\w+)+)(?(1)>|$)", 'user@host.com'))
print(re.search("(<)?(\w+@\w+(?:\.\w+)+)(?(1)>|$)", 'user@host.com>'))
print(re.search("(<)?(\w+@\w+(?:\.\w+)+)(?(1)>|\$)", 'user@host.com$'))
print(re.search("(<)?(\w+@\w+(?:\.\w+)+)(?(1)>|\$)", 'user@host.com'))

print(re.search(r'\bat\b', "as at ay"))

m = re.search(r'(?<=-)\w+', 'spam-egg')

m.group(0)

def is_allowed_specific_char(string):
    charRe = re.compile(r'[^a-zA-Z0-9.]')
    string = charRe.search(string)
    return not bool(string)

re.match("foo.*?foo" , "a")

charRe = re.compile(r'[^a-zA-Z0-9.]*+')
print(charRe.match("ABCDEFabcdef123450."))
print(is_allowed_specific_char("ABCDEFabcdef123450.")) 
print(is_allowed_specific_char("*&%@#!}{"))
# %%
import polars.selectors as cs
import polars as pl

df = pl.DataFrame(
    {
        "w": ["xx", "yy", "xx", "yy", None],
        "x": [1, 2, 1, 4, -2],
        "y": [3.0, 4.5, 1.0, 2.5, -2.0],
        "z": ["a", "b", "a", "b", "b"],
    },
)

df.top_k(k=3, by="w")
df.group_by(cs.string()).agg(cs.numeric().sum())
print(df.select(cs.by_index(1,3)))
print(df.select(cs.by_index([1,3])))
print(df.select(cs.by_index(range(1,3))))

# infinity generated from float division
print( df.select(pl.col("y")/(pl.col("x")-1)) )
# %%
url = "https://theunitedstates.io/congress-legislators/legislators-historical.csv"

schema_overrides = {
    "first_name": pl.Categorical,
    "gender": pl.Categorical,
    "type": pl.Categorical,
    "state": pl.Categorical,
    "party": pl.Categorical,
}

dataset = pl.read_csv(url, schema_overrides=schema_overrides).with_columns(
    pl.col("birthday").str.to_date(strict=False)
)

dataset.shape
# %%
q = (
    dataset.lazy()
    .group_by("first_name")
    .agg(
        pl.len(),
        pl.col("gender"),
        pl.first("last_name"),
    )
    .sort("len", descending=True)
    .limit(5)
)

df = q.collect()
print(df)

# %%
q = (
    dataset.lazy()
    .group_by("state")
    .agg(
        (pl.col("party") == "Anti-Administration").sum().alias("anti"),
        (pl.col("party") == "Pro-Administration").sum().alias("pro"),
    )
    .sort("pro", descending=True)
    .limit(5)
)

df = q.collect()
print(df)
# %%
q = (
    dataset.lazy()
    .group_by("state", "party")
    .agg(pl.count("party").alias("count"))
    .filter(
        (pl.col("party") == "Anti-Administration")
        | (pl.col("party") == "Pro-Administration")
    )
    .sort("count", descending=True)
    .limit(5)
)

df = q.collect()
print(df)
# %%
qq = (
    dataset.lazy()
    .group_by("state", "party")
    .agg(pl.count("party").alias("count"))
    .filter(
        (pl.col("party") == "Anti-Administration")
        | (pl.col("party") == "Pro-Administration")
    )
    .sort("count", descending=True)
    .limit(5)
)

qq.show_graph()
# %%
def get_person() -> pl.Expr:
    return pl.col("first_name") + pl.lit(" ") + pl.col("last_name")


q = (
    dataset.lazy()
    .sort("birthday", descending=True)
    .group_by("state")
    .agg(
        get_person().first().alias("youngest"),
        get_person().last().alias("oldest"),
        get_person().sort().first().alias("alphabetical_first"),
        pl.col("gender")
        .sort_by(pl.col("first_name").cast(pl.Categorical("lexical")))
        .first(),
        get_person().str.len_chars().alias("list_of_len"),
        get_person().str.len_chars().arg_max().alias("len_argmax"),
        get_person().str.len_chars().get(
            get_person().str.len_chars().arg_max()
            ).alias("longest_len"),
        get_person().get(
            get_person().str.len_chars().arg_max()
            ).alias("longest_name")
    )
    .sort("state")
    .limit(5)
)

df = q.collect()
print(df)
# %%
import polars as pl

# then let's load some csv data with information about pokemon
df = pl.read_csv(
    "https://gist.githubusercontent.com/ritchie46/cac6b337ea52281aa23c049250a4ff03/raw/89a957ff3919d90e6ef2d34235e6bf22304f3366/pokemon.csv"
)
print(df.head())
# %%
out = df.select(
    "Type 1",
    "Type 2",
    pl.col("Attack").mean().over("Type 1").alias("avg_attack_by_type"),
    pl.col("Defense")
    .mean()
    .over(["Type 1", "Type 2"])
    .alias("avg_defense_by_type_combination"),
    pl.col("Attack").mean().alias("avg_attack"),
)
print(out)
# %%
filtered = df.filter(pl.col("Type 2") == "Psychic").select(
    "Name",
    "Type 1",
    "Speed",
)
print(filtered)

out = filtered.with_columns(
    pl.col("Name", "Speed").sort_by("Speed", descending=True).over("Type 1"),
)
print(out)

out2 = filtered.sort("Type 1").with_columns(
    pl.col("Name", "Speed").sort_by("Speed", descending=True).over("Type 1"),
)
print(out2)
# %%
# aggregate and broadcast within a group
# output type: -> Int32
pl.sum("foo").over("groups")

# sum within a group and multiply with group elements
# output type: -> Int32
(pl.col("x").sum() * pl.col("y")).over("groups")

# sum within a group and multiply with group elements
# and aggregate the group to a list
# output type: -> List(Int32)
(pl.col("x").sum() * pl.col("y")).over("groups", mapping_strategy="join")

# sum within a group and multiply with group elements
# and aggregate the group to a list
# then explode the list to multiple rows

# This is the fastest method to do things over groups when the groups are sorted
(pl.col("x").sum() * pl.col("y")).over("groups", mapping_strategy="explode")
# %%
out = df.sort("Type 1").select(
    pl.col("Type 1").head(3).over("Type 1", mapping_strategy="explode"),
    pl.col("Name")
    .sort_by(pl.col("Speed"), descending=True)
    .head(3)
    .over("Type 1", mapping_strategy="explode")
    .alias("fastest/group"),
    pl.col("Name")
    .sort_by(pl.col("Attack"), descending=True)
    .head(3)
    .over("Type 1", mapping_strategy="explode")
    .alias("strongest/group"),
    pl.col("Name")
    .sort()
    .head(3)
    .over("Type 1", mapping_strategy="explode")
    .alias("sorted_by_alphabet"),
)
print(out)
# %%

# NOTE more than 3 rows per Type 1 ... the lists are joined to all the rows
out = df.sort("Type 1").select(
    pl.col("Type 1").head(3).over("Type 1", mapping_strategy="join"),
    pl.col("Name")
    .sort_by(pl.col("Speed"), descending=True)
    .head(3)
    .over("Type 1", mapping_strategy="join")
    .alias("fastest/group"),
    pl.col("Name")
    .sort_by(pl.col("Attack"), descending=True)
    .head(3)
    .over("Type 1", mapping_strategy="join")
    .alias("strongest/group"),
    pl.col("Name")
    .sort()
    .head(3)
    .over("Type 1", mapping_strategy="join")
    .alias("sorted_by_alphabet"),
)
print(out)

# %%
weather_by_day = pl.DataFrame(
    {
        "station": ["Station " + str(x) for x in range(1, 11)],
        "day_1": [17, 11, 8, 22, 9, 21, 20, 8, 8, 17],
        "day_2": [15, 11, 10, 8, 7, 14, 18, 21, 15, 13],
        "day_3": [16, 15, 24, 24, 8, 23, 19, 23, 16, 10],
    }
)
print(weather_by_day)

rank_pct = (pl.element().rank(descending=True) / pl.col("*").count()).round(2)
rank_num = (pl.element().rank(descending=True))

out = weather_by_day.with_columns(
    # create the list of homogeneous data
    pl.concat_list(pl.all().exclude("station")).alias("all_temps")
).select(
    # select all columns except the intermediate list
    # pl.all().exclude("all_temps"),
    pl.all(),
    # compute the rank by calling `list.eval`
    pl.col("all_temps").list.eval(rank_pct, parallel=True).alias("temps_rank_pct"),
    pl.col("all_temps").list.eval(rank_num, parallel=True).alias("temps_rank_num"),
)

print(out)
# %%
df = pl.DataFrame(
    {
        "keys": ["a", "a", "b", "b"],
        "values": [10, 7, 1, 23],
    }
)
print(df)

import math


def my_log(value):
    return math.log(value)


out = df.select(pl.col("values").map_elements(my_log, return_dtype=pl.Float64))
print(out)

def diff_from_mean(series):
    # This will be very slow for non-trivial Series, since it's all Python
    # code:
    total = 0
    for value in series:
        total += value
    mean = total / len(series)
    return pl.Series([value - mean for value in series])


# Apply our custom function to a full Series with map_batches():
out = df.select(pl.col("values").map_batches(diff_from_mean))
print("== select() with UDF ==")
print(out)

# Apply our custom function per group:
print("== group_by() with UDF ==")
out = df.group_by("keys").agg(pl.col("values").map_batches(diff_from_mean))
print(out)
# %%
from numba import guvectorize, int64
import polars as pl

@guvectorize([(int64[:], int64[:], int64, int64[:])], '(n),(n),()->(n)', nopython=True)
def make_c(a,b,init_c, res):
    res[0]=(1+a[0]) * init_c * b[0] + (1+b[0]) * init_c * a[0]
    for i in range(1,a.shape[0]):
        res[i] = (1+a[i]) * res[i-1] * b[i] + (1+b[i]) * res[i-1] * a[i]
        
df = pl.DataFrame(((2, 3, 4, 5, 8), (3, 7, 4, 9, 2)), schema=('a', 'b'))

df.with_columns(
    c=make_c(pl.col('a'), pl.col('b'), 3)
)
# %%
rating_series = pl.Series(
    "ratings",
    [
        {"Movie": "Cars", "Theatre": "NE", "Avg_Rating": 4.5},
        {"Movie": "Toy Story", "Theatre": "ME", "Avg_Rating": 4.9},
    ],
)
print(rating_series)
# %%
df_customers = pl.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
    }
)
print(df_customers)

df_orders = pl.DataFrame(
    {
        "order_id": ["a", "b", "c"],
        "customer_id": [1, 2, 2],
        "amount": [100, 200, 300],
    }
)
print(df_orders)

test_join = df_customers.join(df_orders, 
                left_on = "customer_id", right_on="customer_id",
                how="inner")
test_join
# %%
df = (
    pl.date_range(
        start=date(2021, 1, 1),
        end=date(2021, 12, 31),
        interval="1d",
        eager=True,
    )
    .alias("time")
    .to_frame()
)

out = df.group_by_dynamic("time", every="1mo", period="1mo", closed="left").agg(
    pl.col("time").cum_count().reverse().head(3).alias("day/eom"),
    ((pl.col("time") - pl.col("time").first()).last().dt.total_days() + 1).alias(
        "days_in_month"
    ),
    ((pl.col("time") - pl.col("time").first()).alias(
        "time minus first"
    )),
    ( (pl.col("time").max().dt.day().alias("max")))
)
print(out)
# %%
lf = pl.LazyFrame({"foo": ["a", "b", "c"], "bar": [0, 1, 2]}).with_columns(
    pl.col("bar").round(0)
)

try:
    print(lf.collect())
except Exception as e:
    print(e)
# %%
import polars as pl
import numpy  as np
from string import ascii_letters
from numba import jit

@jit(nopython=True, nogil=True)
def numba_func(a, b, c):
    return a+b+c

@jit(nopython=True, nogil=True)
def func_tuple(arg_tuple):
    a, b, c = arg_tuple
    return a+b+c

def python_func(a, b, c):
    return a+b+c

def series_to_numba(x):
    a, b, c = map(lambda x: x.to_numpy(), x.to_frame().unnest("a").get_columns())
    return pl.Series("a", numba_func(a, b, c))

df = pl.DataFrame({
    "a": np.random.randint(0, 1000000, 1000000),
    "b": np.random.randint(0, 1000000, 1000000),
    "c": np.random.randint(0, 1000000, 1000000)})
df.select(pl.col("b"), pl.struct("a", "b", "c").map_batches(series_to_numba))
df.select(pl.struct("a", "b", "c").map_elements(
    lambda x: numba_func(x["a"], x["b"], x["c"])))
# %%
data = {"name": ["Alice", "Bob", "Charlie", "David"], "age": [25, 30, 35, 40]}
df = pl.LazyFrame(data)

ctx = pl.SQLContext(my_table=df, eager=True)

result = ctx.execute(
    """
    CREATE TABLE older_people
    AS
    SELECT * FROM my_table WHERE age > 30
"""
)

print(ctx.execute("SHOW TABLES"))

print(ctx.execute("SELECT * FROM older_people"))
# %%
df = pl.DataFrame(

    {

        "a": ["x", "y", "z"],

        "b": [1, 3, 5],

        "c": [2, 4, 6],

    }

)

import polars.selectors as cs

df.unpivot(cs.numeric())
# %%
df = pl.DataFrame(

    {

        "before": ["foo", "bar"],

        "t_a": [1, 2],

        "t_b": ["a", "b"],

        "t_c": [True, None],

        "t_d": [[1, 2], [3]],

        "after": ["baz", "womp"],

    }

).select("before", pl.struct(pl.col("^t_.$")).alias("t_struct"), "after")

df
# %%
df1 = pl.DataFrame(
    {
        "foo": [1, 2, 3],
        "bar": [6.0, 7.0, 8.0],
        "ham": ["a", "b", "c"],
    }
)

df2 = pl.DataFrame(
    {
        "foo": [3, 2, 1],
        "bar": [8.0, 7.0, 6.0],
        "ham": ["c", "b", "a"],
    }
)
df3 = pl.DataFrame(
    {

        "bar": [6.0, 7.0, 8.0],
        "ham": ["a", "b", "c"],
        "foo": [1, 2, 3]
    }
)
df4 = pl.DataFrame(
    {
        "ham": [1, 2, 3],
        "bar": [6.0, 7.0, 8.0],
        "foo": ["a", "b", "c"],
    }
)

df5 = df1.clone()
df6 = pl.DataFrame(
    {
        "foo": [1, 2, 3],
        "bar": [6.0, 7.0, 8.0],
        "ham": ["a", "b", "c"],
    }
)
print(df1.equals(df1))
print(df1.equals(df2))
print(df1.equals(df3))
print(df1.equals(df4))
print(df1.equals(df5))
print(df1.equals(df6))

df7 = pl.DataFrame(
    {
        "foo": [1, 2, 3],
        "bar": [6.0, 7.0, 8.0],
        "ham": ["ab cd ee", "bc d", "cdd s"],
    }
)
df6_blah = df6.with_columns(pl.col("ham").str.split(" "))
df6
df6_blah

# %%
s = pl.Series("a", [1, 3, 4, None])

print(s.nan_max())
s = pl.Series("a", [1.0, float("nan"), 4.0])

print(s.nan_max())

s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])

print(s.list.head(1))
print(s.list.first())
# %%
s = pl.Series("values", [[-1, 0, 1], [1, 10]])

s.list.std(ddof=2)

s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])

s.list.slice(1, 1)
# %%
s = pl.Series([{"a": 1, "b": 2}, {"a": 3, "b": 4}])

# multiple only works the expresion version atm? not the Series version
print(s.struct.field(["a","b"]))
print(s.struct.field("a","b"))
s.struct.schema
# %%
s = pl.Series("a", [1, 2, 2,3])

(s == 2).arg_true()
# %%
s = pl.Series("a", [1, 2, 3,3,3,3,3,2,2,2,1])

s.gather((s == 2).arg_true())
s.gather(s.arg_unique())
# %%
s = pl.Series(["x", "k", None, "d"])

print(s.cum_count())
print(s.cum_count(reverse=True))
# %%
s = pl.Series("s", [3, 5, 1])

print(s.cum_max())
print(s.cum_max(reverse=True))

s = pl.Series("a", [1, 2, 3])

print(s.cum_prod())
print(s.cum_prod(reverse=True))
# %%
s = pl.Series("s", values=[20, 10, 30, 25, 35], dtype=pl.Int8)

s.diff(-2)
# %%
from numpy import nansum

s = pl.Series([11.0, 2.0, 9.0, float("nan"), 8.0])

def my_func(a):
    return pl.Series.exp(a).sum()
def my_func2(a):
    return a

print(s.rolling_map(nansum, window_size=3))

print(s.rolling_map(sum, window_size=3))

print(s.rolling_map(my_func, window_size=3))
print(s.rolling_map(my_func2, window_size=3))
# %%
s = pl.Series("a", [1.0, None, np.inf])

s.is_finite()
# %%
s = pl.Series("id", ["a", "b", "b", "c", "c", "c","b","b","a"])

s.unique_counts()
# %%
s = pl.Series(
    "lyrics",
    [
        "Everybody wants to rule the world",
        "Tell me whatyou want, what you really really want",
        "Can you feel the love tonight",
    ],
)

s.str.contains_any(["you", "me"])
# %%
s = pl.Series(
    name="url",
    values=[
        "http://vote.com/ballon_dor?candidate=messi&ref=python",
        "http://vote.com/ballon_dor?candidate=weghorst&ref=polars",
        "http://vote.com/ballon_dor?error=404&ref=rust",
    ],
)

test_extr = s.str.extract_groups(r"candidate=(?<candidate>\w+)&ref=(?<ref>\w+)")
print(test_extr)
print(test_extr.struct.fields)
# %%
s = pl.Series("values", ["discontentdisco"])

patterns = ["disco", "onte"]
patterns_2 = ["onte","disco"]
print(s.str.extract_many(patterns, overlapping=True))
print(s.str.extract_many(patterns_2, overlapping=True))
# %%
s = pl.Series("weather", ["Foggyfoggy", "Rainy rainy ", "Sunny  foggy"])

print(s.str.replace_all(r"(?i)foggy|rainy", "Sunny"))

# only replace if beginning of word and end of string
print(s.str.replace_all(r"(?i)(\bfoggy$)|rainy", "Sunny"))
print(s.str.split("y", inclusive=True))
# %%
df = pl.DataFrame({"x": ["a_1_2", None, "c", "d_4"]})

print(df["x"].str.split_exact("_", 1).alias("fields"))
print(df["x"].str.splitn("_", 2).alias("fields"))
# %%
start = datetime(2001, 1, 1)

stop = datetime(2001, 1, 1, 2)

s = pl.datetime_range(start, stop, "15m", eager=True).alias("datetime")

s.dt.round("30m")
# %%
s = pl.Series("a", [1, 2, 3, None,None,5])

s.fill_null(strategy="forward", limit=1)
# %%
s = pl.Series("foo", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10
                      ])

square = s.reshape((5, 2))
square
# %%
s = pl.Series("a", [1, 2, 3, 4, 5])

print(s.new_from_index(1, 3))
print(s.slice(1,3))
# %%
s = pl.Series("a", [1, 2, 3])

s.set(s == 2, 10)
s
# %%
df = pl.DataFrame(
    {
        "group": [
            "one",
            "one",
            "one",
            "two",
            "two",
            "two",
        ],
        "value": [94, 95, 96, 97, 97, 99],
    }
)
print(df.group_by("group", maintain_order=True).agg(pl.col("value").agg_groups()))
print(df.group_by("group", maintain_order=True).agg(pl.col("value").implode()))

# %%
print(df.group_by("group", maintain_order=True).agg(pl.col("value").arg_max()))
print(df.group_by("group", maintain_order=True).agg(pl.col("value").arg_min()))

# %%
df = pl.DataFrame({"a": [0, 1, 2, 3, 4, 5]})
print(df.select(pl.col("a").quantile(0.3)))
print(df.select(pl.col("a").quantile(0.3, interpolation="higher")))
print(df.select(pl.col("a").quantile(0.3, interpolation="lower")))
print(df.select(pl.col("a").quantile(0.3, interpolation="midpoint")))
print(df.select(pl.col("a").quantile(0.3, interpolation="linear")))
# %%
df = pl.DataFrame([
    pl.Series('thing', ['cat', 'plant', 'mouse', 'dog', 'sloth', 'zebra', 'shoe','cat']),
    pl.Series('isAnimal', ['sloth', None, 'sloth', None, 'zebra', 'sloth', None, None]),
    pl.Series('some_range', np.arange(8)*10)
])
test_bunch = df.with_row_index(name="row_index").with_columns(
    pl.col("thing").map_elements(
    lambda x: pl.Series([x])).alias("test_map_elem")
    ).with_columns(
    pl.col("thing").is_in(pl.col("test_map_elem")).alias("test_in_1"),
    pl.col("isAnimal").is_in(pl.col("test_map_elem")).alias("test_in_2"),
    pl.col("isAnimal").is_in(pl.col("test_map_elem").explode()).alias("test_in_3")  ,
    pl.col("test_map_elem").explode().alias("test_explode"),
    pl.col("test_map_elem").explode().implode().alias("test_explode_implode"),
    pl.col("test_map_elem").implode().alias("test_implode"),
    pl.col("test_map_elem").implode().explode().alias("test_implode_explode"),
    pl.col("thing").implode().alias("test_implode_scalar_String"),
    # pl.col("thing").implode().over("thing").alias("test_implode_over"),
    pl.col("some_range").sum().over("thing").alias("test_sum_over"),
    pl.col("thing").over("row_index", mapping_strategy='join').alias("test_over_row_idx"),
    pl.col("thing").over("row_index").alias("test_over_row_idx_no_strategy")
    
    # works since only one thing per row index
    ,pl.implode("thing").over("row_index").alias("test_implode_over")
    # doesn't work WITHOUT SPECIFYING mapping_strategy as there are multiple
    # row index's per thing, that is, for row_index = 0 and 7 where the 
    # thing 'cat' has muiltiple rows
    # implode applied after values are aggregated into list from .over, meaning
    # that the implode causes the list to be wrapped around another []
    ,pl.implode("row_index").over("thing", mapping_strategy='join').alias("test_implode_over_2")
    ,pl.implode("thing").rolling("row_index", period='3i').alias("moving_average_3i")
    # note how offset argument is misleading, 1i moves start of window forward 2?
    ,pl.implode("thing").rolling("row_index", period='3i', offset='1i').alias("moving_average_3i_1i")  
    ,pl.implode("thing").rolling("row_index", period='3i', offset='-2i').alias("moving_average_3i_-2i")   
    # .over('isAnimal') not allowed after the .rolling statement?
    # rolling expression not allowed in aggregation
    ,pl.col("some_range").max().rolling(index_column="row_index",
                                period='2i',offset='-2i').alias("test_max_rolling_over_1")
    ,pl.col('isAnimal').sort(nulls_last=False).alias('isAnimal_Sort')
    )
print(test_bunch)

test_rolling = (
    df.rolling(index_column='some_range', 
                          period='30i',offset='-20i')
    .agg(
        pl.all()
        ,pl.col("thing").alias("thing_Explicit")
        ,pl.col("some_range").alias("some_Range_explicit")
        ,pl.col("some_range").sum().alias("some_range_sum")
        ,pl.col("some_range").implode().alias("some_range_implode")
        ,pl.col("thing").str.len_chars().alias("hmm")
    ))
print(test_rolling)

test_bunch.write_excel("test_bunch.xlsx", worksheet="test_sheet")

print(df.with_columns(pl.col("isAnimal").is_in(pl.col("thing")).alias("test_in")))

# %%
df = pl.DataFrame({
    "a": [1, 2, 3],
    "b": [4, 4, 6],
})

# << over >>
df.with_columns(
    # is this the expected behavior?? useless? Maybe this should already create a list like in agg context?
    a_over_b=pl.col("a").over("b"),

    # crashes! (imo either this or better the above should work and create a list)
    # a_implode_over_b=pl.col("a").implode().over("b"),

    # I want this but over b not over all
    a_implode=pl.col("a").implode(),
)

# << groupby + agg >>
df.group_by("b").agg(
    # plain col("a") already creates a list in agg context (makes sense!)
    agg_col=pl.col("a"),

    # implode creates an additional list
    agg_col_imploded=pl.col("a").implode(), # implode adds another list
)

df.with_columns(goal =
    pl.col("a").over("b", mapping_strategy="join")
)
# %%
data = {'type': ['A', 'A', 'C', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'A'],
        'value': [5, 9, 21, 0, 3, 2, 5, 8, 9, 1, 0, 3, 3, 1, 1, 0, 2, 0, 0, 5, 7, 4, 7, 8, 9, 11, 1, 1, 0, 1, 4, 3, 1]}
df = pl.DataFrame(data)
print(df)

# sum the last 5 seen types (in the provided table order, where each
# of these 5 can be many rows ago)
# that match the given row 'type', only if there have been 5.
rolling_1 = df.with_columns(result = 
   pl.col("value").rolling_sum(window_size=5).over("type")
    )
print(rolling_1)

# in the last 5 rows only, filter down for the current row 'type'
# only sum if there have been 5 of the same 'type' seen
print(df.rolling(pl.int_range(pl.len()).alias("index"), 
           period="5i", group_by="type"))
rolling_2 = (
    df
    .rolling(pl.int_range(pl.len()).alias("index"), period="5i", group_by="type")
    .agg(
        pl.col("value").alias("window"),
        pl.col("value").last(),
        pl.when(pl.len() == 5).then(pl.col("value").sum()).alias("result"),
    )
    # .drop("index")
)
print(rolling_2)

# this is same as rolling_2 above
rolling_3 = (
    df.with_row_index(name="index")
    .rolling(index_column="index", period="5i", group_by="type")
    .agg(
        pl.col("value").alias("window"),
        pl.col("value").last(),
        pl.when(pl.len() == 5).then(pl.col("value").sum()).alias("result"),
    )
    # .drop("index")
)
print(rolling_3)

from xlsxwriter import Workbook

# rolling_1.write_excel(workbook="rolling_examples_2.xlsx", worksheet="rolling_1")
with Workbook("rolling_examples_.xlsx") as wb:  
    rolling_1.write_excel(workbook=wb, worksheet="rolling_1")
    rolling_2.write_excel(workbook=wb, worksheet="rolling_2")
# %%
df = pl.DataFrame(
    {
        "a": [1, 2, 3, 4, 5, 6],
        "b": [6, 5, 4, 3, 2, 1],
        "c": ["Apple", "Orange", "Apple", "Apple", "Banana", "Banana"],
    }
)
df
print(df.select(
    pl.all()
    .bottom_k_by(["c", "a"], 2, reverse=[False, True])
    .name.suffix("_by_ca"),
    pl.all()
    .bottom_k_by(["c", "b"], 2, reverse=[False, True])
    .name.suffix("_by_cb"),
))
# print((
#     df.group_by("c", maintain_order=True)
#     .agg(pl.all().bottom_k_by("a", 2))
#     .explode(pl.all().exclude("c"))
# ))
# %%
(df.group_by("c")
 .agg(
     pl.all().bottom_k_by(by="a",k=2)
 ).explode("a","b"))
# %%
df = pl.DataFrame(
    {
        "a": ["x", "y", "z"],
        "n": [1, 2, 3],
    }
)
print(df.select(pl.col("a").repeat_by("n")))
print(df.select(pl.col("a").repeat_by(5)))
# %%
df = pl.DataFrame(
    {
        "group": ["a", "b", "b"],
        "values": [[1, 2], [2, 3], [4]],
    }
)
print(df.group_by("group").agg(pl.col("values")))
print(df.group_by("group").agg(pl.col("values").flatten())  )
# %%


list_of_structs = [
    [{"a":1, "b": 2}, {"a":3, "b": 4}, {"a":5, "b": 4}],
    [{"a":2, "b": 5}, {"a":3, "b": 5}, {"a":4, "b": 3}]
]
        
df = pl.DataFrame({"structures": list_of_structs})
df
(df.select("structures")
   .with_row_index()
   .explode("structures")
   .unnest("structures")
   .group_by("index", "b", maintain_order=True) 
   .agg(
      pl.struct("a", "b")
   )
   .drop("b")
   .group_by("index", maintain_order=True)
   .all()
)

# %%
df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2]})
df.with_columns(
    rank=pl.concat_list("a", "b").list.eval(pl.element().rank())
)
df.with_columns(
    rank=pl.concat_list("a", "b").list.eval(pl.element())
)
df.with_columns(
    rank=pl.concat_list("a", "b").list.eval(pl.element()/pl.element().sum())
)
# df.with_columns(
#     rank=pl.concat_list("a", "b").list.join(separator=", ")
# )

# same performance between these two apparently
df.with_columns(
    rank=pl.concat_list("a", "b").cast(pl.List(pl.String))
)
df.with_columns(
    rank=pl.concat_list("a", "b").list.eval(pl.element().cast(pl.String))
)
df.with_columns(
    rank=pl.concat_list("a", "b").list.eval(
        pl.when(pl.element()==3).then(30).otherwise(10)
))

df.with_columns(
    rank=pl.concat_list("a", "b").list.eval(
        pl.when(pl.element().sum() == 13)
        .then(pl.element().replace(8, 28)
            
        # pl.element() is current NOT a list at this point  
        # but rather still int64 
        #   .list.drop_nulls()
        # .concat(pl.element().cast(pl.List(pl.Int64))
        )

        .otherwise(pl.element())
    )
    # now it is a list after exiting the .eval
    .drop_nulls()
    ,
    # the boolean is a scalar in a list, so
    # explode it out
    eval_test=
        pl.concat_list("a", "b").list.eval(
        pl.element().sum() == 13).explode()
    ,
    rank_2=pl.when(
        pl.concat_list("a", "b").list.eval(
        pl.element().sum() == 13).explode())
        .then(
            
        pl.concat_list("a", "b").list.eval(            
            pl.element().replace(8, 28))
        .list.concat(
            pl.lit([7,65])
            )
            # pl.element().cast(pl.List(pl.Int64)))
        )
            
        # pl.element() is current NOT a list at this point  
        # but rather still int64 
        #   .list.drop_nulls()
        # .concat(pl.element().cast(pl.List(pl.Int64))
        
        .otherwise(pl.col("a")
                   .cast(pl.List(pl.Int64))
                   .list.concat(pl.lit([65,76])
                   ))
    ,
    rank_3 = pl.col("a").cast(pl.List(pl.Int64))
                   .list.concat(pl.lit([65,76])
                   )
    ,
    rank_4 = pl.col("a").implode()
    
)
# %%
c = 2

df = pl.DataFrame({"a": [1, 2, 3, 1, 2, 3], "b": [1, 1, 1, 2, 2, 2]})

df.group_by("b").agg(pl.lit(c).pow(pl.col("a")).sum())
# %%
df.select(pl.col("a")).to_series().is_sorted()

# %%
def extract_number(expr: pl.Expr) -> pl.Expr:
    """Extract the digits from a string."""
    return expr.str.extract(r"\d+", 0).cast(pl.Int64)
def scale_negative_even(expr: pl.Expr, *, n: int = 1) -> pl.Expr:
    """Set even numbers negative, and scale by a user-supplied value."""
    expr = pl.when(expr % 2 == 0).then(-expr).otherwise(expr)
    return expr * n
df = pl.DataFrame({"val": ["a: 1", "b: 2", "c: 3", "d: 4"]})
print(df.with_columns(
    udfs=(
        pl.col("val").pipe(extract_number).pipe(scale_negative_even, n=5)
    ),
))
print(df.with_columns(
    udfs=(
        pl.col("val").pipe(extract_number)
    ),
))
print(df.with_columns(
    udfs=(
        pl.col("val").map_batches(extract_number)
    ),
))

# %%
df = pl.DataFrame(
    {
        "coords": [{"x": 1, "y": 4}, {"x": 4, "y": 9}, {"x": 9, "y": 16}],
        "multiply": [10, 2, 3],
    }
)
df
df = df.with_columns(
    pl.col("coords").struct.with_fields(
        pl.field("x").sqrt(),
        y_mul=pl.field("y") * pl.col("multiply"),
    )
)
df.get_column("coords").struct.fields
df.unnest("coords")
# %%
import polars.selectors as cs
df = pl.DataFrame({"a": ["Apple", "Apple", "Orange"], 
                   "b": [1, None, 2],
                   "c": ["Zebra", "Horse", "Cow"]
                   })
print(df.group_by("a").len()  )
print(df.group_by("a").len(name="n")  )
print(df.group_by("a").agg(pl.all().len()))
print(df.group_by("a").agg(pl.len().alias("n")))
print(df.group_by("a").agg(pl.max(["b","c"])))
print(df.group_by("a").agg(pl.max(["b","c"]), pl.max("a").alias("a_max")))
print(df.group_by("a").all())
print(df.group_by("a").first())
print(df.group_by("a").count())
print(df.group_by("a").max())
print(df.group_by("a").mean())
print(df.group_by("a").n_unique())
print(df.group_by("a").agg(pl.all().n_unique()).select(
    pl.col("a"), 
    pl.sum_horizontal(cs.numeric()).alias("sum_horizontal"),
    pl.struct(cs.numeric())))
# %%
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
    df_height = df_big.select(pl.len()).collect().item()
    train_num = round(df_height * train_fraction)
    test_num = df_height - train_num
    df_train = df.head( train_num )
    df_test = df.tail( test_num )
    
    return df_train, df_test

N = 1_000_000
df_big = pl.DataFrame(
    [
        pl.arange(0, N, eager=True),
        pl.arange(0, N, eager=True),
        pl.arange(0, N, eager=True),
        pl.arange(0, N, eager=True),
        pl.arange(0, N, eager=True),
    ],
    schema=["a", "b", "c", "d", "e"],
).lazy()
train, test = train_test_split_lazy(df_big)

print(train.collect())
print(test.collect())
print(test.select(pl.exclude("index")).collect())

rng = np.random.default_rng(seed=12345)
# train, test = (df_big
#   .with_columns(pl.lit(np.random.rand(df_big.height)>0.8).alias('split'))
#   .partition_by('split'))

# df_big.tail(5)
# df_big.collect().head(-5)
# %%
df = pl.DataFrame(
    {
        "id": ["a123", "b345", "c567", "d789", "e101"],
        "points": [99, 45, 50, 85, 35],
    }
)
df.write_excel(  
    table_style={
        "style": "Table Style Medium 15",
        "first_column": True,
    },
    column_formats={
        "id": {"font": "Consolas"},
        "points": {"align": "center"},
        "z-score": {"align": "center"},
    },
    column_totals="average",
    formulas={
        "z-score": {
            # use structured references to refer to the table columns and 'totals' row
            "formula": "=STANDARDIZE([@points], [[#Totals],[points]], STDEV([points]))",
            "insert_after": "points",
            "return_dtype": pl.Float64,
        }
    },
    hide_gridlines=True,
    sheet_zoom=125,
)

# %%
import xlsxwriter
workbook = xlsxwriter.Workbook('xlsxwriter_vline_chart.xlsx')
worksheet = workbook.add_worksheet()

### Add the worksheet data to be plotted.
### Set column widths
worksheet.set_column(1, 3, 12)

### Common format
common_format = workbook.add_format({
    'border': 1,
})

### Add Headers
Header_data = ['', 'Day', 'New Sales']
cformat = workbook.add_format({
    'bold': True,
    'bg_color': '#92D050',
    'border': 1,
})
worksheet.write_row('B5', Header_data, cformat)

### ID Column
id_data = list(range(1,16))
worksheet.write_column('B6', id_data, common_format)

### Day Column
dates_format = workbook.add_format({
    'num_format': 'd-mmm-yy',
    'border': 1,
})
dates_data = list(range(44774, 44789))
worksheet.write_column('C6', dates_data, dates_format)

### Values Column
values_data = [132, 128, 95, 115, 83, 87, 67, 97, 112, 84, 77, 97, 86, 77, 83]
worksheet.write_column('D6', values_data, common_format)

### Add Date for Vertical line position
vh_format = workbook.add_format({
    'bold': True,
    'border': 1,
})
worksheet.write('C4', 'Vert line Date', vh_format)

vd_format = workbook.add_format({
    'bold': True,
    'num_format': 'd-mmm-yy',
    'font_color': 'red'
})
worksheet.write('D4', 44780, vd_format)

### Create a new chart object [Line Chart].
chart_name = 'Chart Test'
chart = workbook.add_chart({'type': 'line'})

chart.add_series({
    'name': 'Data',
    'values': '=Sheet1!$D$6:$D$20',
    'categories': '=Sheet1!$C$6:$C$20',
    'smooth': True,
})

### Add 2nd series for the Vertical Line [Scatter Chart]
scatter_series = workbook.add_chart({'type': 'scatter'})

### Set vertical line chart data and format
scatter_series.add_series({
    'name': 'VerticalLine',
    'categories': '=Sheet1!$D$4',
    'values': '=Sheet1!$D$6:$D$20',
    'marker': {'type': 'none'},
    'line':   {'color': 'red',
               'width': 1.5,
    },
    'y_error_bars': {
        'type': 'fixed',
        'value': 80,
        'direction': 'minus',
        'line':   {'color': 'red',
                   'width': 1.5,
                   'dash_type': 'dash',
                   }
    },
})

# # Combine both Line and Scatter charts
chart.combine(scatter_series)

### Chart customisation
chart.set_title({'name': 'Line Chart with Dynamic Vertical line'})
chart.set_size({'width': 580})

chart.set_x_axis({
    'name': 'Day',
    'date_axis': True,
    'minor_unit': 1,
    'minor_unit_type': 'days',
    'major_unit': 2,
    'major_unit_type': 'days',
})

chart.set_y_axis({
    'name': 'New Sales',
    'min': 60,
    'max': 140,
})

# Insert the chart into the worksheet.
worksheet.insert_chart('F6', chart, {'x_scale': 2, 'y_scale': 1})

workbook.close()

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
                "name": "Claim_Freq",
                "values": "={}[{}]".format(col, "Claim_Freq"),
                "categories": "={}[{}]".format(col, col),
                "y2_axis": True,
                "line": {'width': 3}
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
        
        ws.insert_chart(18, temp.width + 1 + summary.width + 1, column_chart)
        
        
        
        ### Exposure and Claim Frequency graph - Seaborn
        fig, ax1 = plt.subplots(figsize=(12,6))
        sns.lineplot(data = temp, x = col, y = 'Claim_Freq', 
                    marker='o', sort = False, ax=ax1, label="Claim Frequency")
        sns.lineplot(data = temp, x = col, y = 'average_freq', 
                    linestyle = '--',sort = False, ax=ax1, label="Average Claim Frequency")
        ax2 = ax1.twinx()
        ax2.grid(False)
        sns.barplot(data = temp, x=col, y='Total_Exposure', alpha=0.5, ax=ax2, 
                    label="Exposure")
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.legend()
        ax1.set(ylabel="Claim Frequency")

        image_file = col + ".png"
        fig.savefig(image_file)
        
        # Insert the chart into the worksheet
        ws.insert_image(18, temp.width + 1 + summary.width + 1, image_file, 
                        {"x_scale": 1, "y_scale": 1, "x_offset": 0, "y_offset": 0})
        
# %%
test_cat = pl.Series(['a','b','c','b'], dtype=pl.Categorical)
test_cat.rename("blah")
test_df = pl.DataFrame({"blah":test_cat})
test_df
test_df2 = test_df.with_columns(pl.col("blah").str.replace('a','dd',literal=True).cast(pl.Categorical))

print(test_df2.select(pl.col("blah").cast(pl.Categorical("lexical")).sort()))

sort_cats = test_df2["blah"].cat.get_categories().sort().to_list()
sort_cats
# .cat.set_ordering("lexical") ??? where is the option
test_df2.select(pl.col("blah").cast(pl.Enum(sort_cats))).sort("blah")

myorder = ['c','b','dd']

with pl.StringCache():
    # fill in the order using global string cache
    pl.Series(myorder).cast(pl.Categorical)
    # re-cast column using categorical, ordered by the string cache
    print(test_df2.cast(pl.String).cast(pl.Categorical).sort("blah"))
print(test_df2.cast(pl.String).cast(pl.Categorical).sort("blah"))

test_df2["blah"].cat.uses_lexical_ordering()