# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels as sm
import pyarrow as pa

from datetime import datetime, timedelta
from pandas.tseries.offsets import Hour, Minute, Day, MonthEnd
from pandas.tseries.frequencies import to_offset
from scipy.stats import percentileofscore
# %%
for i in range(10):
    print(i)

# %%
for i in range(10):
    print(-i)
# %%
msg = "Hello World"
print(msg)
print(msg + " message")
# %%
print(msg)
# %%
# This is a comment
print("Testing")
# %%
class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

  def __str__(self):
    return f"{self.name}({self.age})"

  def myfunc(self):
    print("Hello my name is " + self.name)
    
#   def printname(self):
#     print(self.firstname, self.lastname)
    
# Initiate instance of Person
p1 = Person("John", 36)

# invoke myfunc and __str__ functions
p1.myfunc() 
print(p1)

# change class parameters
p1.age = 40
print(p1)

# delete class parameters
del p1.age 
# can't run the __str__ without an age parameter anymore
# print(p1)

# delete the instance of the class
del p1

# %%
# Inherit from the 'Person' class
class Student(Person):
  def __init__(self, fname, lname, year):
    # Specifically defining inheritance of parent, otherwise no inheritance from parent
    # Person.__init__(self, fname, lname)
    
    # Super() is better? more flexible
    super().__init__(self, fname, lname)
    self.graduationyear = year

  def welcome(self):
    print("Welcome", self.firstname, self.lastname, "to the class of", self.graduationyear)
# %%

# Root is a defensive class to make sure to 'absorb' the 'draw' method before it reaches the Object 
# Parent class
class Root:
    def draw(self):
        # the delegation chain stops here
        assert not hasattr(super(), 'draw')

class Shape(Root):
    def __init__(self, shapename, **kwds):
        self.shapename = shapename
        super().__init__(**kwds)
    def draw(self):
        print('Drawing.  Setting shape to:', self.shapename)
        super().draw()

class ColoredShape(Shape):
    def __init__(self, color, **kwds):
        self.color = color
        super().__init__(**kwds)
    def draw(self):
        print('Drawing.  Setting color to:', self.color)
        super().draw()

cs = ColoredShape(color='blue', shapename='square')
cs.draw()
# %%

# This class doesn't cooperate with inheritance hierarchy, without super() calls, and has 
# __init__ function incompatible with 'object' class
class Moveable:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def draw(self):
        print('Drawing at position:', self.x, self.y)

# If we want to use this class with our cooperatively designed ColoredShape hierarchy, 
# we need to make an adapter with the requisite super() calls:
class MoveableAdapter(Root):
    def __init__(self, x, y, **kwds):
        self.movable = Moveable(x, y)
        super().__init__(**kwds)
    def draw(self):
        self.movable.draw()
        super().draw()

class MovableColoredShape(ColoredShape, MoveableAdapter):
    pass

MovableColoredShape(color='red', shapename='triangle',
                    x=10, y=20).draw()
# %%
class Grandparent:
    def __init__(self):
        print("Grandparent")

class Parent1(Grandparent):
    def __init__(self):
        print("Parent1")
        Grandparent.__init__(self)

class Parent2(Grandparent):
    def __init__(self):
        print("Parent2")
        Grandparent.__init__(self)

class Child(Parent1, Parent2):
    def __init__(self):
        print("Child")
        Parent1.__init__(self)
        Parent2.__init__(self)

c = Child() # this causes "Grandparent" to be printed twice!

# %%
class Grandparent:
    def __init__(self):
        print("Grandparent")

class Parent1(Grandparent):
    def __init__(self):
        print("Parent1")
        super().__init__()

class Parent2(Grandparent):
    def __init__(self):
        print("Parent2")
        super().__init__()

class Child(Parent1, Parent2):
    def __init__(self):
        print("Child")
        super().__init__()

c = Child() # no extra "Grandparent" printout this time

# %%

# underscores represent comma in number
print(100_000 + 500)
# %%
testdict = {"a": "c"}

print(testdict.pop("a", "30"))
print(testdict.pop("a", "30"))
testdict
# %%
car = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}

# setdefault():
#   if 1st arg not in dict:
#       return 2nd arg
#       update dict with {1st arg: 2nd arg}
#   if 1st arg in dict:
#       return dict lookup of 1st arg
#       DO NOT update dict 
x = car.setdefault("model", "Bronco")

print(x)
car
# %%
car = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}

x = car.setdefault("color", "white")

print(x) 
car
# %%
# check hashable
hash("string")

from collections import defaultdict
dd = defaultdict(list)
dd["test"] = "testvalue"
print(dd)

print(set([2,2,2,1,3,3]))
test_set = set()
test_set.add(2)
test_set
# %%
list_1 = [5, 3, 4]
list_2 = [10, list_1]
list_2

list_1.append(100)
list_2

list_3 = [500, 600, 700, list_2]
list_3

import copy

list_1 = [5, 3, 4]
list_2 = [10, list_1]
list_3 = [500, 600, 700, list_2]

list_4_copy = copy.copy(list_3)
list_4_deep = copy.deepcopy(list_3)
list_1.append(100)
list_2.append(567)
print(list_4_copy)
print(list_4_deep)

# %%
enumerate([1,2,3])

# list( enumerate( )) doesn't work
print( tuple(enumerate([1,2,3])) )

sorted( (6, 5, 3) )

tuple(map( len, ["strings","not"]))
# %%
states = ["   Alabama ", "Georgia!", "Georgia", 
          "georgia", "FlOrIda",
"south   carolina##", "West virginia?"]

import re
def clean_string(string):
    value = string.strip()
    value = re.sub("[!#?]", "", value)
    value = value.title()
    return value

cleaned_states = map(clean_string, states)
print(list(cleaned_states))

list(
    map( lambda a: "teehee" + a,
    map( clean_string, 
    map(lambda a: a + "blah", 
        states),
    )
    )
)

# %%
import itertools

def first_letter(x):
    return x[0]

names = ["Alan", "Adam", "Wes", "Will", "Albert", "Steven"]
for letter, names in itertools.groupby(names, first_letter):
    print(letter, list(names)) # names is a generator

# %%
path = r"C:\Users\edwin\OneDrive\Desktop\Python Practice\TestCSV.txt"
new_path = path.replace("\\", "/")
f = open(path, "r", encoding="utf-8")
f.tell()

write_path = "TestWrite.txt"

with open(write_path, mode="w") as handle:
    handle.writelines(x for x in open(path) if len(x) > 1)

with open(write_path) as f:
#    lines = f.readlines()
   test_read = f.read()
   
f.close()

# print(lines)
test_read
# %%
data = b'Sue\xc3\xb1a el '
data[:4]

write_path = "TestWrite.txt"
with open(write_path, mode="wb") as handle:
    handle.write(data)
with open(write_path, mode="rb") as handle:
    print(handle.read(5))
with open(write_path, encoding="utf-8") as handle:
    print(handle.read(4))    
with open(write_path, encoding="utf-8") as handle:
    handle.seek(3)
    # handle.seek(4) -- error due to middle byte
    print(handle.read(4))  


path = r"C:\Users\edwin\OneDrive\Desktop\Python Practice\TestCSV.txt"
new_path = path.replace("\\", "/")
f = open(path, "r", encoding="utf-8")
f.tell()

print(f.read(3))
print(f.read(4))
f.seek(0)
print(f.read(3))
print(f.read(4))
# %%
import numpy as np
arr3d = np.array(
    [
        [
            [1, 2, 3], 
            [4, 5, 6],
            [10,20,30],
            [400,500,600]
        ], 
        [
            [7, 8, 9], 
            [10, 11, 12],
            [70,80,90],
            [1000,1100,1200]
        ]
    ])
# shape goes from outside to inside
print(arr3d)
print(arr3d.shape)
# slicing --> see new dimension shape
print(arr3d[1:,3:,1:])
print(arr3d[1:,3:,1:].shape)

# indexing from highest dimension and slicing by the other dimensions 
#   the 2nd highest dimension becomes the new king
print(arr3d[1,3:,1:])
print(arr3d[1,3:,1:].shape)

# indexing from 2nd highest dimension and slicing by the other dimensions
#    this is same as the one above, (same shape and values)
print(arr3d[1:,3,1:])
print(arr3d[1:,3,1:].shape)


full_array_assign = arr3d
slice_array_assign = arr3d[1:,3:,1:]

# arr3d[1:] = 250

# # neither full assigning or slice and then assigning makes a copy of the array in any way
# print(full_array_assign)
# print("##############################################")
# print(slice_array_assign)
# %%
# shape = (2, 4, 3) for axes = (0, 1, 2)
print(arr3d)

# shape = (4, 2, 3)
print("###################")
print(arr3d.swapaxes(0, 1))

# shape = (3, 4, 2)
print("###################")
print(arr3d.swapaxes(0, 2))

# shape = (2, 3, 4)
print("###################")
print(arr3d.swapaxes(1, 2))
# %%

# Sample code for generation of first example 
import numpy as np 
# from matplotlib import pyplot as plt 
# pyplot imported for plotting graphs 
  
x = np.linspace(-4, 4, 9) 
  
# numpy.linspace creates an array of 
# 9 linearly placed elements between 
# -4 and 4, both inclusive  
y = np.linspace(-5, 5, 11) 
  
# The meshgrid function returns 
# two 2-dimensional arrays  
x_1, y_1 = np.meshgrid(x, y) 
  
print("x_1 = ") 
print(x_1) 
print("y_1 = ") 
print(y_1) 
# %%
print(y_1.sum(axis=0))
print(y_1.sum(axis=1))
# %%
sort_array = np.array(
    [  [ 0.936 ,  1.2385,  1.2728],
       [ 0.4059, -0.0503,  0.2893],
       [ 0.1793,  1.3975,  0.292 ],
       [ 0.6384, -0.0279,  1.3711],
       [-2.0528,  0.3805,  0.7554]
    ]
)

print(sort_array)
sort_array.sort(axis=0)
sort_array
sort_array
# %%
arr_unique = np.array([1,1,2,2,2,3])
np.unique(arr_unique)
# %%
rng = np.random.default_rng(seed=12345)

nwalks = 5000
nsteps = 1000
draws = rng.integers(0, 2, size=(nwalks, nsteps)) # 0 or 1
print(draws.shape)
draws.sum(axis=0)
# %%
arr_unique = np.array([[1,1,2,2,2,3],[5,6,7,9,23,25]])
arr_transpose = np.transpose(arr_unique, axes=(0,1))
arr_transpose[1,1] = 360
arr_unique
(arr_unique > 2).any(axis=1)

old_val = arr_unique[1, 1]
print(arr_unique[1:2, 1:2])
print(arr_unique[1:2, 1])
print(arr_unique[1, 1:2])
print(arr_unique[1, 1])
print(old_val)

# slicing doesn't make a copy, but assigning like this without [ ] doesn't do anything
test_assign = arr_unique[:2, :2]
test_assign = 800
# print(arr_unique[1:2, 1:2])
# print(arr_unique[1:2, 1])
# print(arr_unique[1, 1:2])
print(arr_unique)

# slicing doesn't make a copy, and now with the [] it changes the original array
test_assign = arr_unique[:2, :2]
test_assign[0] = 800
print(arr_unique)

print( arr_unique.argmax(axis=1) )

arr_unique[1, 4] = 1600
# traverse through innermost cells until outer to get the index position, when no axis=option
print( arr_unique.argmax() )
# %%
sdata = {"Ohio": 35000, "Texas": 71000, "Oregon": 16000, "Utah": 5000}
obj3 = pd.Series(sdata)
obj3.to_dict()

states = ["California", "Ohio", "Oregon", "Texas"]
obj4 = pd.Series(sdata, index=states)
print(obj4)

print("########")

states2 = ["California", "Ohio", "Oregon", "Texas", "Utah"]
obj5 = pd.Series(sdata, index=states2)
print(obj5)
# %%
print(pd.isna(obj4))
print(pd.notna(obj4))
~pd.isna(obj4)
# %%
data = {"state": ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada", "Nevada"],
        "year": [2000, 2001, 2002, 2001, 2002, 2003],
        "pop": [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)
frame.head()
frame.tail()
pd.DataFrame(data, columns=["year", "state", "pop"])
frame2 = pd.DataFrame(data, columns=["year", "state", "pop", "debt"])
frame2.columns
frame2["state"]
frame2["debt"] = np.arange(6)

val = pd.Series([-1.2, -1.5, -1.7], index=[2, 4, 5])
frame2["debt"] = val
frame2

val = pd.Series([-1.2, -1.5, -1.7])
frame2["debt"] = val
frame2

val = pd.Series(np.arange(5))
frame2["debt"] = val
frame2

# %%
populations = {"Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6},
               "Nevada": {2001: 2.4, 2002: 2.9}
               }
frame3 = pd.DataFrame(populations)
print(frame3)
print(pd.DataFrame(populations, index=[2001, 2002, 2003]))

pdata = {"Ohio": frame3["Ohio"][:-1],
        "Nevada": frame3["Nevada"][1:]}
print(frame3["Ohio"][:-1])
print(frame3["Nevada"][1:])
# dictionary of Series --> goes off index when combining using DataFrame
print(pd.DataFrame(pdata))

# get columns of df
frame3.columns.values.tolist()

# assign index to a series
labels = pd.Index(np.array([6,2,4]))
obj2 = pd.Series([1.5,-2.5,0], index=labels)
obj2

# row number indexing
frame3.iloc[[0,1]]

# row label indexing
frame3.loc[[2000,2002]]
frame3.loc[2000]

# generates error without .index
frame3.loc[frame3["Ohio"].index]

# boolean doesn't need .index?
frame3.loc[frame3["Ohio"] > 1.3]

# reindex makes a copy, assigning doesn't change original
test_reindex = frame3.reindex(labels=[2001], columns=["Ohio"])
test_reindex["Ohio"] = 34
print(frame3)

# .loc creates a view, so changes do propogate to original, with single or multiple selections
test_loc = frame3.loc[2001]
test_loc["Ohio"] = 56
print(frame3)

test_loc = frame3.loc[[2001]]
test_loc["Ohio"] = 56
print(frame3)
# %%
frame3.to_json("TestJsonOut.json")
frame3.to_json("TestJsonOutOrientRecords.json", orient='records')
frame3.to_json("TestJsonOutOrientIndex.json", orient='index')
frame3.to_json("TestJsonOutOrientColumns.json", orient='columns')
# frame3
# %%
s1 = pd.Series([0, 1], index=["a","b"], dtype="Int64")
pd.concat([s1,s1,s1])
# %%
path = "pydata-book-3rd-edition\examples\macrodata.csv"
data = pd.read_csv(path)

data = data.loc[:, ["year", "quarter", "realgdp", "infl", "unemp"]]

data.head()

periods = pd.PeriodIndex(year=data.pop("year"),
                         quarter=data.pop("quarter"),
                        name="date")

data.index = periods.to_timestamp("D")

data.head()
data = data.reindex(columns=["realgdp", "infl", "unemp"])
data.columns.name = "item"
data.head()

data.stack().reset_index().rename(columns={0: "value"})
# %%
import matplotlib.pyplot as plt

data = np.arange(10)

plt.plot(data)
# %%
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax1.hist(np.random.standard_normal(100), bins=20, color="black", alpha=0.3);
ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.standard_normal(30));
help(plt.plot)
# %%
df = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
subset = df["foo"]
subset.iloc[0] = 100
df


# %%
df = pd.DataFrame({"a": [1, 2], "b": [1.5, 2.5]})
df2 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

# Returns copy ?
arr = df.to_numpy()
# Returns view ? Shares data with df2?
arr2 = df2.to_numpy()
# Returns a copy this time
arr3 = df2.to_numpy().copy()

arr[0,0] = 100
print(df)

arr2[0,0] = 100
print(df2)

# arr3[0,0] = 100
# print(df2)
# %%
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

df2 = df.reset_index(drop=True)

df2.iloc[0, 0] = 100

print(df)
print(df2)
# %%
print(pd.__version__)
# %%
dict(facecolor="black", headwidth=4, width=2,
                                headlength=4)
# %%
tips = pd.read_csv("pydata-book-3rd-edition/examples/tips.csv")
tips.head()
party_counts = pd.crosstab(tips["day"], tips["size"])
party_counts = party_counts.reindex(index=["Thur", "Fri", "Sat", "Sun"])

party_counts = party_counts.loc[:, 2:5]

test_sum = party_counts.sum(axis="columns")
print(party_counts)
print(test_sum)

party_pcts = party_counts.div(party_counts.sum(axis="columns"),
                            axis="index")
party_pcts
# %%
import seaborn as sns
tips["tip_pct"] = tips["tip"] / (tips["total_bill"] - tips["tip"])
tips.head()
sns.barplot(x="tip_pct", y="day", data=tips, orient="h")

# %%
sns.set_style("whitegrid")
sns.barplot(x="tip_pct", y="day", data=tips, orient="h")
# %%
sns.set_palette("Greys_r")
sns.barplot(x="tip_pct", y="day", data=tips, orient="h")
# %%
sns.set_palette(sns.color_palette("tab10"))
sns.barplot(x="tip_pct", y="day", hue="time", data=tips, orient="h")
# %%
comp1 = np.random.standard_normal(200)
comp2 = 10 + 2 * np.random.standard_normal(200)
values = pd.Series(np.concatenate([comp1, comp2]))
sns.histplot(values, bins=100, color="black", kde=True)
# %%
x = np.array([1, 2, 4, 7, 0])
np.diff(x)
# equivalent to np.diff(np.diff(x))
np.diff(x, n=2)
np.diff(np.diff(x))
# %%
def get_stats(group: pd.DataFrame) -> pd.DataFrame :
    return pd.DataFrame(
        {"min": group.min(), "max": group.max(),
        "count": group.count(), "mean": group.mean()}
    )    

# %%
suits = ["H", "S", "C", "D"]  # Hearts, Spades, Clubs, Diamonds
card_val = (list(range(1, 11)) + [10] * 3) * 4
base_names = ["A"] + list(range(2, 11)) + ["J", "K", "Q"]
cards = []
for suit in suits:
    cards.extend(str(num) + suit for num in base_names)

deck = pd.Series(card_val, index=cards)
deck.head(13)
# %%
df = pd.DataFrame({"category": ["a", "a", "a", "a",
                                "b", "b", "b", "b"],
                   "data": np.random.standard_normal(8),
                   "weights": np.random.uniform(size=8)})

grouped = df.groupby("category")
def get_wavg(group):
    return np.average(group["data"], weights=group["weights"])
# THIS IS NOT AVERAGE OF "data" * "weights",
# but rather the weights assigned are normalised first across each group,
# or alternatively the AVERAGE OF "data" * "weights" / (SUM OF GROUP WEIGHTS)
grouped.apply(get_wavg)
# %%
from io import StringIO
test_str = """Sample  Nationality  Handedness
 1   USA  Right-handed
 2   Japan    Left-handed
 3   USA  Right-handed
 4   Japan    Right-handed
 5   Japan    Left-handed
 6   Japan    Right-handed
 7   USA  Right-handed
 8   USA  Left-handed
 9   Japan    Right-handed
 10  USA  Right-handed"""

data = pd.read_table(StringIO(test_str), sep="\s+")
data

# equivalents, replace len with "count" if want to exclude NULL
pd.crosstab(data["Nationality"], data["Handedness"], margins=True)
pd.pivot_table(data=data, index=["Nationality"], columns=["Handedness"], 
               aggfunc=len)
data.pivot_table(index=["Nationality"], columns=["Handedness"], 
               aggfunc=len)
# %%
test_cross = pd.crosstab([tips["time"], tips["day"]], 
                         tips["smoker"], margins=True)
test_cross

test_cross / test_cross.sum()
test_cross.apply(sum, axis=0)

test_nparray = np.array([[1,2,3],[4,5,6]])
test_axis_eq0 = pd.DataFrame(test_nparray)
print(test_nparray)
print(test_nparray.sum(axis=0))
print(test_axis_eq0.apply(sum, axis=0))
# %%
stamp = pd.Timestamp("2012-11-04 00:30", tz="US/Eastern")
print(stamp)
print(stamp + 1 * Hour())
print(stamp + 2 * Hour())

# %%
dates = pd.date_range("2000-01-01", periods=6, freq="D")
ts2 = pd.Series(np.random.standard_normal(6), index=dates)
# ts2 = ts2.to_period("M")
pts = ts2.to_period("M")
pts.asfreq("D",how="start")
pts.to_timestamp(how="end")
# %%
dates = pd.date_range("2000-01-01", periods=12, freq="T")
ts = pd.Series(np.arange(len(dates)), index=dates)
ts.resample("5min").sum()
ts.resample("5min", closed="right", label="left").sum()
ts.resample("5min", closed="right", label="right").sum()
ts.resample("5min", closed="left", label="left").sum()
ts.resample("5min", closed="left", label="right").sum()
# %%
import pandas as pd

pd.tseries.frequencies.is_subperiod('M', '3M')
pd.tseries.frequencies.is_subperiod('M', 'Q')
# %%
import patsy
data = pd.DataFrame({
    'x0': [1, 2, 3, 4, 5],
     'x1': [0.01, -0.01, 0.25, -4.1, 0.],
  'y': [-1.5, 0., 3.6, 1.3, -2.]})

y, X = patsy.dmatrices('y ~ I(x0 + x1)', data)


# %%
arr1 = np.array([[1, 2, 3], [4, 5, 6]])

arr2 = np.array([[7, 8, 9], [10, 11, 12]])

# [] vs () -- the same
print(np.concatenate([arr1, arr2], axis=0))
print(np.concatenate((arr1, arr2), axis=0))

print(np.concatenate([arr1, arr2], axis=1))
print(np.concatenate((arr1, arr2), axis=1))
# %%
indexer = [slice(None)] * 3
indexer[1] = np.newaxis
indexer

rng = np.random.default_rng(seed=12345)
arr = rng.standard_normal((3, 4, 5))
arr

def demean_axis(arr, axis=0):
    means = arr.mean(axis)

    print("means = ")
    print(means)
    # This generalizes things like [:, :, np.newaxis] to N dimensions
    
    # one slice for each dimension to index
    indexer = [slice(None)] * arr.ndim
    # create new axis at desired axis
    indexer[axis] = np.newaxis
    # turn the means array into the right shape for broadcasting
    return arr - means[tuple(indexer)]

print(demean_axis(arr, 1))
print("###")
print(arr)
arr.mean(axis=1)
# means = arr.mean(axis=1)
# indexer = [slice(None)] * arr.ndim
# indexer = tuple([slice(None)] * arr.ndim)
# indexer

# means
# means[1]

# means[(slice(None), slice(1, 3))]
# arr.ndim

# must be 2d unfortunately
# pd.DataFrame(arr).to_excel("test_demean.xlsx")
# %%
col = np.array([1.28, -0.42, 0.44, 1.6])
# take the new values in a new outer dimension,
# with a new axis set in the inner dimension
col[:, np.newaxis]
col.reshape((4,1))
# %%
data = np.array([[1,1,1],[2,2,2],[3,3,3]])
vector = np.array([1,2,3])

print(data - vector)
print(data - np.transpose(vector))
print(data - vector[: , np.newaxis])
print(data / vector[: , np.newaxis])
# %%
df = pd.DataFrame({"category": ["a", "b", "a", "a",
                                "b", "b", "a", "b"],
                   "data": np.random.standard_normal(8),
                   "weights": np.random.uniform(size=8)})
print(df.index.name)
print(df.index.names)
df.index.name = 'index_name'
print(df.index.name)
print(df.index.names)

grouped = df.groupby("category")
def get_wavg(group):
    return np.average(group["data"], weights=group["weights"])
# THIS IS NOT AVERAGE OF "data" * "weights",
# but rather the weights assigned are normalised first across each group,
# or alternatively the AVERAGE OF "data" * "weights" / (SUM OF GROUP WEIGHTS)
grouped.apply(get_wavg)

df_grouptest = df.groupby("category").transform("sum")
print(df_grouptest)
print(df_grouptest.sum())

print(df_grouptest.shape)
print(df_grouptest.sum().shape)

print(df_grouptest.sum().reset_index(drop=False))
print(df_grouptest.sum().reset_index(drop=True))

# broadcasting normally, AND
# broadcasting using to_numpy and introducing new axis to match shape
# default broadcasting by matching index and broadcasting down the rows
print(df_grouptest - df_grouptest.sum())
print(df_grouptest - df_grouptest.sum().to_numpy()[np.newaxis,:])
print(df_grouptest - df_grouptest.sum().reset_index(drop=True))
# %%
x = np.arange(10)


a = np.arange(3)
b = np.arange(3)[:, np.newaxis]

b[:] = 25
a

# %%
# x and y have 50 steps from 0 to 5
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]

z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

plt.imshow(z, origin='lower', extent=[0, 5, 0, 5],
           cmap='viridis')
plt.colorbar()

print(z)


# %%
test = np.array([np.NaN])
np.sum(test == 0)
np.sum(test != 0)
# %%
# np.random.seed(1234)
rng = np.random.default_rng(seed=1234)

X = rng.uniform(0, 1, (5,2))

import matplotlib.pyplot as plt
import seaborn
plt.scatter(X[:, 0], X[:, 1], s=100)

# print(X)
# print("###########")
broadcast_1 = X[:, np.newaxis, :]
# print(broadcast_1)
# print("###########")
broadcast_2 = X[np.newaxis, :, :]
# print(broadcast_2)
# print("###########")
differences = broadcast_1 - broadcast_2
# print(differences)
# print(differences.shape)
dist_sq = np.sum(differences ** 2, axis=-1)
dist_sq.shape
dist_sq.diagonal()

nearest = np.argsort(dist_sq, axis=1)
print(nearest)

print(differences ** 2)
print("################################")
print(nearest)
print("################################")
print( (differences ** 2)[nearest] )


test_fancy_broadcast_index = np.arange(dist_sq.shape[0])
test_fancy_broadcast_index
test_hardcoded_index = np.array([[0, 0, 0, 0, 0],
                                 [1, 1, 1, 1, 1],
                                 [2, 2, 2, 2, 2],
                                 [3, 3, 3, 3, 3],
                                 [4, 4, 4, 4, 4]])
test_hardcoded_index_2 = test_hardcoded_index.T
print(test_hardcoded_index_2)

print( dist_sq )
print( np.sort(dist_sq, axis=1) )
print( dist_sq[ test_hardcoded_index, nearest])
# test_fancy_broadcast_index[:np.newaxis]
print( dist_sq[ test_fancy_broadcast_index[np.newaxis , :], nearest])
print( dist_sq[ test_fancy_broadcast_index[np.newaxis , :], nearest] == dist_sq[ test_hardcoded_index_2, nearest])
print( dist_sq[ test_fancy_broadcast_index                , nearest] == dist_sq[ test_hardcoded_index_2, nearest])
print( dist_sq[ test_fancy_broadcast_index[ :, np.newaxis], nearest])
print( dist_sq[ test_fancy_broadcast_index[ :, np.newaxis], nearest] == np.sort(dist_sq, axis=1))

# %%

# don't need to do a full sort of the dist_sq, but just the k-nearest neighbours
K = 2
nearest_partition = np.argpartition(dist_sq, K + 1, axis=1)
print(nearest_partition)
# check with full sorted
print( np.argsort(dist_sq, axis=1) )

plt.scatter(X[:, 0], X[:, 1], s=100)

print(X[0])
# draw lines from each point to its two nearest neighbors
for i in range(X.shape[0]):
    # K+1 ensures that distance from point to itself is 'counted' as a neighbour
    for j in nearest_partition[i, :K+1]:
        # plot a line from X[i] to X[j]
        # use some zip magic to make it happen:
        # NOTE::: zip to get x-coords together and y-coords together
        # * operator to separate the x-coords and y-coords
        plt.plot(*zip(X[j], X[i]), color='black')
        
        # equivalent
        # plt.plot([X[j][0], X[i][0]],[X[j][1], X[i][1]])
        
# alternative way that doesn't use loop to draw?

# %%
plt.plot(*zip(X[0], X[1]))
print(list(zip(X[0], X[1])))
print(sum(*zip(X[0], X[1])))
print(np.add(*zip(X[0], X[1])))
# %%
plt.plot(np.arange(5), np.arange(5)[::-1], color='black')
plt.plot([3,5,4,2,2], [5,1,2,2,3], color='black')
# %%

#### FOR DATAFRAMES creation --> dictionary key relate to columns
#### FOR SERIES creation     --> dictionary key relate to the index
#### FOR SERIES --> DATAFRAMES CONVERSION ??? Series index becomes Dataframe index, column manually assigned
data = [{'a': i, 'b': 2 * i}
        for i in range(3)]
print(pd.DataFrame(data))

data_series = {'a':1, 'b':5}
print(pd.Series(data_series))
print(pd.DataFrame(pd.Series(data_series)))
# %%
import pandas as pd
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
data

print(data.keys())
print(data.index.argmax(axis=0))

print(data.values)
print(list({'a':1, 'b':25}.values()))
# %%
area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
print(data)
print(data.values[0])
print(data.values[2, 1])
print(data.values[2][1])
print(data['area'])

data.stack().mean()

data.iloc[0]
# %%
area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': np.nan})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': np.nan,
                 'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
print(data)
data_np = data.to_numpy()
print(data_np)

data.fillna(method="ffill", axis=1)
print(data)

data = pd.DataFrame({'area':area, 'pop':pop})
data.fillna(method="ffill", axis=0)
print(data)

# MISLEADING _is_view
test_iloc = data.iloc[1:2, [0,1]]
print(test_iloc._is_copy)
print(test_iloc._is_view)
print("test_iloc = ")
print(test_iloc)
test_iloc[:] = 3500

print(data)

print(type(data.iloc[1, 0]))
print(type(data.iloc[1, [1]]))
print(type(data.iloc[[1], 1]))
print(type(data.iloc[[1], [1]]))
# %%
vals2 = np.array([1, np.nan, 3, 4]) 
print(vals2.sum(), vals2.min(), vals2.max())
print(np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2))

vals2 = pd.Series(vals2)
print(vals2.sum(), vals2.min(), vals2.max())
# %% 
vals1 = np.array([1, None, 3, 4])
# NumPy vs Pandas null type handling for dtype, regarding the 'None' value in Python
print(vals1.dtype)
print(pd.Series([1, np.nan, 2, None]).dtype)

# %%
s = pd.Series([1,2,3,None], dtype='Int64')
print(pd.__version__)

import pyarrow as pa
pd.options.mode.copy_on_write = True
df_arrow = pd.read_csv("TestCSV.txt", dtype_backend='pyarrow',
                       engine='pyarrow')
print(df_arrow)
df_arrow.dtypes
# %%
test_data = np.arange(10_000_000).reshape((10_000, 1000))
test_data[test_data>100] = 100
np.exp(test_data)
# %%
print(area.iloc[1])
print(area[area>500_000])
print(area.reindex([5,3,4,2,"California"]))
# %%
index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]
pop = pd.Series(populations, index=index)
print(pop)

index = pd.MultiIndex.from_tuples(index)
index.names = ['State', 'Year']
print(index)

pop = pop.reindex(index)
print(pop.index.name)
print(pop.index.names)
print(pop)

# %%
# SERIES CONCAT
pop.name = 'Pop_series_name'
print(pd.concat([pop, pd.Series([9267089, 9284094,
                                   4687374, 4318033,
                                   5906301, 6879014], 
                          index=index, name='under18')], axis=1,
                keys=["series1", "series2"]))
print("#####")
print(pd.concat([pop, pd.Series([9267089, 9284094,
                                   4687374, 4318033,
                                   5906301, 6879014], 
                          index=index, name='under18')], axis=1,
                keys=None))
print("#####")
print(pd.concat([pop, pd.Series([9267089, 9284094,
                                   4687374, 4318033,
                                   5906301, 6879014], 
                          index=index, name='under18')], axis=0,
                keys=["series1","series2"]))


### DATAFRAME CONCAT
df1 = pd.concat([pop, pd.Series([9267089, 9284094,
                                   4687374, 4318033,
                                   5906301, 6879014], 
                          index=index, name='under18')], axis=1,
                keys=None)

print(pd.concat([df1, df1]))
print(pd.concat([df1, df1], axis=0, keys=["df1","df1_dup"]))
print(pd.concat([df1, df1], axis=1, keys=["df1","df1_dup"]))

# %%
pop_df = pd.concat([pop, pd.Series([9267089, 9284094,
                                   4687374, 4318033,
                                   5906301, 6879014], 
                          index=index, name='under18')], axis=1,
                keys=None)

print(pop_df)
print(pop_df["under18"] / pop_df["Pop_series_name"])
print((pop_df["under18"] / pop_df["Pop_series_name"]).unstack())
print(pop_df.index)

# Add new row
pop_df.loc[('New Jersery', 2010),:] = {'Pop_series_name': 11133213, 
                                     'under18': 666222}
pop_df
# %%
pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1)])
# %%
# hierarchical indices and columns
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])

# mock some data
rng = np.random.default_rng(seed=12345)
data = np.round(rng.standard_normal((4,6)), 1)
data[:, ::2] *= 10
data += 37

# create the DataFrame
health_data = pd.DataFrame(data, index=index, columns=columns)
print(health_data)

print(health_data['Guido'])
print(type(health_data['Guido']))
print(health_data['Guido'].index.names)
print(health_data['Guido'].columns.names)

print(health_data[['Guido']])
print(type(health_data[['Guido']]))
print(health_data[['Guido']].index.names)
print(health_data[['Guido']].columns.names)

# %%

# equivalents, but tuple is clearer
print(health_data['Guido','HR'])
print(health_data[('Guido','HR')])

# equivalents, 
print(health_data.loc[2013])
print(health_data.loc[(2013, ), ])
print(health_data.loc[2013,])

print(health_data.loc[2013, ('Guido','HR')])
print(health_data.loc[(2013, 2), ('Guido','HR')])

print(health_data.iloc[:2, :2])
print(health_data.loc[:, ('Bob', 'HR')])

print(health_data.loc[:, :'HR'])
print(health_data.loc[:, :'Temp'])



idx = pd.IndexSlice
health_data.loc[idx[:, 1], idx[:, 'HR']]
health_data.loc[idx[:, 1], idx[:, :'Temp']]

# %%
s = pd.Series(
    [1, 2, 3, 4, 5, 6],
    index=pd.MultiIndex.from_product([["A", "B"], ["c", "d", "e"]]))

#Importantly, a list of tuples indexes several complete MultiIndex keys, 
# whereas a tuple of lists refer to several values within a level:

# list of tuples
print(s.loc[[("A", "c"), ("B", "d")]]  )

# tuple of lists
print(s.loc[(["A", "B"], ["c", "d"])]  )


# %%
def mklbl(prefix, n):
    return ["%s%s" % (prefix, i) for i in range(n)]

miindex = pd.MultiIndex.from_product(
    [mklbl("A", 4), mklbl("B", 2), mklbl("C", 4), mklbl("D", 2)], names=["Ax","Bx","Cx","Dx"]
)

micolumns = pd.MultiIndex.from_tuples(
    [("a", "foo"), ("a", "bar"), ("b", "foo"), ("b", "bah")], names=["lvl0", "lvl1"]
)

dfmi = (
    pd.DataFrame(
        np.arange(len(miindex) * len(micolumns)).reshape(
            (len(miindex), len(micolumns))
        ),
        index=miindex,
        columns=micolumns,
    )
    .sort_index()
    .sort_index(axis=1)
)

dfmi

# %%
idx = pd.IndexSlice

# slice("A1", "A3") includes "A2" ("A1" through to "A3")
# see differences for Cx :: slice(None) vs [C1, C3] vs slice(C1, C3)
print(dfmi.loc[(slice("A1", "A3"), slice(None), slice(None)), :])
print(dfmi.loc[(slice("A1", "A3"), slice(None), ["C1", "C3"]), :])
print(dfmi.loc[(slice("A1", "A3"), slice(None), slice("C1", "C3")), :])
print(dfmi.loc[(slice("A1", "A3"), slice(None), slice("C0", "C2")), :])
print("#######")
# ["A1","A3"] includes just "A1" and "A3"
print(dfmi.loc[(["A1","A3"], slice(None), ["C1", "C3"]), :])
print("#######")
# using pd.IndexSlice
print(dfmi.loc[idx[:, :, ["C1", "C3"]], idx[:, "foo"]])
print(dfmi.loc[idx[:, :, "C1":"C3"], idx[:, "foo"]])
print(dfmi.loc[idx["A0":"A2", :, "C1":"C3"], idx[:, "foo"]])
print(dfmi.loc[idx[["A0","A2"], :, "C1":"C3"], idx[:, "foo"]])
# %%
mask = dfmi[("a", "foo")] > 200
dfmi.loc[idx[mask, :, ["C1", "C3"]], idx[:, "foo"]]

# %%
df2 = dfmi.copy()
print(df2.loc[idx[:, :, ["C1", "C3"]], :])

print(df2.shape)
print(df2.loc[idx[:, :, ["C1", "C3"]], :].shape)
df2.loc[idx[:, :, ["C1", "C3"]], :] = df2 * 1000
df2
# %%
df2 = dfmi.copy()
df2.loc[idx[:, :, ["C1", "C3"]], :] = df2 * 1000
test_df2 = df2.loc[idx[:, :, ["C1", "C3"]], :]
test_df2 *= 1000
test_df2
df2
# %%
index = pd.MultiIndex.from_product([[2013, 2014], [2013, 2015]],
                                   names=['year', 'year2'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])

# mock some data
rng = np.random.default_rng(seed=12345)
data = np.round(rng.standard_normal((4,6)), 1)
data[:, ::2] *= 10
data += 37

# create the DataFrame
health_data = pd.DataFrame(data, index=index, columns=columns)

# change 2014 to 2013 --> changes both levels ... BAD method?
# .rename needs to be assigned to a variable to be used
health_data.rename(index={2014: 2023})
print(health_data)

# axis 0 refers to row indexes, axis 1 refers to column indexes
print(health_data.sort_index(level=1, axis=1, inplace=False))
print(health_data)
print(health_data.sort_index(level=1, axis=0, inplace=False))

print(health_data.sort_index(level=1, axis=1, inplace=True))
print(health_data)
health_data.sort_index(level=1, axis=0, inplace=True)
print(health_data)
health_data.sort_index(level=0, axis=0, inplace=True)
print(health_data)

idx = pd.IndexSlice
health_data.loc[idx[2014:,2015:], idx[:,:]]

# %%
print(health_data)
print(health_data.reset_index())
print(health_data.reset_index().stack())
print(health_data.stack().reset_index())
print(health_data.stack().stack().reset_index(name='health_score'))
print(health_data.stack(level=[1,0]).reset_index(name='health_score'))
print(health_data.stack(level=[0,1]).reset_index(name='health_score'))

# %%
df = pd.DataFrame(

    [["bar", "one"], ["bar", "two"], ["foo", "one"], ["foo", "two"]],

    columns=["first", "second"],

)

# %%
txt1 = "My name is {fname}, I'm {age}".format(fname = "John", age = 36)
txt2 = "My name is {0}, I'm {1}".format("John",36)
txt3 = "My name is {}, I'm {}".format("John",36) 

print(txt1)
print(txt2)
print(txt3)
# %%
spice_list = ['salt', 'pepper', 'oregano', 'sage', 'parsley',
              'rosemary', 'tarragon', 'thyme', 'paprika', 'cumin']

recipes = pd.DataFrame(
        pd.Series(['saltyyyy', 'tarragoner', 'PaprIKAEA', None], 
                  name='ingredients'))

import re
spice_df = pd.DataFrame(dict(
    (spice, recipes["ingredients"].str.contains(
        pat=spice, case=False, na="INGREDIENT NOT AVAILABLE"))
                             for spice in spice_list))
spice_df.head()

  
# %%
import pandas as pd
nrows, ncols = 10000, 100
rng = np.random.RandomState(42)
df1, df2, df3, df4 = (pd.DataFrame(rng.rand(nrows, ncols))
                      for _ in range(4))
# %%
x = np.linspace(0, 10, 30)
y = np.sin(x)

plt.plot(x, y, 'o', color='black')
# %%
rng = np.random.RandomState(0)
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    plt.plot(rng.rand(5), rng.rand(5), marker,
             label="marker='{0}'".format(marker))
plt.legend(numpoints=1)
plt.xlim(0, 1.8)
# %%
# - means solid line?, o means the circle marker for each point,
# k means black colour?
plt.plot(x, y, '-ok');
# %%
plt.plot(x, y, '-p', color='gray',
         markersize=15, linewidth=4,
         markerfacecolor='white',
         markeredgecolor='gray',
         markeredgewidth=2)
plt.ylim(-1.2, 1.2)

# %%
x = np.linspace(0, 10, 50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)

plt.errorbar(x, y, yerr=dy, fmt='.k')
# %%
# extra arguments not specifically related to errorbar 
# go into the .plot() for plotting stuff
plt.errorbar(x, y, yerr=dy, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
# %%
x = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), '-b', label='Sine')
ax.plot(x, np.cos(x), '--r', label='Cosine')
ax.axis('equal')
leg = ax.legend()


# %%
y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5))
print(y.shape)
print(x.shape)
lines = plt.plot(x, y)

# lines is a list of plt.Line2D instances
plt.legend(lines[:2], ['first', 'second'])
# %%
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])
plt.subplot(grid[1, 2])
# %%
import polars as pl

df = pl.DataFrame(
    {
        "a": [1, 4, 3, 2, 8, 4, 5, 6],
        "b": [2, 3, 1, 3, 9, 7, 6, 8],
        "c": [1, 1, 1, 1, 2, 2, 2, 2],
    }
)

df.group_by("c").agg(
    pl.when(pl.col("c") == 1)
        .then(pl.min_horizontal("a","b"))
        # .then(pl.min_horizontal(pl.all()))
        .otherwise(pl.col("a"))
        .alias("testing")
).explode("testing")

df.group_by("c", maintain_order=True).agg(
    pl.when(pl.col("c") == 1)
        # .then(pl.reduce(lambda a, b: a if a < b else b, pl.all()))
        .then(pl.reduce(lambda s1, s2: s1.zip_with(s1 < s2, s2), pl.col("a","b")))
        .otherwise(pl.col("a"))
        .alias("testing")
).explode("testing")



# %%
df = pl.DataFrame({"a": [None, 2, 3],
                   "b": [3, None, 4],
                   "c": [5, 4, None]})

find_first_in_row = lambda a, b: a.zip_with(a.is_not_null(), b)

print(
    df.select(pl.fold(acc=pl.lit(None), function=find_first_in_row, exprs=pl.col("*")))
)
print(
    df.select(pl.fold(acc=pl.Series(values=[None, None, None]), function=find_first_in_row, exprs=pl.col("*")))
)
print(
    df.fold(find_first_in_row)
)
# %%
weights = [7.4, 3.2, -0.13]

df = pl.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "p1": [44.3, 2.3, 2.4, 6.2],
        "p2": [7.3, 8.4, 10.3, 8.443],
        "p3": [70.3, 80.4, 100.3, 80.443],
        "p4": [16.4, 18.2, 11.5, 18.34],
    }
)
df

col_names = ["p1","p2","p3"]
df.with_columns(
    ((pl.col(col) * weights[i]) 
    for i, col in enumerate(col_names))
).with_columns(pl.sum_horizontal(col_names).alias("p1-3_horizontal"))

df.with_columns(
    pl.sum_horizontal(
        [pl.col(col_nm) * wgt
         for col_nm, wgt in zip(col_names, weights)]
    ).alias("index")
)

wghtd_cols = [
    pl.col(col_nm) * wgt
    for col_nm, wgt in zip(col_names, weights)
    if wgt != 0.0
]
df.with_columns(pl.sum_horizontal(wghtd_cols).alias("index"))
# %%
df = pl.DataFrame({
   'values': [[1, 3, 2], [5, 7]],
    'weights': [[.5, .3, .2], [.1, .9]]
})

df.with_columns(
    # pl.col("values").list.sum().alias("v"),
    # pl.col("weights").list.sum().alias("w"),
    pl.col("values").list.eval(
        pl.element() * pl.col("*").list.eval(pl.element())
    )
)
# %%
df = pl.DataFrame({
   'values': [[1, 3], [5, 7]],
    'weights': [[.5, .3], [.1, .9]]
},
schema={'values': pl.Array(pl.Int64, 2), 'weights': pl.Array(pl.Float64, 2)}
)

df.with_columns(
    # pl.col("values").list.sum().alias("v"),
    # pl.col("weights").list.sum().alias("w"),
    (pl.col("values") * pl.col("weights")).alias("v*w")
)
# %%
df = pl.DataFrame({
   'values': [[1, 3, 2], [5, 7]],
    'weights': [[.5, .3, .2], [.1, .9]]
},
schema={'values': pl.List(pl.Int64), 'weights': pl.List(pl.Float64)}
)

# df.with_columns(
#     # pl.col("values").list.sum().alias("v"),
#     # pl.col("weights").list.sum().alias("w"),
#     (pl.col("values") * pl.col("weights")).alias("v*w")
# )

df.explode("values")
df.explode("values").explode("weights")

(df.with_row_index(name="index")
 .explode("values","weights")).group_by("index").agg(
     (pl.col("values") * pl.col("weights")).sum()
 )
# %%
df = pl.DataFrame({
    'x1': [[1,2],[3,4]],
    'x2': [10,20]
})

x2_temp = df.get_column('x2')
# df.with_columns(
#     scaled_x1 = pl.col('x1').list.eval(pl.element() / x2_temp)
#     scaled_x1 = pl.col('x1') / pl.col('x2')
#     scaled_x1 = (pl.col('x1').list.eval(pl.element() / pl.lit(pl.col('x2').sum())))
# )

# this has 4 rows
df.with_row_index().explode('x1').with_columns(
    # BUT outside has 4 rows, this has only 2 rows?
    # pl.col('x2').implode().over("index").alias("test1")
    pl.col('x2').implode(),
    pl.col('x1').sum().over('index').alias("testover1"),
    # sort of replicating the implode.over strategy but without implode()
    pl.col('x2').over("index", mapping_strategy='join').alias("testover2")
)

df.with_row_index().explode('x1').select(
    # this has only 2 rows? imcompatible with with_columns due to above being 4 rows
    # pl.col('x2').implode(),
    pl.col('x2').implode().over("index").alias("test1")
)

# %%
df = pl.DataFrame({
    'dense': [[0, 9],[8,6,0,9], None, [3,3]],
})
df.with_columns(
    sparse_indices=pl.col('dense').list.eval(
        pl.arg_where(pl.element() != 0)
    ),
    test = pl.col('dense').list.contains(pl.lit(0)),
    # using gather and arg_where
    test_1 = pl.col('dense').list.gather(
        pl.col('dense').list.eval(
        pl.arg_where(pl.element() != 0)
        )
    )
).with_columns(
    test_2 = pl.col('dense').list.gather("sparse_indices")
)

# %%
from urllib.parse import urlparse

urlparse("scheme://netloc/path;parameters?query#fragment")
# %%
import urllib.request

link = "https://docs.google.com/document/d/e/2PACX-1vRMx5YQlZNa3ra8dYYxmv-QIQ3YJe8tbI3kqcuC7lQiZm-CSEznKfN_HYNSpoXcZIV3Y_O3YoUB1ecq/pub"
f = urllib.request.urlopen(link)
myfile = f.read()
print(myfile)
# %%
import requests

def retrieve_parse_and_print_doc(docURL):
    response = requests.get(docURL)
    assert response.status_code == 200, 'Wrong status code'
    lines = response.content.splitlines()
    for line in lines:
        print(line)

retrieve_parse_and_print_doc('https://docs.google.com/document/d/e/2PACX-1vRMx5YQlZNa3ra8dYYxmv-QIQ3YJe8tbI3kqcuC7lQiZm-CSEznKfN_HYNSpoXcZIV3Y_O3YoUB1ecq/pub')
# %%
np.arange(100)