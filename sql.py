# %%
import polars as pl
import sqlalchemy as sa
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey, Sequence
# string --> dialect+driver://username:password@host:port/database

from sqlalchemy.engine import URL

url_object = URL.create(
    "mssql+pyodbc",
    # next two arguments created using:::
    # USE PythonTests;
    # CREATE LOGIN YourUsername WITH PASSWORD = 'YourPassword';
    # CREATE USER YourUsername FOR LOGIN YourUsername;
    username="YourUsername",
    password="YourPassword",  # plain (unescaped) text
    host="EZ",
    port=None,
    database="PythonTests",
    query={
        "driver": "ODBC Driver 17 for SQL Server",
        "LongAsMax": "Yes",
        "trusted_connection": "yes",
        # "TrustServerCertificate": "yes",
        # "authentication": "ActiveDirectoryIntegrated",
    }
)
url_object.render_as_string(hide_password=False)
url_object.render_as_string(hide_password=True)
url = "mssql+pyodbc://EZ/PythonTests?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
url

test_sql = pl.DataFrame([pl.Series("some_name", [1,4,5,1]), pl.Series("some_name2", [2,6,1,7])])
# test_sql["some_name"].qcut(2)
# test_sql.write_database("test_sql", url_object.render_as_string(hide_password=False), if_table_exists="append")
# test_sql.write_database(table_name="test_sql", connection=url, if_table_exists="append")

engine = sa.create_engine(url)
pl.read_database("SELECT * from test_sql", engine)

url2 = r"mssql://EZ/PythonTests?driver=SQL+Server&trusted_connection=yes&encrypt=true"
url2t = r"mssql://EZ/PythonTests?driver=SQL+Server&trusted_connection=true&encrypt=true"
url3 = f'mssql://YourUsername:YourPassword@EZ/PythonTests?driver=SQL+Server&trusted_connection=yes'
url3t = f'mssql://YourUsername:YourPassword@EZ/PythonTests?driver=SQL+Server&trusted_connection=true'
url4 = r"mssql://EZ/PythonTests?encrypt=true&trusted_connection=true"

# this is timing out at the moment, 
# read_database_uri relies on connectorx so also times out
# import connectorx as cx
# cx.read_sql(url, 
#             "SELECT * FROM test_sql")

# pl.read_database_uri(query="SELECT * from test_sql", uri=url, 
#                     #  partition_num=3
#                      )

# %%

from sqlalchemy.orm import Session
session = Session(engine)

from sqlalchemy import text
with engine.connect() as connection:
    result = connection.execute(text("select * from test_sql"))
    for row in result:
        print(row)

# %%
##############################
customer_list = ['ABC', '123']

# parameterized query placeholders
placeholders = ",".join("?" * len(customer_list))
# query table
query = """
SELECT
[ID],
[Customer]
FROM xyz.dbo.abc
WHERE [Customer] IN (%s)
""" % placeholders
print(query)

############################
var1 = 5
var2 = 6
sql = "INSERT INTO test_sql VALUES (%d, %d)"
args = var1, var2

metadata = MetaData()
users = Table('test_sql', metadata,
   Column('some_name'),
   Column('some_name2')
   )
ins = users.insert().values(some_name=5, some_name2=6)
ins2 = users.insert()

engine = sa.create_engine(url)
# with engine.connect() as connection:
#     result = connection.execute(ins)
#     connection.commit()
    
# with engine.connect() as connection:
#     result = connection.execute(text("select * from test_sql"))
#     for row in result:
#         print(row)
        
with engine.connect() as connection:
    result = connection.execute(ins2, some_name=50, some_name2=60)
    connection.commit()
    
with engine.connect() as connection:
    result = connection.execute(text("select * from test_sql"))
    for row in result:
        print(row)
        



# %% ### In memory sqlite database (transient) ###
# https://docs.sqlalchemy.org/en/20/
sa.__version__
from sqlalchemy import create_engine
engine = create_engine("sqlite+pysqlite:///:memory:", echo=True)

# %% #### Write to and read to sqlite database #####
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData
import sqlite3

engine = create_engine('sqlite:///example.db')
metadata = MetaData()

users = Table('users', metadata,
              Column('id', Integer, primary_key=True),
              Column('name', String),
              Column('age', Integer))

metadata.create_all(engine)

# Insert data using a list of dictionaries
data = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25}
]

with engine.connect() as conn:
    conn.execute(users.insert(), data)
    conn.commit()
    
### Reading the results back
# Connect to the database
conn = sqlite3.connect('example.db')

# Create a cursor object
cursor = conn.cursor()

# Execute a SELECT query
cursor.execute("SELECT * FROM users")

# Fetch and print the results
rows = cursor.fetchall()
for row in rows:
    print(row)

# Close the connection
conn.close()
# %%
