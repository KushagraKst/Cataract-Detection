import pandas as pd

df = pd.read_parquet("cataract-training-dataset.parquet")

print(df.head())      # shows first 5 rows
print(df.columns)     # shows column names

import pandas as pd

df = pd.read_parquet("cataract-training-dataset.parquet")

print(df.columns)