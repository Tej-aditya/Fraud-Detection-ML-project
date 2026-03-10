from data.dataset_loader import load_data
from data.dataset_loader import load_data

df = load_data(nrows=100_000)
print(df.columns.tolist())

df = load_data(nrows=100_000)
print(df.head())
print(df.shape)
