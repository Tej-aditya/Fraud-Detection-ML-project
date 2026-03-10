from data.dataset_loader import load_data

df = load_data(nrows=200_000)  # sample, fast

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df["isFraud"].value_counts())
