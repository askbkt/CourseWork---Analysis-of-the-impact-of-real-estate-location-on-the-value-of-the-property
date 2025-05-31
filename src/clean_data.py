import pandas as pd

df = pd.read_csv('cleaned_data.csv')

df = df.dropna()

df.to_csv('cleaned_data.csv', index=False)

print(f"Очищенные данные: {df.shape}")
