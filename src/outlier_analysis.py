import pandas as pd
import numpy as np

df = pd.read_csv('cleaned_data.csv')

df['error'] = (df['true_price'] - df['pred_price']).abs()

mae = df['error'].mean()
print(f"MAE: {mae:.2f}")

threshold = 2 * mae

df = df[df['error'] <= threshold]

df.to_csv('cleaned_data.csv', index=False)

outliers = df[df['error'] > threshold]
print(f"Число выбросов: {len(outliers)}")
print(outliers[['true_price', 'pred_price', 'error']].sort_values('error', ascending=False).head(10))
