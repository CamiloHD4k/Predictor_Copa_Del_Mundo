import pandas as pd
from pathlib import Path

path = Path('data/processed/matches_master.csv')
if not path.exists():
    raise SystemExit('matches_master.csv no existe')

df = pd.read_csv(path)
print('shape', df.shape)
print('columns', df.columns.tolist())
print('sources', df['source'].value_counts().to_dict())
print('result distribution', df['result'].value_counts(dropna=False).to_dict())
print('date nulls', df['date'].isna().sum())
if 'date' in df.columns:
    sample = df['date'].dropna().head(10).tolist()
    print('date sample', sample)
print(df.head(20).to_string(index=False))
