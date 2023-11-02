import pandas as pd
from prettytable import PrettyTable

# Load CSV files
dfs = {
    'M_v': pd.read_csv('./data.gender.vad.scores/data.M.v.csv'),
    # 'M_a': pd.read_csv('./data.gender.vad.scores/data.M.a.csv'),
    # 'M_d': pd.read_csv('./data.gender.vad.scores/data.M.d.csv'),
    'F_v': pd.read_csv('./data.gender.vad.scores/data.F.v.csv'),
    # 'F_a': pd.read_csv('./data.gender.vad.scores/data.F.a.csv'),
    # 'F_d': pd.read_csv('./data.gender.vad.scores/data.F.d.csv'),
}

# Function to compute mean and standard deviation
def compute_stats(df, mean_key, std_key):
    mean_val = df[mean_key].mean()
    std_val = df[std_key].std()
    return mean_val, std_val

# Prepare table
table = PrettyTable()
table.field_names = ["Data", "Mean Value", "Standard Deviation"]

for name, df in dfs.items():
    if name.endswith('_v'):
        mean_key, std_key = 'V', 'std(V)'
    elif name.endswith('_a'):
        mean_key, std_key = 'A', 'std(A)'
    elif name.endswith('_d'):
        mean_key, std_key = 'D', 'std(D)'
    
    mean_val, std_val = compute_stats(df, mean_key, std_key)
    table.add_row([name, mean_val, std_val])

with open('new_table.txt', 'w') as f:
    f.write(str(table))
