import pandas as pd
import os.path

# Get the absolute path, move to the root directory and add file path (works for everyone)
my_path = os.path.abspath(os.path.dirname(__file__))
path = my_path.replace('\project_code\eda','')
full_path = os.path.join(path, "data/raw/HAM10000/HAM10000_metadata.csv")
with open(full_path) as f:
    ham10000_df = pd.read_csv(f)
print(ham10000_df)