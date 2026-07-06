from pathlib import Path
import pandas as pd
from config import Config

label_switch_dictionary = {"Type 1": "y1", "Type 2": "y2", "Type 3": "y3", "Type 4": "y4",}
text_columns = [Config.TICKET_SUMMARY, Config.INTERACTION_CONTENT]
required_columns = [Config.TICKET_SUMMARY, Config.INTERACTION_CONTENT, *Config.TYPE_COLS]

# Loading the data
input_paths = [Config.DATA_DIR / name for name in Config.INPUT_FILES]

# Reading the CSV files
def read_csv(path):
    if not path.exists():
        raise ValueError(f"Input file not found: {path}")
    
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=df.columns[df.columns.str.startswith("Unnamed")])
    df = df.rename(columns=label_switch_dictionary)
    for col in text_columns:
        df[col] = df[col].fillna("").astype(str)
    return df

# Validating loaded data
def validate_data(df, path) -> None:

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing} in file {path}")
    
    if df.empty:
        raise ValueError(f"Dataframe is empty after loading from {path}")
    
def get_final_dataframe():
    final_df = pd.DataFrame()

    for path in input_paths:
        df = read_csv(path)
        validate_data(df, path)
        final_df = pd.concat([final_df, df], ignore_index=True)
        print(f"Loaded and validated data from {path}")

    print("Final dataframe shape:", final_df.shape)
    print("Columns in final dataframe:", final_df.columns.tolist())
    return final_df

if __name__ == "__main__":
    final_df = get_final_dataframe()
