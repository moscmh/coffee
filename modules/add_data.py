# Add new coffee data to the dataset

from pathlib import Path
import time
from pathlib import Path
import pandas as pd
import numpy as np
import sqlite3

def retrieve_data() -> pd.DataFrame:
    '''
    Retrieve the latest coffee data from the cleaned csv file.
    -----
    Parameters
    -----
        None

    Returns
    -----
        pd.DataFrame: A pandas DataFrame containing the cleaned coffee data.
    '''
    global script_dir
    script_dir = Path(__file__).parent
    global data_dir
    data_dir = script_dir.parent / "data"
    coffee = pd.read_csv(data_dir / 'coffee_cleaned.csv')
    return coffee

def add_data(data: pd.DataFrame) -> None:
    '''
    Add new coffee data to the latest cleaned dataset and save a copy in the backup folder.
    -----
    Parameters
    -----
        data (pd.DataFrame): A pandas DataFrame containing the new coffee data to be added.

    Returns
    -----
        None
    '''
    coffee = retrieve_data()
    coffee = pd.concat([coffee, data], ignore_index=True)
    coffee.to_csv(data_dir / 'coffee_cleaned.csv', index=False)

    coffee.to_csv(data_dir / f'backup/coffee_{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}.csv', index=False)
    backup_dir = data_dir / 'backup'
    backup_files = sorted(backup_dir.glob('*.csv'))
    for file in backup_files[:-10]:
        file.unlink()
    
    conn = sqlite3.connect(script_dir.parent / "database.db")
    data.to_sql("coffee", conn, if_exists="append", index=False)
    conn.close()
    pass

def main():
    new_data = pd.DataFrame(
        {
            'Date': '2026-07-11',
            'Method': 'Filter',
            'Origin': 'Ethiopia',
            'Size': 9,
            'Grind': 16.5,
            'Time': 180,
            'Output': 300,
            'Note': 'Great aroma',
            'Score': 5
        },
        index=[0]
        )
    add_data(new_data)

if __name__ == "__main__":
    main()