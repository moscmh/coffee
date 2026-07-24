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
    coffee = pd.read_csv(data_dir / 'coffee_v2_cleaned.csv')
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
    coffee.to_csv(data_dir / 'coffee_v2_cleaned.csv', index=False)

    coffee.to_csv(data_dir / f'backup/coffee_{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}.csv', index=False)
    backup_dir = data_dir / 'backup'
    backup_files = sorted(backup_dir.glob('*.csv'))
    for file in backup_files[:-10]:
        file.unlink()
    
    conn = sqlite3.connect(script_dir.parent / "database.db")
    data.to_sql("coffee", conn, if_exists="append", index=False)
    print("Latest coffee records...")
    cur = conn.cursor()
    cur.execute("SELECT * FROM coffee ORDER BY Date DESC LIMIT 3")
    rows = cur.fetchall()
    for row in rows:
        print(row)
    conn.close()
    pass

def main():
    new_data = pd.DataFrame(
        {
            'Date': '2026-07-21',
            'Method': 'Filter',
            'Temperature': 16,
            'Humidity': 15,
            'Origin': 'Ethiopia',
            'Elevation': 2050,
            'Region': 'Guji',
            'Process': 'Natural',
            'Varietal': 'Heirloom',
            'DateRoast': '2026-06-11',
            'DateOpen': '2026-07-18',
            'Size': 53,
            'Grind': 16.5,
            'WaterTemp': 98,
            'Time': 180,
            'Output': 300,
            'Note': '',
            'Score': 5
        },
        index=[0]
        )
    add_data(new_data)

if __name__ == "__main__":
    main()