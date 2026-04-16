"""
Utility functions for visualization experiments.
"""

import json
import pandas as pd
from io import StringIO


def load_data(path, path_is="jsonl"):
    """Load data from JSONL or CSV file."""
    data = []

    if path_is == "csv":
        if path is not None:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            table = pd.read_csv(StringIO(''.join(lines)))
            if "class" in table.columns:
                table.drop(columns=["class"], inplace=True)
            for _, row in table.iterrows():
                row_json = row.to_dict() 
                data.append(row_json)

    elif path_is == "jsonl":
        if path is not None:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))

    return data
