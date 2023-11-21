"""generic utils"""
from typing import Dict
import csv

def read_class_names(class_map_csv: str) -> Dict:
    """Read the class name definition file and return a list of strings."""
    with open(class_map_csv, mode='r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip header
        return {
            int(i): display_name for (i, _, display_name) in reader
            }
