from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import re
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_points(filename: str) -> Tuple[np.ndarray, Dict[Tuple[float, float], int]]:
    points_list = []  # To store only (x, y) tuples for alpha shape solver
    coords_to_id = {}  # To map (x, y) to ID
    
    df = pd.read_csv(filename)
    if df.size > 0 :
        logger.info('Successfully read file!')
    for index,row in df.iterrows():      
        try:
            #fields = re.split(r'\s+', line)
            point_id = float(row[0])  # Extract ID
            x = float(row[1])       # Extract x
            y = float(row[2])       # Extract y

            # Append (x, y) to the points array
            points_list.append((x, y))

            # Add (x, y) -> ID mapping to the dictionary
            coords_to_id[(x, y)] = point_id
        except ValueError:
            # Skip lines not containing valid data
            continue

    points_array = np.array(points_list, dtype=np.float64)
    

    return points_array, coords_to_id

