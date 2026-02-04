
import os

try:
    # Try UTF-16
    with open('region_matrix.txt', 'r', encoding='utf-16') as f:
        print(f.read())
except:
    try:
        # Try UTF-8 as fallback
        with open('region_matrix.txt', 'r', encoding='utf-8') as f:
            print(f.read())
    except Exception as e:
        print(f"Error: {e}")
