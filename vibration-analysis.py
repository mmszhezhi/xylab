import os,json
import numpy as np
import pandas as pd

with open("trimed_vib_data/T05.json","r") as f:
    data = json.load(f)

# data["1"]["vibration-x"]