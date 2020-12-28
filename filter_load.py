from filereader import extract_index_dict
import numpy as np
import sklearn
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import os

from filereader import search,search_total_raw,get_filted_data
search_list = ("T03", "T04", "T05", "T06", "T02", "T01")
search_list = ("T02",)
search_dir = r"D:\xy-repo\data"
tool_path = search_total_raw(search_dir,search_list)
result = get_filted_data(tool_path)
for k,v, in result.items():
    v.to_csv(os.path.join("filted_basic_info",k+".csv"))

















