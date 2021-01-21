import joblib
import pandas as pd
import numpy as np
from filereader import vib_read_raw,extract_index_dict
import glob,os
import json

tools = glob.glob("./filted_basic_info/*")
for tool in tools:
    print(tool)
    name = os.path.basename(tool)
    df = pd.read_csv(tool,index_col=0)
    index_dict = extract_index_dict(df["paths"])
    trimed = vib_read_raw(index_dict,df,('1','2','3'))
    with open(os.path.join("trimed_vib_data",name.split(".")[0]) + ".json","w") as f:
        json.dump(trimed,f)
    # joblib.dump(trimed,os.path.join("trimed_vib_data",name.split(".")[0]))
