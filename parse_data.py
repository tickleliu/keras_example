import pandas as pd
import numpy as np
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

f = open("/home/liuml/keras/dataset/dataset_1.csv", encoding="GBK")
df = pd.read_csv(f)
data = np.array(df["最高价"])
print(data.shape)
data = data[::-1]
