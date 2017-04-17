# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import requests
import io
import numpy as np


url = 'http://blog.topspeedsnail.com/wp-content/uploads/2016/12/铁路客运量.csv'
ass_data = requests.get(url).content
data_csv = io.StringIO(ass_data.decode('utf-8'))
df = pd.read_csv("铁路客运量.csv")  # python2使用StringIO.StringIO

data = np.array(df['铁路客运量_当期值(万人)'])
# normalize
normalized_data = (data - np.mean(data)) / np.std(data)

plt.figure()
plt.plot(data)
plt.show()