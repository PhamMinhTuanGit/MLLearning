import torch 
import pandas as pd


data = pd.read_csv("/home/parallels/Documents/code/MLlearning/Housing.csv")
data_filtered = data.iloc[:, 0:9]
data_filtered = data_filtered.replace({'no': '0', 'yes': '1'})
data_filtered.to_csv("Housing_processed.csv")




print(data_filtered.head)
