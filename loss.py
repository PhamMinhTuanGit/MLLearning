import torch
import numpy

def mean_squared_error(predict, label):
    error = predict - label
    squared_error = error ** 2
    return torch.mean(squared_error)

