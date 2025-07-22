import torch
import numpy as np

def mean_squared_error(predict, label):
	'''
	predict(array): predicted value of the network
	label(array): ground truth value
	'''
    assert predict.shape == label.shape
    error = predict - label
    squared_error = error ** 2
    return torch.mean(squared_error)

def cross_entropy_loss(probs, label):
	"""
	probs(array): output probality vector of the model
	label(array): one-hot encoding label
	"""
	assert probs.shape == label.shape
	assert np.isclose(np.sum(label), 1.0) #to check if label is one-hot encoding?
    assert np.all(probs >= 0) and np.all(probs <= 1) #to check if probs is probality vector or not
	smooth = 1e-12  #smooth operator to avoid divide by 0
	return -np.dot(label, np.log2(probs + smooth))

def negative_log_likelihood(logits, label):
	'''
	logits(array): softmax-ed probs
	label(array): one-hot encoding label
	'''
	assert logits.shape == label.shape
	assert np.isclose(np.sum(label), 1.0) #to check if label is one-hot encoding?
    assert np.all(logits >= 0) and np.all(logits <= 1) #to check if probs is probality vector or not
	smooth = 1e-12
	return -np.dot(label, np.log2(logits + smooth))



def Dice_loss(predict, label):
	'''
	predict(array): argmax-ed predict of the model
	label(array): label
	'''
	smooth = 1e-6
	intersection = np.sum((predict==1)&(label==1))
	total = np.sum((predict==1)) + np.sum((label==1))
	return 2*(intersection+smooth)/(total+smooth)