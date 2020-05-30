import numpy as np


def average_metric(metrics):
    average_metric = np.mean(metrics)
    std_metric = np.sqrt(np.mean(np.square(metrics - average_metric)))

    return (average_metric, std_metric)

# Calculate accuracy percentage
def accruacy(actual, predicted):
	'''
		Args : 
			actual : np.array [datapoint,]
			predicted : np.array [datapoint, ]

		Returns : 
			accuracy : scalar
	'''
	assert np.shape(actual) == np.shape(predicted)
	# The number of point
	data_count = np.shape(actual)[0]

	# Comaprison
	correct = np.sum(np.equal(actual, predicted))

	return float(correct) / float(data_count) * 100.0