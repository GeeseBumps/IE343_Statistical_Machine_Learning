import numpy as np
from .metric import gini_index

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	'''
		split the dataset into 'left' and 'right' given the feature index and the threshold

		Args:
			index : index of features (scalar)
			value : the treshold (scalar)
			dataset : dataset to be splitted ([n_datapoints, n_features + 1])

		Return 
			(left, right) : splitted dataset ([n_datapoints, n_features+1], [n_datapoints, n_features+1])
	'''
	# Compare the value with the split point
	left_indices = dataset[:,index] < value
	right_indices = [not bool for bool in left_indices]

	left = dataset[left_indices, :]
	right = dataset[right_indices, :]

	return left, right

# Select the best split point for a dataset
def get_split(dataset, n_features):
	'''
		Looking for the split point and split dataset into 'left' and 'right'
		
		Args:
			dataset : dataset should be splited [n_points, n_features]
			n_features : total number of features [n_features,]

		Returns:
			result : dict {'index', 'value', 'groups'}
				'index' : feature index(scalar) 
				'value' : value of split point in the indiced feature (scalar)
				'groups' : left set / right set ([n_datapoints, n_features + 1], [n_datapoints, n_features])
	'''
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	index_list = np.random.permutation(np.arange(n_features))
	
	for index in index_list:
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
	'''
		At the terminal node, we determine the predicted value based on most frequent value in this node. 

		Args:
			group : (left, right) dataset [n_datapoints, n_features + 1], [n_datapoints, n_features + 1]

		Returns:
			values[ind] : the most frequent value in this node.
	'''
	# most frequent value
	(values, counts) = np.unique(group[:,-1], return_counts=True)
	ind = np.argmax(counts)

	assert values[ind] == 0.0 or values[ind] == 1.0

	return values[ind]

# Create child splits for a node or make terminal (Dynamic Programming)
	# Binary Split
def split(node, max_depth, min_size, n_features, depth):
	'''
		Main algorithm (CART) by using dynamic programming (DP)
		There are 4 cases
			1. Terminal node : at least, empty container exists between both left and right nodes. 
			2. Depth : For not exceeding max_depth, we do not split the dataset anymore. 
			3. / 4.  Split a left node or a right node

		Args:
			group : (left, right) dataset [n_datapoints, n_features + 1], [n_datapoints, n_features + 1]

		Returns:
			values[ind] : the most frequent value in this node.
	'''	
	
	left, right = node['groups']
	del(node['groups'])

	# Termination condition and keep going in DP
		# check for a no split
	if left.size==0 or right.size==0:
		node['left'] = node['right'] = to_terminal(np.vstack((left, right)))
		return None

	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return None

	# process left child
	if np.shape(left)[0] <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left, n_features)
		split(node['left'], max_depth, min_size, n_features, depth+1)

	# process right child
	if np.shape(right)[0] <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right, n_features)
		split(node['right'], max_depth, min_size, n_features, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
	'''
		Initialize a root node and run the decision tree algorithm 

		Args:
			train : dataset ([n_points, n_features + 1])
			max_depth : max depth (scalar)
			min_size : min_size (scalar)
			n_features : n_features (scalar)

		Return : 
			result : dict {'index', 'value', 'groups'}
			'index' : feature index(scalar) 
			'value' : value of split point in the indiced feature (scalar)
			'groups' : the pointed dictionary 
	'''
	# Initialize a root node
	root = get_split(train, n_features)
	# Run the decision tree with depth 1
	split(root, max_depth, min_size, n_features, 1)

	# In the root node, the followiing nodes are sequentally defined.
	return root

# Predicting func. 
def predict(node, row):
	'''
		Predicting a testing data "row" by using dynamic programming (DP)

		Args:
			node : tree dictionary (index, value, groups)
			row : test data to be predicted [n_features]
		
		Return : 
			return node['left'] when node['left' or 'right'] is terminal, it will be scalar by 'to_terminal' functions. 
	'''

	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

def predicts(tree, dataset):
	'''
		predict test dataset.
	'''
	result = list()
	for datapoint in dataset:
		prediction = predict(tree, datapoint)
		result.append(prediction)
	return np.array(result)

# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
	'''
		make a decision tree using the training dataset and predict test dataset.
		max_depth and min_size are hyper-parameters

		Args:
			train : training dataset ([n_datapoints, n_features + 1])
			test : testing dataset ([n_datapoints, n_features + 1])
			max_depth : scalar
			min_size : scalar
		
		Return : 
			tree : dict (structured dict)
			predictions : the predicted array 
	'''
	tree = build_tree(train, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return tree, np.array(predictions)		

    		
    		
