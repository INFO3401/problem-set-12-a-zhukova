# Anastasiya Zhukova
# INFO 3401 - Assignment 12
# Answers to problems and tests are at the bottom of the file. Graphics can be found in repo

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import ML support libraries
from sklearn.cross_validation import train_test_split # allows us to divide our data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans


def loadData(dataFile):
	with open(dataFile, 'r') as csvFile:
		data = pd.read_csv(csvFile)

	#Inspect data
	print(data.columns.values)
	return data

def runKNN(dataset, prediction, ignore, neighbors):
	#set up our data set
	X = dataset.drop(columns=[prediction, ignore])
	Y = dataset[prediction].values

	# split data into training and testing
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .4, random_state = 1, stratify = Y)

	#run kNN algorithm, instatiate new object, specify # of neighbors
	knn = KNeighborsClassifier(n_neighbors = neighbors)

	#train the model (X = feature values, Y = what you're tryingto predict)
	knn.fit(X_train, Y_train)

	#Test model
	score = knn.score(X_test, Y_test)
	Y_predict = knn.predict(X_test)
	F1 = f1_score(Y_test, Y_predict, average = 'macro')
	print("Predicts " + prediction + " with " + str(score) + " accuracy")
	print('Chance is ' + str(1.0/len(dataset.groupby(prediction))))
	print('F1: ', F1)

	return knn


def classifyPlayer(targetRow, dataset, model, prediction, ignore):
	X = dataset.drop(columns=[prediction, ignore])

	#Dtermine 5 closstet neighbors to target row
	neighbors = model.kneighbors(X, n_neighbors = 5, return_distance = False)

	#print out neighbors data
	for neighbor in neighbors[0]:
		print(dataset.iloc[neighbor])


def runKNNCrossfold(dataset, kVal, neighbors):
	dataset = dataset.sample(frac = 1)
	test_size = dataset.shape[0]//kVal #tuple of # of rows, # of columns
	test_start = 0
	test_end = test_size

	for k in range(kVal):
		test = dataset.iloc[test_start:test_end]
		train1 = dataset.iloc[0:test_start]
		train2 = dataset.iloc[test_end:]
		test_start += test_size
		test_end += test_size
		train = pd.concat([train1, train2])

		#set up our data set
		X_train = train.drop(columns=['pos', 'player'])
		Y_train = train['pos'].values

		X_test = test.drop(columns=['pos', 'player'])
		Y_test = test['pos'].values

		
		#run kNN algorithm, instatiate new object, specify # of neighbors
		knn = KNeighborsClassifier(n_neighbors = neighbors)

		#train the model (X = feature values, Y = what you're tryingto predict)
		knn.fit(X_train, Y_train)

		score = knn.score(X_test, Y_test)
		print('Fold {} of {} predicts position with an accuracy of {}'.format(k+1, kVal, score))


def determineK(dataset):
	scores = {}

	for val in range(2,11):
		dataset = dataset.sample(frac = 1)
		test_size = dataset.shape[0]//5 #tuple of # of rows, # of columns
		test_start = 0
		test_end = test_size
		k_scores = []
		print('val: ', val)

		for k in range(5):
			test = dataset.iloc[test_start:test_end]
			train1 = dataset.iloc[0:test_start]
			train2 = dataset.iloc[test_end:]
			test_start += test_size
			test_end += test_size
			train = pd.concat([train1, train2])

			#set up our data set
			X_train = train.drop(columns=['pos', 'player'])
			Y_train = train['pos'].values

			X_test = test.drop(columns=['pos', 'player'])
			Y_test = test['pos'].values

			
			#run kNN algorithm, instatiate new object, specify # of neighbors
			knn = KNeighborsClassifier(n_neighbors = val)

			#train the model (X = feature values, Y = what you're trying to predict)
			knn.fit(X_train, Y_train)

			score = knn.score(X_test, Y_test)
			k_scores.append(score)
			print('Fold {} of {} predicts position with an accuracy of {}'.format(k+1, 5, score))

		scores[val] = sum(k_scores)/len(k_scores)
		print(scores[val])
	max_accuracy = max(scores.values())
	
	for k in scores.keys():
		if scores[k] == max_accuracy:
			print(k)
			maxK = k
	print(maxK, max_accuracy)
	print(scores)
			


# find best k: look at the average distance between data points and centroids?
#only feed in dataset, no target, unsupervised learning
def runKMeans(dataset, ignore):
	# set up the dataset
	X = dataset.drop(columns = ignore)

	#run KMeans algorithm
	kmeans = KMeans(n_clusters = 5)
	#train the model
	kmeans.fit(X)
	# Add the predictions to your data frame
	dataset['cluster'] = pd.Series(kmeans.predict(X), index = dataset.index)
	#print a scatter plot matrix
	scatterMatrix = sns.pairplot(dataset.drop(columns = ignore), hue = 'cluster', palette = 'Set2')
	#save the visualization
	scatterMatrix.savefig('kmeansClusters.png')

	return kmeans




def findClusterK(dataset, ignore):
	X = dataset.drop(columns = ignore)
	current_i = []
	improvement_scores = []

	for k in range(1, 21):
		#run KMeans algorithm
		kmeans = KMeans(n_clusters = k)
		#train the model
		kmeans.fit(X)

		inertia = kmeans.inertia_
		if k != 1: #inertia for k=1 is 0, skip over it
			old_i = current_i[-1]
			improvement = old_i - inertia
		else:
			improvement = 0
		
		improvement_scores.append(improvement)
		current_i.append(inertia)
		print('clusters:', k, 'inertia:', inertia, 'improvement:', improvement)
	plt.plot(range(1,21), current_i)
	plt.show()


#########################################################################
######################## Tests Start Here ###############################
#########################################################################

# Need to determine what features (columns) we have to play with
nbaData = loadData('nba_2013_clean.csv')
knnModel = runKNN(nbaData, 'pos', 'player', 5)
print('')
print('')

print('Classifying player:')
classifyPlayer(nbaData.loc[nbaData['player']=='LeBron James'], nbaData, knnModel, 'pos', 'player')
print('')
print('')

print('Running crossfold validation:')
for kval in [5, 7, 10]:
	runKNNCrossfold(nbaData, kval, 5)
print('')
print('')

print('Determine K:')
determineK(nbaData)
print('')
print('')

print('Running K-means')
print('will gnerate a new graphic every time it is run')
# kmeansModel = runKMeans(nbaData, ['pos', 'player'])
print('')
print('')

print('Finding the best number of clusters')
converge = findClusterK(nbaData, ['pos', 'player'])

#############################################################################
######################### Problem Questions #################################
#############################################################################

# 2. Update your k-NN code such that your runKNN function takes an additional 
# parameter standing for the number of neighbors to consider and uses this parameter to perform a classification. 

# 3. Run the classification with a 60/40 split (60% training data, 40% testing data) using 5 neighbors. 
# Compute the F1 score (hint: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
# and the accuracy score for the data. What do these scores tell you about using k-NN to classify a player's 
# position based on their statistics? 

# 4. Create a new function runKNNCrossfold that takes two arguments, 
# one being the dataset and a second being a k-value standing for the number of folds. 
# Update your classifier code to use k-fold cross-validation. Note that you can set up
# this cross validation manually or experiment with SciKit Learn's built-in methods for doing this. 
# Print the accuracy for each fold for k equal to 5, 7, and 10.

# 5. Write a function called determineK which takes a dataset as an argument. 
# Use this function to determine what the optimal setting of k is for kNN, 
# where the best k is the one that maximizes the mean accuracy in your crossfold validation. 
# Print this k and the resulting accuracy. 

# 6. Write a function called runKMeans which takes in a dataset and 
# list of columns to ignore and number of clusters to create. The function should
# return a k-Means clustering on the provided dataset. Use this dataset to create clusters of similar players for your NBA data. 

# 7. Write a function called findClusterK that takes in a dataset and list of columns to ignore
# and determines an optimal k for your k-Means clustering. Then, use this function to
# determine the optimal number of clusters for the NBA data.

#############################################################################
######################### Problem Answers ###################################
#############################################################################


# 3.What do these scores tell you about using k-NN to classify a player's position based on their statistics?
# These scores tell me that the kNN model not very useful for this kind of classification because 
# none of these scores get above a 50% accuracy no matter what fold they use.dataset

# 6. Clusters image can be found in repo

# 7.The optimal number of clusters for this dataset is 5 (probably because there's 5 positions in the NBA). 
#   See ClusterDistances graphic in repo. The inertia (sum of squared improvements) levels out when it reaches 
#   5, which means that past 5 clusters there aren't any improvements being made to the accuracy.




