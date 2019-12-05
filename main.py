import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz
import graphviz
from cleaning import clean_data, delete_dimension

newData = clean_data(pd.read_csv('data.csv'))
# Eliminating education from the dimensions
newData = delete_dimension(newData, 3)

# Turning all labels into float type
for column_name in newData.columns:
    if newData[column_name].dtype == object:
        newData[column_name] = preprocessing.LabelEncoder().fit_transform(newData[column_name])
    else:
        pass

#Splitting the data into training and test
numDataPoints = len(newData)
index=np.random.permutation(numDataPoints)
## Dividing 75% for training
temp=int(numDataPoints*0.75)
indexTrain= index[0:temp-1]
indexTest= index[temp:numDataPoints-1]
dataTrain = newData.iloc[indexTrain,:]
dataTest = newData.iloc[indexTest,:]

#Splitting the label of the data
dataTrainLabel = dataTrain.iloc[:,dataTrain.shape[1]-1]
dataTestLabel = dataTest.iloc[:,dataTest.shape[1]-1]
dataTrain = dataTrain.iloc[:,0:dataTrain.shape[1]-1]
dataTest = dataTest.iloc[:,0:dataTest.shape[1]-1]


def DecisionTrees(X_train,Y_train, depth):
    model = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
    model.fit(X_train,Y_train)
    return(model)

def ShowTree(model,columns,targetnames, dot_file_name):
    data = export_graphviz(model,out_file=dot_file_name,
                         feature_names=columns,
                         class_names=targetnames,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = graphviz.Source(data)
    return(graph)

depths = [7] ## use [5,7,10,20,50,100] or any other list, [7] is only for the last part

# Reducing dimensionality
dataTrain = dataTrain.iloc[:,[0,1,2,3,5,6,9,10,11]]
dataTest = dataTest.iloc[:,[0,1,2,3,5,6,9,10,11]]
results = []
for depth in depths:
    TrainedTree = DecisionTrees(dataTrain,dataTrainLabel, depth)
    PredictionTree = TrainedTree.predict(dataTest)

    print("Accuracy:",metrics.accuracy_score(dataTestLabel, PredictionTree))
    print(metrics.classification_report(dataTestLabel, PredictionTree))
    print(metrics.confusion_matrix(dataTestLabel, PredictionTree))

    targetnames = [str(i) for i in list(set(dataTrainLabel))]
    arbol = ShowTree(TrainedTree,list(dataTrain.columns),targetnames, 'images/tree_{}.dot'.format(depth))
    results.append({'dept_{}'.format(depth): [metrics.accuracy_score(dataTestLabel, PredictionTree),
                   metrics.classification_report(dataTestLabel, PredictionTree),
                   metrics.confusion_matrix(dataTestLabel, PredictionTree)]})
    # # Makes tree.dot into a visualizable tree.png
    from subprocess import call
    call(['dot', '-Tpng', 'images/tree_{}.dot'.format(depth), '-o', 'images/tree_depth_{}.png'.format(depth), '-Gdpi=300'])

print(results)


## DataFrame to evaluate

evalData = clean_data(pd.read_csv('evalData.csv'))
# Eliminating education from the dimensions
evalData = delete_dimension(evalData, 3)

# Turning all labels into float type
for column_name in evalData.columns:
    if evalData[column_name].dtype == object:
        evalData[column_name] = preprocessing.LabelEncoder().fit_transform(evalData[column_name])
    else:
        pass

#Splitting the data into training and test
numDataPoints = len(evalData)
index=np.random.permutation(numDataPoints)
## Dividing 75% for training
temp=int(numDataPoints*0)
indexTrain= index[0:temp-1]
indexTest= index[temp:numDataPoints-1]
dataTrain = newData.iloc[indexTrain,:]
dataTest = newData.iloc[indexTest,:]

#Splitting the label of the data
dataTrainLabel = dataTrain.iloc[:,dataTrain.shape[1]-1]
dataTestLabel = dataTest.iloc[:,dataTest.shape[1]-1]
dataTrain = dataTrain.iloc[:,0:dataTrain.shape[1]-1]
dataTest = dataTest.iloc[:,0:dataTest.shape[1]-1]

dataTrain = dataTrain.iloc[:,[0,1,2,3,5,6,9,10,11]]
dataTest = dataTest.iloc[:,[0,1,2,3,5,6,9,10,11]]


TrainedTree = DecisionTrees(dataTrain,dataTrainLabel, 7)
PredictionTree = TrainedTree.predict(dataTest)

print("Accuracy:",metrics.accuracy_score(dataTestLabel, PredictionTree))
print(metrics.classification_report(dataTestLabel, PredictionTree))
print(metrics.confusion_matrix(dataTestLabel, PredictionTree))

targetnames = [str(i) for i in list(set(dataTrainLabel))]
arbol = ShowTree(TrainedTree,list(dataTrain.columns),targetnames, 'images/test_tree_{}.dot'.format(7))
results.append({'dept_{}'.format(7): [metrics.accuracy_score(dataTestLabel, PredictionTree),
                metrics.classification_report(dataTestLabel, PredictionTree),
                metrics.confusion_matrix(dataTestLabel, PredictionTree)]})
# # Makes tree.dot into a visualizable tree.png
yes = metrics.confusion_matrix(dataTestLabel, PredictionTree)[1][1] + metrics.confusion_matrix(dataTestLabel, PredictionTree)[0][1]
no = metrics.confusion_matrix(dataTestLabel, PredictionTree)[0][0] + metrics.confusion_matrix(dataTestLabel, PredictionTree)[1][0]
print('{} yes and {} no'.format(yes, no))

final = pd.DataFrame(np.array([True for x in range(yes)] + [False for x in range(no)]), columns=['predictions'])
print(final)

final.to_csv('results', encoding='utf-8', index=False)

from subprocess import call
call(['dot', '-Tpng', 'images/test_tree_{}.dot'.format(7), '-o', 'images/test_tree_depth_{}.png'.format(7), '-Gdpi=300'])
