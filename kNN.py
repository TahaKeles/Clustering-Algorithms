from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

def euclidean_distance(row1,row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance+=(row1[i]-row2[i])**2
    return sqrt(distance)


def get_neighbours(train,testrow,number_of_neighbours):
    distances = list()
    #indexes = list()
    #i=0
    for train_row in train:
        dist = euclidean_distance(train_row[0:len(train_row)-1],testrow)
        distances.append((train_row,dist))


    distances.sort(key=lambda tup:tup[1])
    neighbours = list()
    for i in range(number_of_neighbours):
        neighbours.append(distances[i][0])
    return neighbours

def predict_classification(train,test_row,num_of_neigh):
    neighbour = get_neighbours(train,test_row,number_of_neighbours=num_of_neigh)
    output_values =  [row[-1] for row in neighbour]
    prediction = max(set(output_values),key=output_values.count)
    return prediction

def evaluatekNN(test_labels1,test_data1,new_train1,kvalue):

    results = []
    for a in kvalue:
        i = 0
        acc = 0
        for testing in test_data1:
            predict1 = int(predict_classification(new_train1,testing,a))
            if predict1 == test_labels1[i]:
                acc+=1
            i+=1
        results.append(acc/len(test_labels))


    plt.plot(kvalue, results)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Accuracy')
    plt.savefig("accuracy.png")
    return results

def crossvalidationgraph(val_label,val_data,new_train2,rangee):
    results = []

    for a in rangee:
        i = 0
        acc = 0
        for testing in val_data:
            predict1 = int(predict_classification(new_train2,testing,a))
            if predict1 == val_label[i]:
                acc+=1
            i+=1
        results.append(acc/len(val_label))


    plt.plot(rangee, results)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.savefig("crossaccuracy.png")
    plt.clf()
    #plt.show()
    return results


if __name__ == '__main__':

    train_data = np.load("./data/knn/train_data.npy")
    train_labels = np.load("./data/knn/train_labels.npy")
    test_data = np.load("./data/knn/test_data.npy")
    test_labels = np.load("./data/knn/test_labels.npy")
    new_train = list()
    new_test = list()
    i=0
    for train_data_row in train_data:
        new_train_row=np.append(train_data_row,[train_labels[i]])
        new_train.append(new_train_row)
       #train_data_row.append(train_labels[i])
        i+=1
    i=0
    new_train_valid= list()
    val_data = list()
    val_labels = list()
    index =0
    for each in new_train:
        if index<(0.8 * len(new_train)): # splitting 0.8 ratio
            new_train_valid.append(each)
        else:
            val_data.append(each)
            val_labels.append(train_labels[index])
        index+=1
    # print(len(new_train_valid))
    # print(len(val_data))
    # print(len(val_labels))
    # print(val_labels)
    rarara = range(1,200)  # defining k range for validation graph
    crossvalidationgraph(val_labels, val_data, new_train_valid, rarara)
    rarara1 = range(1,200) #defining k range for test accuracy graph
    evaluatekNN(test_labels,test_data,new_train_valid,rarara1)







