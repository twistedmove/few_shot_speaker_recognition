import os
import numpy as np
import pandas

def create_partition_few_shot_test(partition_file, num_test_classes=20, num_support=5):
    data1 = np.genfromtxt(partition_file, delimiter=',', dtype=np.str)  
    test_data = []
    test_labels = []
    for i in range(data1.shape[0]):
        target = data1[i,3]
        vid = data1[i,0].split('=')[1]
        vid = "data/audio/{}_{}.wav".format(vid, i)
        test_data.append(vid)
        test_labels.append(target)
    test_data = np.array(test_data)
    test_data = test_data.reshape((test_data.shape[0],1))
    test_labels = np.array(test_labels)
    test_labels = test_labels.reshape((test_labels.shape[0],1))
    test_data = np.concatenate((test_data, test_labels), axis=1)

    labels = {}
    target = test_data[:,1]

    target = [int(x) for x in target]
    for i in range(test_data.shape[0]):
        labels[test_data[i,0]] = target[i]

    np.random.shuffle(test_data)
    # print(data.shape)
    partition = {}
    
    test = test_data
    support = []
    query = []
    for i in range(1,num_test_classes+1):
        elements = []
        for j in range(test_data.shape[0]):
            if int(test_data[j,1])==i:
                elements.append(test_data[j,0])
        elements = np.array(elements)
        elements = list(np.random.permutation(elements))
        support.extend(elements[:num_support])
        query.extend(elements[num_support:])

    return labels,support,query