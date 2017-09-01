import numpy as np
import sys
from sklearn.decomposition import PCA
import os

#extract the data from data_batch_1
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

#divide train set and test set
def divide_train_and_test(dict, n):
    data         = dict['data']
    data_test    = data[ : n]
    data_test    = data_test.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    data_train   = data[n : 1000]
    data_train   = data_train.reshape(1000 - n, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    labels       = dict['labels']
    labels_test  = np.array(labels[ : n])
    labels_train = np.array(labels[n : 1000])
    return data_test, data_train, labels_test, labels_train

#RGB image to gray image
def RGBtogray(data):
    newdata = np.zeros([data.shape[0], data.shape[1], data.shape[2]], float)
    num    = data.shape[0]
    height = data.shape[1]
    width  = data.shape[2]
    for n in range(num):
        for row in range(height):
            for col in range(width):
                newdata[n, row, col] = 0.299 * data[n, row, col, 0] + 0.587 * data[n, row, col, 1] + 0.114 * data[n, row, col, 2]
    return newdata

#KNN algorithm
def knn(test, data_train, data_labels, k):
    if k <= 0:
        return None
    pre  = np.zeros([test.shape[0]], int)
    for i in range(test.shape[0]):
        diff         = np.tile(test[i], (data_train.shape[0], 1)) - data_train
        sqdiff       = diff ** 2
        sqdistance   = sqdiff.sum(axis=1)
        sortdistance = sqdistance.argsort()
        Count        = {}
        for j in range(k):
            label        = data_labels[sortdistance[j]]
            Count[label] = Count.get(label, 0) + 1
        maxx         = 0
        index        = -1
        for key in Count:
            if Count[key] > maxx:
                maxx  = Count[key]
                index = key
        pre[i] = index
    return pre

#write it to the file
def write(pred, labels):
    if os.path.exists('4251258720.txt'):
        os.remove('4251258720.txt')
    with open('4251258720.txt', 'a') as f:
        for i in range(pred.shape[0]):
            f.write(str(pred[i]) + " " + str(labels[i]) + "\n")


if __name__ == "__main__":
    K    = int(sys.argv[1])
    D    = int(sys.argv[2])
    N    = int(sys.argv[3])
    file = sys.argv[4]

    #upload data
    dict = unpickle(file)
    data_test, data_train, labels_test, labels_train = divide_train_and_test(dict, N)

    #RGB to gray
    data_gray_test  = RGBtogray(data_test)
    data_gray_train = RGBtogray(data_train)

    #reshape for PCA
    data_pca_train = data_gray_train.reshape(1000 - N, 1024)
    data_pca_test = data_gray_test.reshape(N, 1024)

    #PCA
    pca = PCA(n_components = D, svd_solver = 'full')
    data_pca_train = pca.fit_transform(data_pca_train)
    data_pca_test  = pca.transform(data_pca_test)

    #begin knn
    predict = knn(data_pca_test, data_pca_train, labels_train, K)

    #write it to the file
    write(predict, labels_test)

    #output
    for i in range(predict.shape[0]):
        print(predict[i], labels_test[i])

