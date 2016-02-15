import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
import scipy as sc
from sklearn import preprocessing
from sklearn.cross_validation import KFold
import csv

def PerformanceMeasures(predicted_labels , test_labels):

    count = 0
    precision_count = 0
    precision_total = 0
    recall_count = 0
    recall_total = 0

    for i in range(0,len(predicted_labels)):
        if predicted_labels[i] == test_labels[i]:
            count += 1

        # Precision Calculation
        if predicted_labels[i] == 1.0:
            precision_total += 1
            if predicted_labels[i] == test_labels[i]:
                precision_count += 1

        # Recall calculation
        if test_labels[i] == 1.0:
            recall_total += 1
            if predicted_labels[i] == test_labels[i]:
                recall_count += 1



    accuracy = float(count)/len(predicted_labels)
    precision = float(precision_count)/precision_total
    recall = float(recall_count)/recall_total
    f1 = 2*((precision*recall)/(precision+recall))

    return accuracy,precision,recall,f1


def GaussianNaiveBayes(train_samples, train_labels, test_samples, test_labels):

    #This will contain the predicted labels for test data set
    predicted_labels = []
    #Train samples with label = 0 and 1 respectively
    samples_0 = []
    samples_1 = []

    for i in range(0, train_samples.shape[0]):
        if(int(train_labels[i][0]) == 1):
            samples_1.append(train_samples[i])
        else:
            samples_0.append(train_samples[i])

    # Calculate mean and standard deviation for each attribute on separated sets as per labels
    mean_samples_0 = np.array(samples_0).mean(axis = 0)
    std_dev_samples_0 = np.array(samples_0).std(axis = 0)
    mean_samples_1 = np.array(samples_1).mean(axis = 0)
    std_dev_samples_1 = np.array(samples_1).std(axis = 0)
    count  = 0

    for t in range(0,test_samples.shape[0]):
        probability_0 = 1.0
        probability_1 = 1.0

        # Compute Normal distribution for each attribute with the value of that attribute for current test sample and
        # multiply it with the total probabiliy. Consider posterior probability too.
        for i in range(0,test_samples.shape[1]):
            probability_0 *= sc.stats.norm(mean_samples_0[i], std_dev_samples_0[i]).pdf(test_samples[t,i])

        probability_0 *= len(samples_0) / float(len(samples_0) + len(samples_1))

        for i in range(0,test_samples.shape[1]):
            probability_1 *= sc.stats.norm(mean_samples_1[i], std_dev_samples_1[i]).pdf(test_samples[t,i])

        probability_1 *= len(samples_1) / float(len(samples_0) + len(samples_1))

        # Check for which label the probibility is highest and add that label in the predicted_labels
        if probability_0 > probability_1:
            predicted_labels.append(0.0)
        else:
            predicted_labels.append(1.0)


    accuracy,precision,recall,f1 = PerformanceMeasures(predicted_labels , test_labels)

    print "Implementation Accuracy : "
    print accuracy


    print "Implementation Precision : "
    print precision


    print "Implementation Recall : "
    print recall

    print "Implementation F1 Score : "
    print f1
    return accuracy,precision,recall,f1

if __name__ == '__main__':

    #Retrieving the data and loading in Numpy matrix
    #Some preprocessing is required before loading it indo numpy array
    #We need to convert nominal data into continous (technically from alphanumeric to numeric)
    data_file = open(raw_input('Enter file name'), 'r')
    raw_data = [line.split('\t') for line in data_file.readlines()]
    trans_data = [list(x) for x in zip(*raw_data)]

    for row in trans_data:
        ind = trans_data.index(row)
        for val in row:
             if any(c.isalpha() for c in val) == True:
                 le = preprocessing.LabelEncoder()
                 le.fit(list(set(row)))
                 temp =  le.transform(row).tolist()
                 for i in range(0, len(temp)):
                     temp[i] = int(temp[i]) + 1
                 trans_data[ind] = temp
    raw_data = [list(x) for x in zip(*trans_data)]

    for sample in raw_data:
        sample[-1] = sample[-1][:-1]
        for i in range(0,len(sample)):
            sample[i] = float(sample[i])

    data = np.array(raw_data)
    all_samples = data[:,:-1]
    all_labels = data[:,-1:]
    samples_train, samples_test, labels_train, labels_test = train_test_split(all_samples, all_labels, test_size=0.30, random_state=0)

    train_samples =  np.array(samples_train)
    train_labels = np.array(labels_train)
    test_samples = np.array(samples_test)
    test_labels = np.array(labels_test)

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    kf = KFold(all_samples.shape[0], n_folds=10)
    iteration = 1

    total_accuracy,total_precision,total_recall,total_f1 = 0,0,0,0

    for train_index, test_index in kf:

        train_samples, test_samples = all_samples[train_index], all_samples[test_index]
        train_labels, test_labels = all_labels[train_index], all_labels[test_index]

        print "Iteration no. : " , iteration
        accuracy,precision,recall,f1 = GaussianNaiveBayes(train_samples, train_labels, test_samples, test_labels)
        print "---------------------------------------"
        iteration += 1
        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        total_f1 += f1

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    print "Average accuracy : " , total_accuracy / 10
    print "Average precision : " , total_precision / 10
    print "Average recall : " , total_recall / 10
    print "Average F1 measure : " , total_f1 / 10


    clf = GaussianNB()
    clf.fit(train_samples , train_labels)
    print "sci-kit learn accuracy : "
    print clf.score(test_samples, test_labels)

    myfile = open('accuracy.csv', 'wb')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(accuracy_list)
    myfile.close()


    myfile = open('precision.csv', 'wb')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(precision_list)
    myfile.close()


    myfile = open('recall.csv', 'wb')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(recall_list)
    myfile.close()


    myfile = open('f1.csv', 'wb')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(f1_list)
    myfile.close()


