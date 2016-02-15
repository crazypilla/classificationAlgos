_author__ = 'Harshita V'
import numpy as np
import math
from sklearn import preprocessing
from sklearn.cross_validation import KFold

class Node:
    def __init__(self,ind,par):
        self.name = ind
        self.data = {}
        self.labels = {}
        self.lab = None
        self.left = None
        self.right = None
        self.isleaf = False
        self.parent = par
        self.split = None

class DecisionTree:
    def __init__(self):
            self.split_list = []
    
    def build_tree(self,root,features,data,label,entropy_dataset):
        if len(features) == 0:
            return root

        if len(set(label)) == 1:
            root.isleaf = True
            root.lab = list(set(label))[0]
            return root

        j = root.name
        root.data[0],root.data[1] = [],[]
        root.labels[0],root.labels[1] = [],[]
        for i in range(len(data)):
            if data[i,j] <= root.split:
                root.data[0].append(data[i,:])
                root.labels[0].append(label[i])
            else:
                root.data[1].append(data[i,:])
                root.labels[1].append(label[i])

        root.data[0], root.data[1] = np.array(root.data[0]),np.array(root.data[1])
        root.labels[0],root.labels[1] = np.array(root.labels[0]),np.array(root.labels[1])
        #print root.data[0].shape,root.data[1].shape
        features1 = features.copy()
        features2 = features.copy()

        if len(root.data[0]) != 0:

            entropy_dataset = self.entropy(list(root.labels[0]).count(0),list(root.labels[0]).count(1))
            best_feat_left,best_ig = self.infogain(features1,entropy_dataset,root.data[0],root.labels[0])
            #print "left ", features,best_feat_left,len(root.data[0])
            #features1.remove(best_feat_left)
            child = Node(best_feat_left,root.name)
            child.split = best_ig
            root.left = self.build_tree(child,features1,root.data[0],root.labels[0],entropy_dataset)
        if len(root.data[1]) != 0:

            entropy_dataset = self.entropy(list(root.labels[1]).count(0),list(root.labels[1]).count(1))
            best_feat_right,best_ig = self.infogain(features2,entropy_dataset,root.data[1],root.labels[1])
            #print "right ",features,best_feat_right,len(root.data[1])
            #features2.remove(best_feat_right)
            child = Node(best_feat_right,root.name)
            child.split = best_ig
            #print set(root.data[1][:,child.name])
            root.right = self.build_tree(child,features2,root.data[1],root.labels[1],entropy_dataset)

        return root

    def entropy(self,feat1,feat2):
        if feat1==0 and feat2==0:
            return 0.0
        p1 = float(feat1)/(feat1+feat2)
        p2 = float(feat2)/(feat1+feat2)
        e1,e2=0,0
        if p1 != 0.0:
            e1 = p1*math.log(p1,2)
        if p2 != 0.0:
            e2 = p2*math.log(p2,2)
        return - (e1+e2)


    def infogain(self,features,entropy_dataset,data,label,):
        best_col_ig,best_feat,best_feat_val = 0.0,0,0.0
        for best_feat in features:
            break

        for j in features:
            col = data[:,j]
            best_ig,best_val = 0.0,0
            for val in col:
                less = sum(i<=val for i in col)
                more = sum(i>val for i in col)
                l0,l1,m0,m1 = 0,0,0,0
                for i in range(data.shape[0]):
                    if data[i,j] <= val:
                        if label[i] == 0:
                            l0 += 1
                        else:
                            l1 += 1

                    else:
                        if label[i] == 0:
                            m0 += 1
                        else:
                            m1 += 1
                ig = entropy_dataset -((float(less)/(less+more))*self.entropy(l0,l1) + (float(more)/(less+more))*self.entropy(m0,m1))

                if ig > best_ig:
                    best_ig = ig
                    best_val = val
            if best_ig > best_col_ig:
                best_feat_val = best_val
                best_col_ig = best_ig
                best_feat = j
        #print best_col_ig
        #print best_feat_val
        return best_feat,best_feat_val

    def predict(self,data,root):
        result = []
        for i in range(len(data)):
            row = data[i]

            tmp = root
            while(True):
                #if tmp == None:
                    #result.append(0.0)
                    #break
                j = tmp.name
                if tmp.isleaf == True:
                    result.append(tmp.lab)
                    break
                if row[j] <= tmp.split:
                    tmp = tmp.left
                elif row[j] > tmp.split:
                    tmp = tmp.right
        return result


    def training(self,data,label):
        features = set(np.arange(0,data.shape[1],1))
        entropy_dataset = self.entropy(list(label).count(0),list(label).count(1))
        best_feat,best_ig = self.infogain(features,entropy_dataset,data,label)
        #features.remove(best_feat)
        root = Node(best_feat,None)
        root.split = best_ig
        root = self.build_tree(root,features,data,label,entropy_dataset)
        return root


def cal_accuracy(pred,actual):
    count = 0
    for i in range(len(actual)):
        if pred[i] == actual[i]:
            count += 1

    return float(count)/len(actual)*100

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

def main():

    data_file = open('ds1.txt', 'r')
    raw_data = [line.split('\t') for line in data_file.readlines()]
    trans_data = [list(x) for x in zip(*raw_data)]

    for row in trans_data:
        ind = trans_data.index(row)
        for val in row:
             if any(c.isalpha() for c in val) == True:
                le = preprocessing.LabelEncoder()
                le.fit(list(set(row)))
                temp =  le.transform(row).tolist()
                #print "Categorical attribute converted : " , row
                #print "Categorical attribute converted : " , temp
                for i in range(0, len(temp)):
                    temp[i] = int(temp[i]) + 1
                    trans_data[ind] = temp
                break
    raw_data = [list(x) for x in zip(*trans_data)]

    for sample in raw_data:
        sample[-1] = sample[-1][:-1]
        for i in range(0,len(sample)):
            sample[i] = float(sample[i])
    data1 = np.array(raw_data)
    
    dc = DecisionTree()
    dc.split_list = [0 for i in range(data1.shape[1] - 1)]

    kf = KFold(data1.shape[0], n_folds=10)
    iteration = 1
    average_accuracy = 0.0
    accuracy,precision,recall,f1 = [],[],[],[]
    for train_index, test_index in kf:
    #for m in range(1):
        print "Iteration : " , iteration
      

        data1 = np.array(raw_data)
        #np.random.shuffle(data1)
        train,test = data1[train_index], data1[test_index]
        #p = data1.shape[0]*0.7
        #train,test = data1[0:p,:], data1[p:,:]

        train_data,train_label = train[:,0:(train.shape[1]-1)],train[:,(train.shape[1]-1):].reshape((train.shape[0]))
        test_data,test_label = test[:,0:(test.shape[1]-1)],test[:,(test.shape[1]-1):].reshape((test.shape[0]))

        #data = dc.preprocess(train_data,train_label)
        #np.random.shuffle(data)
        root = dc.training(train_data,train_label)
        #print leaf_nodes
        print "Completed train"
        pred = dc.predict(test_data,root)
        print cal_accuracy(pred,test_label)
        average_accuracy += cal_accuracy(pred,test_label)
        acc,pre,rec,f = PerformanceMeasures(pred,test_label)
        accuracy.append(acc)
        precision.append(pre)
        recall.append(rec)
        f1.append(f)
        iteration += 1

    print "Average Accuracy after 10-fold cross validation : " , average_accuracy/10
    print accuracy
    print precision
    print recall
    print f1

global split_list
if __name__=='__main__':

    main()
