import numpy as np
from scipy.optimize import minimize
import math
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.cross_validation import KFold

def initializeWeights(in_val,out_val):
   
    epsilon = math.sqrt(6) / math.sqrt(in_val+ out_val + 1);
    W = (np.random.rand(out_val, in_val + 1)*2* epsilon) - epsilon;
    #W=np.zeros((n_in,n_out))
    return W



def makemat(list):
    newmatt=np.zeros((len(list),2))
    newmat = [[0 for x in range(2)] for x in range(len(list))] 
    for i in range(0,len(list)):
        if list[i]==0 :
           newmatt[i][0]=1
           newmatt[i][1]=0
        else :
            newmatt[i][0]=0
            newmatt[i][1]=1
    return newmatt


def activation_function(z):
    return  .5 * (1 + np.tanh(.5 * z))
    
def objective_function(params, *args):

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    n_samples = training_data.shape[0];
    inp = np.zeros((1,n_input+1))
    hid = np.zeros((1,n_hidden+1))
    out = np.zeros((1,n_class))
    grad_w1 = np.zeros(w1.shape)
    grad_w2 = np.zeros(w2.shape)
    gradw1sum = np.zeros(w1.shape)
    gradw2sum = np.zeros(w2.shape)
    obj_val = 0
    lab = np.array([0]*10)
    hid[0][n_hidden] = 1 #setting bias terms
    for sam in range(n_samples):
        inp = np.concatenate((training_data[sam,:],[1])) #appending bias terms
        lab = training_label[sam,:]
        # feed forward
        inp = inp.reshape(1,inp.size)
        hid[0][:-1] = np.dot(inp,w1.T)
        hid[0][:-1] = activation_function(hid[0][:-1])
        out = np.dot(hid,w2.T)
        out = activation_function(out)
        # back propagation
        obj_val_temp = lab*np.log(out) + (1-lab)*np.log(1 - out)
        obj_val += np.sum(obj_val_temp)
        delval = out - lab
        grad_w2 = np.dot(delval.T,hid)
        temp = np.zeros((1,n_hidden))
        temp = np.dot(delval,w2)
        temp1 = ((1-hid[0][:-1])*hid[0][:-1]*temp[0][:-1])
        temp1 = temp1.reshape(1,temp1.size)
        grad_w1 = np.dot(temp1.T,inp)
        gradw1sum += grad_w1
        gradw2sum += grad_w2
    
    w1sum = np.sum(np.square(w1))
    w2sum = np.sum(np.square(w2))
    
    obj_val += lambdaval*(w1sum + w2sum)/2    
    grad_w1 = (gradw1sum + lambdaval*w1)/n_samples
    grad_w2 = (gradw2sum + lambdaval*w2)/n_samples    
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_val = -1*obj_val/n_samples
    return (obj_val,obj_grad)

def Predict(w1,w2,data):
    
    n_input = w1.shape[1] - 1
    n_hidden = w1.shape[0]
    n_class = w2.shape[0]
    n_samples = data.shape[0];
    inp = np.zeros((1,n_input+1))
    hid = np.zeros((1,n_hidden+1))
    out = np.zeros((1,n_class))
    label = np.array([])
    #print label
    hid[0][n_hidden] = 1 #setting bias terms
    initLabel = False
    for sam in range(n_samples):
        inp = np.concatenate((data[sam,:],[1])) #appending bias terms
        inp = inp.reshape(1,inp.size)
        # feed forward
        hid[0][:-1] = np.dot(inp,w1.T)
        hid[0][:-1] = activation_function(hid[0][:-1])
        out = np.dot(hid,w2.T)
        out = activation_function(out)
        temp = np.array([0]*n_class)
        temp[np.argmax(out)] = 1
        if (initLabel):
            label = np.vstack((label,temp))
        else:
            label = np.copy(temp)
            initLabel = True
            
    return label


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
    if(precision_total==0):
        precision=1
    else:    
        precision = float(precision_count)/precision_total
    recall = float(recall_count)/recall_total
    f1 = 2*((precision*recall)/(precision+recall))

    return accuracy,precision,recall,f1

def makelabel(arr):
    pred_label=()
    for row in arr:
        pred_label=np.hstack((pred_label,np.argmax(row)))
    return pred_label

def split(train_samples,train_labels):
    print"input shapes"
    print train_samples.shape
    print train_labels.shape 
    print"*********end*********"
    num_train=int(round(0.8*train_samples.shape[0]))
    print "num train"
    print num_train
    num_val=int(train_samples.shape[0]-num_train)
    print "num val"
    print num_val
    matt=np.zeros((1,train_samples.shape[1]))
    matt1=np.zeros((1,train_labels.shape[1]))
    
    for r in range(0,num_train):
            matt=np.vstack((matt,train_samples[r].reshape(1,matt.shape[1])))
            matt1=np.vstack((matt1,train_labels[r].reshape(1,matt1.shape[1])))
    # matt
    #print matt
    #print matt1
    matt2=np.zeros((1,train_samples.shape[1]))
    matt3=np.zeros((1,train_labels.shape[1]))
    for r in range(num_train,num_val+num_train):
            matt2=np.vstack((matt2,train_samples[r].reshape(1,matt2.shape[1])))
            matt3=np.vstack((matt3,train_labels[r].reshape(1,matt3.shape[1])))
    
    
    num_feat=train_samples.shape[1]     
    train= matt[1:,:].reshape((num_train,num_feat))
    
    train_label=matt1[1:,:].reshape((num_train,))
    
    val= matt2[1:,:].reshape((num_val,num_feat))
    val_label=matt3[1:,:].reshape((num_val,))
    
 
    
    return (train),makemat(train_label),(val),makemat(val_label)

 

def neuralnetwork(train_data,train_label,val_data,val_label,test_data,test_label):
    
    n_input=train_data.shape[1]
    n_hidden = 25;			   
    n_class = 2;
    
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class);
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
    lambdaval = 0.00
    opts = {'maxiter' :10}
    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
    nn_params = minimize(objective_function, initialWeights, jac=True, args=args,method='CG', options=opts)
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    predicted_label = Predict(w1,w2,train_data)
    temp_label=np.copy(predicted_label)
    print('\n Training set Accuracy:' + str(100*np.mean(np.argmax(predicted_label,1) == np.argmax(train_label,1))) + '%')
    predicted_label2 = Predict(w1,w2,val_data)
    print('\n Validation set Accuracy:' + str(100*np.mean(np.argmax(predicted_label2,1) == np.argmax(val_label,1))) + '%')
    predicted_label3 = Predict(w1,w2,test_data)
    print('\n Test set Accuracy:' + str(100*np.mean(np.argmax(predicted_label3,1) == np.argmax(test_label,1))) + '%')
    y_pred=makelabel(predicted_label3)
    y_actual=makelabel(test_label)
    
    return y_actual,y_pred


if __name__ == "__main__":
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

    mat = np.array(raw_data)
    print mat.shape 
    all_samples=mat[:,:-1]
#print all_samples.shape
    all_labels=mat[:,mat.shape[1]-1]
    
    kf = KFold(all_samples.shape[0], n_folds=10)
    iteration = 1

    total_accuracy,total_precision,total_recall,total_f1 = 0,0,0,0

    for train_index, test_index in kf:

        train_samples, test_data = all_samples[train_index], all_samples[test_index]
        train_labels, test_labels = all_labels[train_index], all_labels[test_index]
        t_labels=train_labels.reshape((train_labels.shape[0],1))
        train_data,train_l,val_data,val_l= split(train_samples,t_labels)
        print "Iteration no. : " , iteration
        y_actual,y_predict=neuralnetwork(train_data,(train_l),val_data,(val_l),test_data,makemat(test_labels))
        accuracy,precision,recall,f1 = PerformanceMeasures(y_predict,y_actual)
        print accuracy
        print precision
        print recall
        print f1
        print "---------------------------------------"
        iteration += 1
        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        total_f1 += f1


    print "Average accuracy : " , total_accuracy / 10
    print "Average precision : " , total_precision / 10
    print "Average recall : " , total_recall / 10
    print "Average F1 measure : " , total_f1 / 10
    
    #print PerformanceMeasures(y_pred,y_actual)
    
    
    #print precision_recall_fscore_support(test_label, predicted_label3, average='weighted')
   
       

