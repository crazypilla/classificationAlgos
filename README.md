# classificationAlgos
CLASSIFICATION ALGORITHMS

Aim:
Implement three classification algorithms: Neural Network, Decision Tree, and Naïve Bayes.
Adopt 10-fold Cross Validation to evaluate the performance of all methods on the provided datasets in terms of Accuracy, Precision, Recall, and F-1 measure.
Sample inputs are attached below as sampleinput1.txt, sampleinput2.txt.
Each line represents one data sample. The last column of each line is class label, either 0 or 1. The rest columns are feature values, each of them can be a real-value (continuous type) or a string (nominal type).

Implementation of the neural network: 
 
• Language Used: Python
• Dataset:  
We chose to split the input data using KFold method of sklearns in python into training set 
and test set and divided the training set into 80% training data and the remaining 20% of
that as validation data. 
It was made sure to handle the case of categorical attributes.
• Feed-forward pass: The input features are applied to 50 hidden input nodes whose output goes to theactivation  function. Maxiter value of 70 was used .  These numbers change with the input.
• Back-propagation: The error in prediction was calculated and transmitted backwards by
updating the weights.  
• Regularization: To avoid over-fitting, regularization was incorporated with term lambda
in error function. 



Implementation of decision Trees:
 
• Language Used: Python
• Dataset:  
Made sure to handle categorical Attributes. 
• Implementation details:
1. Collect data, Splitting data to train and test data: 
 Python’s  Kfold from sklearns  package was used. To deal with categorical attributes, we
used LabelEncoder. 
2.  For selecting the best feature, we have chosen to use infogain.  
3.  For each feature, we computed the info-gain for each value and we select the maximum of     
those as the best. Out of these max, the maximum of those is calculated and is chosen for splitting.
4.  Traverse the tree based on the split at each node till the leaf node is reached.




Implementation of Naïve Bayes Classification Algorithm:
 
• Language Used: Python
• Dataset:  
  Algorithm handles the case with the data set that  had categorical features. 
• Implementation details:
1. First data is read from the input file 
2. Then, if any of the features (columns) are categorical (for eg. Present absent in dataset 2),
they are encoded to numerical values using LabelEncoder in sklearn 
3. This is done because most of the data is numerical. So, we are going to use Gaussian Naïve
Bayes. 
4. Normal distribution is used for predicting class labels in test dataset.
5. For each column in training data, mean and standard deviation values are calculated.
6. Then, while predicting in test data, value of each column in test sample is plugged in the 
formula of normal distribution along with mean and standard deviation values for that
column in training dataset 
7. These mean and standard deviation calculations are obviously per class label. 
8. So, as mentioned earlier, probabilities are calculated for each class label.
9. The one with highest value is chosen.
10. These probabilities are then multiplied with prior probability for the corresponding class 
label as well
11. Final, class label with highest probability is predicted.
12. Once, we have the predicted labels, those are sent to a method called PerformanceMeasures 
which returns accuracy, precision, recall and F1 measure.
13. We use 10-fold cross validation. So, that takes care of splitting the data into train and test 
data.
14. Numpy arrays are used throughout the implementation while working with training and 
testing ,samples and labels. sklearn package KFold is used for getting splits of 10 fold cross
validation.


