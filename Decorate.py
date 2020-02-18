
# coding: utf-8

# In[199]:


import pandas as pd
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import KFold
import time
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support as score, precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd


# In[200]:


class Decorate(object):

    def __init__(self, tree_parameters, C_max_size=100, R_size=0.5, Itr_max=300, generator='basic'):
        """
        Initializing the Decorate that will be in use to build the ensemble classifier
        :param tree_parameters: The parameters of the base learning algorithm ('Decision Tree Classifier')
        :param C_max_size: Maximum desired ensemble size
        :param R_size: Factor that determines number of artificial examples to generate
        :param Itr_max: Maximum number of iterations to build an ensemble
        :param generator: 'basic' VS 'GAN'
        """
        self.tree_parameters = tree_parameters
        self.C_max_size = C_max_size
        self.R_size = R_size
        self.Itr_max = Itr_max
        self.C_all = []
        self.classes = None
        self.generator = generator

    def fit(self, X_train, Y_train):
        """
        Runs the DECORATE algorithm from the article ("Creating Diversity In Ensembles Using Artificial Data" p.8)
        :param X_train: The training input samples
        :param Y_train: The target values
        """
        C_i = DecisionTreeClassifier(**self.tree_parameters)
        self.C_all.append(C_i.fit(X_train, Y_train))
        C_size, I_number = 1, 1
        self.classes = pd.unique(Y_train)
        ensemble_error = self._compute_error(X_train, Y_train)
        while C_size < self.C_max_size and I_number < self.Itr_max:
            if self.generator == 'basic':
                training_artificial_examples = self._generate_artificial_examples_basic(X_train)
            else:
                print("training_artificial_examples = self._generate_artificial_examples_GAN(X_train)")
                training_artificial_examples = self._generate_artificial_examples_basic(X_train)
            artificial_labels_examples = self._generate_artificial_labels(training_artificial_examples)
            training_data = pd.concat([X_train, training_artificial_examples], axis=0)
            training_labels = np.concatenate((Y_train, artificial_labels_examples), axis=0)
            C_candidate = DecisionTreeClassifier(**self.tree_parameters)
            self.C_all.append(C_candidate.fit(training_data, training_labels))
            training_error = self._compute_error(X_train, Y_train)
            if training_error <= ensemble_error:
                ensemble_error = training_error
                C_size += 1
            else:
                self.C_all.pop()
            I_number = I_number + 1

    def predict_proba(self, X):
        """
        Predict class probabilities of the input samples X
        :param X: Input samples
        :return:
        """
        probas = np.zeros((X.shape[0], len(self.classes)))
        for tree in self.C_all:
            probas += tree.predict_proba(X)
        return probas / len(self.C_all)

    def predict(self, X):
        """
        Predict class value for X
        :param X: Input samples
        :return:
        """
        probas_list = self.predict_proba(X)
        results = np.argmax(probas_list, axis=1)
        return results

    def _compute_error(self, X, Y):
        """
        Compute ensemble error
        :param X: Input samples
        :param Y: Ground truth (correct) labels
        :return:
        """
        pred = np.argmax(self.predict_proba(X), axis=1)
        ensemble_error = round(1 - accuracy_score(Y, pred), 5)
        return ensemble_error

    def _generate_artificial_examples_basic(self, X):
        """
        Generate artificial training data by randomly picking data points from an approximation of the training-data 
        distribution.
        :param X: Input samples
        """
        number_of_samples = int(X.shape[0] * self.R_size)
        artificial_examples = np.zeros((number_of_samples, X.shape[1]))
        for col_index, column_name in enumerate(X.columns):
            column = X[column_name]
            if len(pd.unique(column)) == 2 and (pd.unique(column)[0] == 0 or pd.unique(column)[0] == 1) and (
                    pd.unique(column)[1] == 0 or pd.unique(column)[1] == 1):
                proba_1 = column.tolist().count(1) / len(column)
                artificial_examples[:, col_index] = np.random.choice([0, 1], size=number_of_samples,
                                                                     p=[1 - proba_1, proba_1])
            else:
                artificial_examples[:, col_index] = np.random.normal(np.mean(column), np.std(column),
                                                                     size=number_of_samples)
        return pd.DataFrame(artificial_examples, columns=X.columns)

    def _generate_artificial_labels(self, training_artificial_examples):
        """
        first, giving to each training artificial example the class membership probabilities predicted by the ensemble.
        then, replace zero probabilities with 0.001 and normalize it. next, labels are selected, such that the 
        probability of selection is inversely proportional to the current ensemble’s predictions.
        :param training_artificial_examples: Generated artificial training data
        """
        labels = np.zeros(training_artificial_examples.shape[0])
        probabilities = self.predict_proba(training_artificial_examples)
        normalized_probabilities = self._normalize_probas(probabilities)

        for prob in range(len(normalized_probabilities)):
            normalized_probabilities[prob] = 1 / normalized_probabilities[prob]
            normalized_probabilities[prob] = normalized_probabilities[prob] / np.sum(normalized_probabilities[prob])
            labels[prob] = np.random.choice(self.classes, p=normalized_probabilities[prob])
        return labels

    def _normalize_probas(self, probabilities):
        """
        Replace zero probabilities with 0.001 and normalize the probabilities to make it a distribution.
        :param probabilities: Probabilities predicted by the ensemble
        """
        for probas in range(len(probabilities)):
            probabilities[probas] = [0.001 if p == 0.0 else p for p in probabilities[probas]] / np.sum(
                probabilities[probas])
        return probabilities


# # Evaluation functions

# In[201]:


def evaluate_i_fold(Y_test, pred, fit_time,prediction_time, total_scores_func, dataset, generator_name):
    """
    Calculating the different scores of each cross validation run and save it to a scores dictionary. 
    :param Y_test: Ground truth (correct) labels
    :param pred: Predicted labels, as returned by a classifier.
    :param fit_time: Amount of tme that the fit function took
    :param prediction_time: Amount of tme that the pred function took
    :param total_scores_func: saves all CV results
    :param dataset: data set name
    :param generator_name: 'basic' VS 'GAN'
    :return: total_scores_func (dictionary)
    """
    configuration = 'c1'
    accuracy = round(accuracy_score(Y_test, pred), 3)
    precision = round(precision_score(Y_test, pred, average='macro'), 3)
    recall = round(recall_score(Y_test, pred, average='macro'), 3)
    f1 = round(f1_score(Y_test, pred, average='macro'), 3)
    total_scores_func[dataset][generator_name][configuration]['fit_time'].append(round(float(fit_time), 2)) 
    total_scores_func[dataset][generator_name][configuration]['prediction_time'].append(round(float(prediction_time), 2)) 
    total_scores_func[dataset][generator_name][configuration]['accuracy'].append(accuracy)
    total_scores_func[dataset][generator_name][configuration]['precision'].append(precision)
    total_scores_func[dataset][generator_name][configuration]['recall'].append(recall)
    total_scores_func[dataset][generator_name][configuration]['f1'].append(f1)
    return total_scores_func

def write_data_set_results_to_csv(dataset,total_scores,generators):
    """
    Writes the CV results to a csv file.
    :param dataset: data set name
    :param total_scores: saves all CV results
    :param generators: 'basic' VS 'GAN'
    """
    configuration='c1'
    for generator_name in generators:
        log_list_test = [dataset,generator_name,
                         np.mean(total_scores[dataset][generator_name][configuration]['fit_time']),
                         np.mean(total_scores[dataset][generator_name][configuration]['prediction_time']),
                         np.mean(total_scores[dataset][generator_name][configuration]['accuracy']),
                         np.mean(total_scores[dataset][generator_name][configuration]['precision']),
                         np.mean(total_scores[dataset][generator_name][configuration]['recall']),
                         np.mean(total_scores[dataset][generator_name][configuration]['f1']) ]
        writer = csv.writer(open("results.csv", "a"), lineterminator='\n', dialect='excel')
        writer.writerow(log_list_test)

def write_headline():
    """
    Writes the file header.
    """
    log_list_header = ['dataset', 'generator',  'fit_time','prediction_time', 'accuracy', 'precision', 'recall', 'f1']
    writer = csv.writer(open("results.csv", "a"), lineterminator='\n', dialect='excel')
    writer.writerow(log_list_header)


# ## Evaluation Utiles 

# In[202]:


data_sets = { 'biodeg': 'datasets/biodeg.csv',
               'glass':'datasets/glass.csv', 'image segmentation':'datasets/image segmentation.csv',
               'Indian Liver Patient Dataset (ILPD)':'datasets/Indian Liver Patient Dataset (ILPD).csv', 'isolet':'datasets/isolet.csv',
               'magic04':'datasets/magic04.csv','movement_libras':'datasets/movement_libras.csv','wilt':'datasets/wilt.csv',
               'Wine_classification':'datasets/Wine_classification.csv'}

generators = ['basic','GAN']

configuration='c1'

tree_parameters={'max_depth': 5, 'min_samples_leaf': 10 , 'class_weight': 'balanced'}


# # Main

# In[203]:


write_headline()
total_scores={}
for dataset in data_sets:
    # Reading and processing the data
    ds = pd.read_csv(data_sets[dataset])
    print("Dataset name: {}".format(dataset))
    ds = ds.dropna()
    le = LabelEncoder()
    Y = ds.pop('class')
    X = pd.get_dummies(ds)
    Y =le.fit_transform(Y)
    Y=pd.DataFrame(data=Y, columns=['class'])
    X, Y = shuffle(X, Y, random_state=10)    
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    total_scores.setdefault(dataset, {})
    
    for generator_name in generators:
        Scores = {
        'c1': {'fit_time': [],'prediction_time': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        }
        total_scores[dataset].setdefault(generator_name, Scores)
        
    # Run the CV
    for train_index, test_index in kf.split(X):
        for generator_name in generators:           
            X_train=X.iloc[train_index]
            X_test=X.iloc[test_index]
            Y_train=Y.iloc[train_index]
            Y_test=Y.iloc[test_index] 
            clf=Decorate(tree_parameters, C_max_size=100, R_size=0.4, Itr_max=100,generator=generator_name)
            start_time = time.time()
            clf.fit(X_train,Y_train['class'].values.tolist())
            fit_time = (time.time() - start_time)
            start_time = time.time()
            pred = clf.predict(X_test)
            prediction_time = (time.time() - start_time)
            seconds = (time.time() - start_time)
            total_scores = evaluate_i_fold(Y_test, pred, fit_time,prediction_time, total_scores, dataset, generator_name)
    write_data_set_results_to_csv(dataset, total_scores, generators)


# In[ ]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

