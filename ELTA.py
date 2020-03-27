
"""
    Python script to submit as a part of the project of ELTP 2020 course.
    
    This script serves as a template. Please use proper comments and meaningful variable names.
"""

"""
    Group Members:
        (1) Pravallika Mavilla
        (2) Shahmir Kazi
        (3) Veerendarnath Naladala
        etc.
"""

"""
    Import necessary packages
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
import spacy
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from bs4 import BeautifulSoup

def normalize_accent(string):
    string = string.replace('á', 'a')
    string = string.replace('â', 'a')

    string = string.replace('é', 'e')
    string = string.replace('è', 'e')
    string = string.replace('ê', 'e')
    string = string.replace('ë', 'e')

    string = string.replace('î', 'i')
    string = string.replace('ï', 'i')

    string = string.replace('ö', 'o')
    string = string.replace('ô', 'o')
    string = string.replace('ò', 'o')
    string = string.replace('ó', 'o')

    string = string.replace('ù', 'u')
    string = string.replace('û', 'u')
    string = string.replace('ü', 'u')

    string = string.replace('ç', 'c')
    
    return string
    
#takes a string, preprocesses and returns the string
def preprocessing(string, spacy_nlp):
    string = string.lower()
    string = normalize_accent(string)
    spacy_tokens = spacy_nlp(string)
    tokens = [token.orth_ for token in spacy_tokens if not token.is_punct]
    string = " ".join(tokens) 
    soup = BeautifulSoup(string, 'lxml')
    html_free = soup.get_text()
    return html_free
    
    
#takes documents and returns array of counts  
def vect(xtr, xvald):
    vectorizer = CountVectorizer()
    xtr = vectorizer.fit_transform(xtr)
    xvald = vectorizer.transform(xvald)
    xtr = xtr.toarray()
    xvald = xvald.toarray()
    return xtr, xvald

#takes count array and gives transformer fit and the tfidf
def transform(xtr, xvald):
    transformer = TfidfTransformer()
    transformer.fit(xtr)
    xtr = transformer.transform(xtr).todense()
    xvald = transformer.transform(xvald).todense()
    return xtr, xvald
  






"""
    Your methods implementing the models.
    
    Each of your model should have a separate method. e.g. run_random_forest, run_decision_tree etc.
    
    Your method should:
        (1) create the proper instance of the model with the best hyperparameters you found
        (2) fit the model with a given training data
        (3) run the prediction on a given test data
        (4) return accuracy and F1 score
        
    Following is a sample method. Please note that the parameters given here are just examples.
"""
def model_random_forest(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = RandomForestClassifier(n_estimators = 100, criterion= 'gini' ,max_depth=2, random_state=0) # please choose all necessary parameters
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")

    return rf_accuracy, rf_f1
    
def model_xgboost(X_train, y_train, X_test, y_test):
    
    clf = XGBClassifier()
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    xg_accuracy = accuracy_score(y_test, y_predicted)
    xg_f1 = f1_score(y_test, y_predicted, average="weighted")

    return xg_accuracy, xg_f1

def model_Extra_trees(X_train, y_train, X_test, y_test):
   
    clf = ExtraTreesClassifier(n_estimators=100, n_jobs= 4)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    et_accuracy = accuracy_score(y_test, y_predicted)
    et_f1 = f1_score(y_test, y_predicted, average="weighted")

    return et_accuracy, et_f1
    
def model_Adaboost(X_train, y_train, X_test, y_test):

     
    DTC = DecisionTreeClassifier(random_state = 11, max_features = 'auto', max_depth = 50)  
    clf = AdaBoostClassifier(base_estimator = DTC)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    ad_accuracy = accuracy_score(y_test, y_predicted)
    ad_f1 = f1_score(y_test, y_predicted, average="weighted")

    return ad_accuracy, ad_f1



def model_bernoullinb(X_train, y_train, X_test, y_test):
    
    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    br_accuracy = accuracy_score(y_test, y_predicted)
    br_f1 = f1_score(y_test, y_predicted, average="weighted")

    return br_accuracy, br_f1    



"""
   The main function should print all the accuracies and F1 scores for all the models.
   
   The names of the models should be sklearn classnames, e.g. DecisionTreeClassifier, RandomForestClassifier etc.
   
   Please make sure that your code is outputting the performances in proper format, because your script will be run automatically by a meta-script.
"""
if __name__ == "__main__":


    Xtrain = pd.read_csv('/home/pravallika/Downloads/X_train.csv',usecols = [1]) #importing X_train
    Xtrain = Xtrain.fillna(' ')
    Ytrain = pd.read_csv('/home/pravallika/Documents/elta/Y_train.csv', usecols = [1])
    spacy_nlp = spacy.load('fr_core_news_sm')
    Xtrain = Xtrain.assign(designation = Xtrain['designation'].apply(lambda x: preprocessing(x,spacy_nlp))) #preprocessing of Xtrain
    
    xtr, xvald, ytr, yvald = train_test_split(Xtrain, Ytrain, test_size = 0.30) #splitting the dataset  


    # getting TF-IDF
    xtrn,xvaldn = vect(xtr['designation'],xvald['designation'] )
    xtrn,xvaldn = transform(xtrn,xvaldn)
    
    #reducing TF-IDF
    svd = TruncatedSVD(n_components=4000)
    xtrn = svd.fit_transform(xtrn)
    xvaldn = svd.transform(xvaldn)
        
    ytr = np.array(ytr).ravel()
    yvald = np.array(yvald).ravel()
    
    print('trainig Random forest')
    rf_acc, rf_f1 = model_random_forest(xtrn, ytr, xvaldn, yvald)
    print('trainig XGboost')
    xg_acc, xg_f1 = model_xgboost(xtrn, ytr, xvaldn, yvald)
    print('trainig extratrees')
    et_acc, et_f1 = model_Extra_trees(xtrn, ytr, xvaldn, yvald)
    print('trainig Adaboost')
    ad_acc, ad_f1 = model_Adaboost(xtrn, ytr, xvaldn, yvald)
    print('bernoulliNB')
    br_acc, br_f1 = model_bernoullinb(xtrn, ytr, xvaldn, yvald)
    
    

    # print the results
    print("random_forest",rf_acc, rf_f1)
    print("xg_boost", xg_acc, xg_f1)
    print(" ExtraTrees", et_acc, et_f1)
    print("Adaboost", ad_acc, ad_f1)
    print("bernoulliNB", br_acc, br_f1)
