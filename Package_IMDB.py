# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 17:12:47 2025

@author: SuSwim

IMDB Sentiment Classification Toolkit
"""

import os
import re
import nltk
import pickle
import collections
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer



def clean_text(str_in):
    """
    Clean the input text by removing non-alphabetic characters and converting to lowercase.
    
    The function uses regular expressions to:
        - Remove all characters that are not English letters, space, or apostrophes.
        - Strip leading/trailing whitespace.
        - Convert all characters to lowercase.

    Parameters
    ----------
    str_in : str
        The input text string to be cleaned.

    Returns
    -------
    clean_t : str
        A cleaned text, containing only lowercase alphabetic characters, space, and apostrophes.
        
    Example
    -------
    clean_text("Hello, world! I'm clean_text ")
    'hello world i'm clean text'

    """
    clean_t = re.sub(r"[^A-Za-z\s']+", " ", str_in).strip().lower()
    return clean_t 

def read_file(path_in):
    """
    Reads the content of a file, cleans the text, and returns it as a string.
    
    The function try to open a text file at the given path using UTF-8 encoding, and using 'clean_text(str_in)' to clean the content.
    If the file cannot be opened, it will print an error message.

    Parameters
    ----------
    path_in : str
        A full file path to read.

    Returns
    -------
    text_t : str
        Cleaned file content, or 'None' if the file cannot be opened.

    Note
    ----------
    Depends on an external function 'clean_text(str_in)'
    """
    text_t = None
    try:
        f = open(path_in, "r", encoding="UTF8")
        text_t = clean_text(f.read())
        f.close()
    except:
        print ("Hey", path_in, "does NOT exist!!")
        pass
    return text_t

def file_crawler(path_in):
    """
    Recursively traverses all subdirectories and read file contents.
    
    Files content is stored along with the name of the folder it resides in, served as its label.
    The function returns a pandas Dataframe with two columns:
        -'body': the content of each file
        -'label': the name of the folder
        
    Parameters
    ----------
    path_in : str
        The root directory path to search for files. Subdirectories are treated as labels.
            
    Returns
    ----------
    tmp_pd : pandas.DataFrame
        -'body':str
            file content
        -'lable':str
            folder name(in this case is 'pos' or 'neg')
                
    Note
    ----------
    Depends on an enternal function 'read_file(path_in)'
    """
    tmp_pd = pd.DataFrame()
    for root, dirs, files in os.walk(path_in, topdown=False):
       for name in files:
           tmp = read_file(root + "/" + name) # read file
           if tmp is not None:
               if len(tmp) != 0:
                   t_pd = pd.DataFrame(
                       {"body": tmp,
                        "label": root.split("/")[-1]}, index=[0])
                   tmp_pd = pd.concat([tmp_pd, t_pd], ignore_index=True)
    return tmp_pd

def word_freq(corp_in):
    """
    Calculates word frequency from a given input string.
    
    The input text is split by whitespace into words, and use Python's 'collections.Counter' to compute each word frequency.
    
    The result is returned as a dictionary.

    Parameters
    ----------
    corp_in : str
        A string would be analyzed.

    Returns
    -------
    wrd_freq_new : dict
        Keys are words, and values are frequencies.
    
    Example
    -------
    word_freq("hello world hello")
    {'hello':2, 'world':1}
    """
    wrd_freq_new = dict(collections.Counter(corp_in.split()))
    return wrd_freq_new

def wrd_main(df_in, col_in):
    """
    Compute word frequency for each label.
    
    Use the function 'word_freq(corp_in)' to calculate the word frequency, and stores the result in a dictionary.
    
    There are a new label named "all" which contains all word frequency.
    
    Parameters
    ----------
    df_in : pandas.DataFrame
        The input DataFrame containing text and its label.
    col_in : str
        The name of the column in 'df_in'.

    Returns
    -------
    wrd_dict : dict
        A dictionary mapping label and its word frequency dictionary.

    Note
    -------
    Depends on an enternal function 'word_freq(corp_in)'
    """
    topics = list(df_in.label.unique())
    topics.append("all")
    wrd_dict = dict()
    for t in topics:
        if t == "all":
            tmp = df_in[col_in].str.cat(sep=" ")
        else: 
            tmp = df_in[df_in[
                "label"] == t][col_in].str.cat(sep=" ")
        wrd_dict[t] = word_freq(tmp)   
    return wrd_dict

def rem_sw(str_in):
    """
    Remove English stopwords.
    
    This function uses NLTK's predefined list of English stopworks.
    
    The remaining words are joined back into single space-separated string.

    Parameters
    ----------
    str_in : str
        A string of text.

    Returns
    -------
    ex_text : str
        A string without stopwords.
        
    Example
    -------
    rem_sw("it is a simple example")
    'simple example'
        
    Note
    -------
    Need to download the NLTK stopwords corpus.
    >>>import nltk
    >>>nltk.download('stopwords')
    """
    sw = nltk.corpus.stopwords.words('english')
    filt = [word for word in str_in.split() if word not in sw]
    ex_text = ' '.join(filt)
    return ex_text

def stem_fun(str_in, sw_in):
    """
    Applies stemming or lemmatization to the input text string.
    
    According to 'sw_in' argument to stems or lemmatizes each word.

    Parameters
    ----------
    str_in : str
        A input string containing whitespace-separated words.
    sw_in : str
        The method to applied:
            - "stem" : stemming.
            - others : WordNet lemmatization

    Returns
    -------
    stem_fun : str
        A string which applied stemming or lemmatization. 
        
    Example
    -------
    stem_fun("studies played going", "stem")
    "studi play go"
    
    stem_fun("studies played going", "stem")
    "study played going"
    
    Note
    -------
    For lemmatization, need to download the NLTK WordNet corpus.
    >>>import nltk
    >>>nltk.download('wordnet')

    """                      
    if sw_in == "stem":
        stem = nltk.stem.PorterStemmer()
        stem_fun = [stem.stem(word) for word in str_in.split()]
    else:
        stem = nltk.stem.WordNetLemmatizer()
        stem_fun = [stem.lemmatize(word) for word in str_in.split()]
    stem_fun = ' '.join(stem_fun)
    return stem_fun

def write_pickle(obj_in, path_in, name_in): 
    """
    Serializes and writes a Python object to a '.pk' file using pickle.

    Parameters
    ----------
    obj_in : any
        The python object to be serialized and saved.
    path_in : str
        The path where the file saved(must end with '/' or '\\').
    name_in : str
        The saved pickle file name..

    Returns
    -------
    None.
    """
    pickle.dump(obj_in, open(path_in + name_in + ".pk", 'wb'))

def read_pickle(path_in, name_in):
    """
    Loads a python object from a '.pk' file using pickle.

    Parameters
    ----------
    path_in : str
        Where the '.pk' file is located(must end with '/' or '\\').
    name_in : str
        The filename(without extension) of the pickle file to read.

    Returns
    -------
    the_data_t : any
        The python object that was stored in the pickle file.

    """
    the_data_t = pickle.load(
        open(path_in + name_in + ".pk", 'rb'))
    return the_data_t

def transform_fun(df_in, col_in, m_in, n_in, p_in, nm_in):
    """
    Using TF or TF-IDF to transforms a text column in a DataFrame into vectorized feature matrix.
    
    The function uses 'CountVectorizer' or 'TfidfVectorizer' from scikit-learn to convert text data into a numeric matrix of token counts or TF-IDF scores.
    
    Save the fitted vectorizer as a pickle file.

    Parameters
    ----------
    df_in : pandas.DataFrame
        The input DataFrame containing a text column to vectorize.
    col_in : str
        The name of the column of 'df_in'.
    m_in : int
        The minimum n-gram size.
    n_in : int
        The maximum n-gram size.
    p_in : str
        The vectorizer saved path.
    nm_in : str
        The name of the vectorizer to be used and saved.
        - 'tf' : Use CountVectorizer (term frequency)
        - other : Use TfidfVectorizer (TF-IDF)

    Returns
    -------
    xform_data : pandas.DataFrame
        column - feature.
        row - vectorized document.
    """
    if nm_in == "tf":
        cv = CountVectorizer(ngram_range=(m_in, n_in))
    else:
        cv = TfidfVectorizer(ngram_range=(m_in, n_in))
    xform_data = pd.DataFrame(
        cv.fit_transform(df_in[col_in]).toarray())
    xform_data.columns = cv.get_feature_names_out()
    write_pickle(cv, p_in, nm_in)
    return xform_data

def pca_fun(df_in, var_in, o_path, n_in):
    """
    Applied PCA to reduce the dimensionality of input features.
    
    Prints the cumulative explained variance.
    
    Saves the fitted PCA model.

    Parameters
    ----------
    df_in : pandas.DataFrame
        The input DataFrame containing numerical features to reduce.
    var_in : int or float
        if int : the number of principal components to keep.
        if float(0~1) : the amount of variance to preserve.
    o_path : str
        Output path where PCA model saved.
    n_in : str
        The name of the PCA model file.

    Returns
    -------
    dim_red : pandas.DataFrame
        A DataFrame after PCA-transformed.
        
    Prints
    -------
    exp_var : float
        The cumulative explained variance by the retained components.
    """
    pca = PCA(n_components=var_in)
    dim_red = pd.DataFrame(pca.fit_transform(df_in))
    write_pickle(pca, o_path, n_in)
    exp_var = sum(pca.explained_variance_ratio_)
    print (exp_var)
    return dim_red

def model_fun(df_in, lab_in, t_s, n_in, o_in, p_in, c_in):
    """
    Trains and evaluates a classification model with hyperparameter tuning.
    
    The function splits the data into training and testing sets.
    
    Performs grid search for hyperparameter optimization.
    
    Using Random Forest(rf), Gaussian Naive Bayes(GNB), or Support Vector Machine(SVM).
    
    Retrain the best model, evaluates performance, and saves the model and feature importance.

    Parameters
    ----------
    df_in : pandas.DataFrame
        The input DataFrame.
    lab_in : array-like
        The label vector or series for classification.
    t_s : float
        The proportion of test set in dataset.
    n_in : str
        The classifier to use
        - 'rf' : Random Forest.
        - 'gnb' : Gaussian Naive Bayes.
        - 'svm' : Support Vector Machine.
    o_in : str
        The saved model and feature importance path.
    p_in : dict
        Dictionary of hyperparameters to search over in GridSearchCV.
    c_in : int
        Number of cross-validation folds to use in GridSearchCV.

    Returns
    -------
    m : sklearn classifier object
        The trained classifier with optimal hyperparameters.
    m_metrics : pandas.DataFrame
        A DataFrame containing the precision, recall, and F1-score(weighted average).

    Prints
    -------
    - Best cross-validation performance score
    - Best hyperparameters found
    
    Notes
    -------
    Feature importance is only supported for models with 'feature_importances' attribute(Random Forest).
    For GNB, and SVM, feature importance will be skipped with a notice.
    """
    X_train, X_test, y_train, y_test = train_test_split(df_in, lab_in, test_size=t_s, random_state=42)
    
    if n_in == "rf":
        m = RandomForestClassifier(random_state=123)
    elif n_in == "gnb":
        m = GaussianNB()
    elif n_in == "svm":
        m = SVC()
    
    clf = GridSearchCV(m, p_in, cv=c_in)
    clf.fit(X_train, y_train)
    best_perf = clf.best_score_
    print ("Best Perf", best_perf)
    opt_params = clf.best_params_
    print ("Best Params", opt_params)
    
    if n_in == "rf":
        m = RandomForestClassifier(**opt_params, random_state=123)
    elif n_in == "gnb":
        m = GaussianNB(**opt_params) 
    elif n_in == "svm":
        m = SVC(**opt_params)
           
    m.fit(X_train, y_train)
    
    feat_imp = None
    try:
        feat_imp = pd.DataFrame(m.feature_importances_)
        feat_imp.index = df_in.columns
        feat_imp.columns = ["score"]
        perc_feat = round(len(feat_imp[feat_imp["score"] > 0.0]) / len(feat_imp)*100, 2)
        feat_imp.to_csv(o_in + n_in + "_"  + "fi.csv")
    except:
        print (n_in, "Not supported for feature importance")
        pass
    
    y_pred = m.predict(X_test)
    
    m_metrics = pd.DataFrame(precision_recall_fscore_support(y_test, y_pred, average='weighted'))
    m_metrics.index = ["precision", "recall", "fscore", None]

    write_pickle(m, o_in, n_in)
    return m, m_metrics
