# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 03:16:15 2025

@author: swimfighting
"""

from Package import *
import pandas as pd
import matplotlib.pyplot as plt

the_path = "C:/Users/25789/Documents/NLP/project_IMDB/aclImdb/train/"
out_path = "C:/Users/25789/Documents/NLP/project_IMDB/"

# read the data
the_data = file_crawler(the_path)

# clean the data
the_data["body"] = the_data["body"].apply(clean_text)

# remove stopwords
the_data["body_sw"] = the_data["body"].apply(rem_sw)

# stemming
the_data["body_sw_stem"] = the_data["body_sw"].apply(lambda x: (stem_fun(x, "stem")))

# lemmatization
the_data["body_sw_lemma"] = the_data["body_sw"].apply(lambda x: (stem_fun(x, "lemma")))

# save
write_pickle(the_data, out_path, "the_data") 

# read the save
the_data = read_pickle(out_path, "the_data")

# vectorize(TF)
x_data_tf_stem = transform_fun(the_data, "body_sw_stem", 1, 1, out_path, "tf")
x_data_tf_lemma = transform_fun(the_data, "body_sw_lemma", 1, 1, out_path, "tf")

# Define hyperparameters
params_rf = {"n_estimators": [10, 100], "max_depth": [None, 50]}
params_gnb = {"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]}
params_svm = {"gamma": ['scale', 'auto']}

# datatype -> Numpy
x_data_tf_stem = x_data_tf_stem.to_numpy()
x_data_tf_lemma = x_data_tf_lemma.to_numpy()

# PCA
x_data_tf_stem = pca_fun(x_data_tf_stem, 0.95, out_path, "pca")
x_data_tf_lemma = pca_fun(x_data_tf_lemma, 0.95, out_path, "pca")

# model training
m_mod_stem_rf, model_perf_stem_rf = model_fun(x_data_tf_stem, the_data.label, 0.2, "rf", out_path, params_rf, 5)
m_mod_stem_gnb, model_perf_stem_gnb = model_fun(x_data_tf_stem, the_data.label, 0.2, "gnb", out_path, params_gnb, 5)
m_mod_stem_svm, model_perf_stem_svm = model_fun(x_data_tf_stem, the_data.label, 0.2, "svm", out_path, params_svm, 5)

m_mod_lemma_rf, model_perf_lemma_rf = model_fun(x_data_tf_lemma, the_data.label, 0.2, "rf", out_path, params_rf, 5)
m_mod_lemma_gnb, model_perf_lemma_gnb = model_fun(x_data_tf_lemma, the_data.label, 0.2, "gnb", out_path, params_gnb, 5)
m_mod_lemma_svm, model_perf_lemma_svm = model_fun(x_data_tf_lemma, the_data.label, 0.2, "svm", out_path, params_svm, 5)

# word frequency
wrd_fun_sw = wrd_main(the_data, "body_sw")
wrd_fun_stem = wrd_main(the_data, "body_sw_stem")
wrd_fun_lemma = wrd_main(the_data, "body_sw_lemma")

# to dataframe(word-freq)
df_stem = pd.DataFrame(list(wrd_fun_stem.items()), columns=['word', 'frequency'])
df_lemma = pd.DataFrame(list(wrd_fun_lemma.items()), columns=['word', 'frequency'])

# get freq
frequency_dict_stem_neg = df_stem[df_stem['word'] == 'neg']['frequency'].iloc[0]
frequency_dict_stem_pos = df_stem[df_stem['word'] == 'pos']['frequency'].iloc[0]

frequency_dict_lemma_neg = df_lemma[df_lemma['word'] == 'neg']['frequency'].iloc[0]
frequency_dict_lemma_pos = df_lemma[df_lemma['word'] == 'pos']['frequency'].iloc[0]

# type -> dataframe
freq_df_stem_neg = pd.DataFrame(list(frequency_dict_stem_neg.items()), columns=['word', 'frequency'])
freq_df_stem_pos = pd.DataFrame(list(frequency_dict_stem_pos.items()), columns=['word', 'frequency'])

freq_df_lemma_neg = pd.DataFrame(list(frequency_dict_lemma_neg.items()), columns=['word', 'frequency'])
freq_df_lemma_pos = pd.DataFrame(list(frequency_dict_lemma_pos.items()), columns=['word', 'frequency'])

# sort the word(freq)
freq_df_stem_neg = freq_df_stem_neg.sort_values(by='frequency', ascending=False).reset_index(drop=True)
freq_df_stem_pos = freq_df_stem_pos.sort_values(by='frequency', ascending=False).reset_index(drop=True)

freq_df_lemma_neg = freq_df_lemma_neg.sort_values(by='frequency', ascending=False).reset_index(drop=True)
freq_df_lemma_pos = freq_df_lemma_pos.sort_values(by='frequency', ascending=False).reset_index(drop=True)

# draw the chart

# put dataframe together
comp_df_stem = pd.merge(freq_df_stem_neg, freq_df_stem_pos, on='word', how='outer').fillna(0)

# the word(2-11)
comp_df_stem = comp_df_stem.iloc[2:12]
comp_df_stem.plot(x='word', kind='bar', figsize=(12, 6), color=['skyblue', 'orange'])
plt.xlabel('Words') 
plt.ylabel('Frequency')
plt.legend(['Negative Frequency', 'Positive Frequency'], loc='upper right')
plt.title('Top ten words(stemming)')
plt.xticks(rotation=45)
plt.show()

comp_df_lemma = pd.merge(freq_df_lemma_neg, freq_df_lemma_pos, on='word', how='outer').fillna(0)

comp_df_lemma = comp_df_lemma.iloc[2:12]
comp_df_lemma.plot(x='word', kind='bar', figsize=(12, 6), color=['skyblue', 'orange'])
plt.xlabel('Words') 
plt.ylabel('Frequency')
plt.legend(['Negative Frequency', 'Positive Frequency'], loc='upper right')
plt.title('Top ten words(lemmatization)')
plt.xticks(rotation=45)
plt.show()