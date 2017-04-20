# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 00:29:15 2017

@author: Administrator
"""

import numpy as np
import pandas as pd
import time
import re
import nltk
import random
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.svm import SVC
from nltk.stem.porter import *
from sklearn import pipeline
from collections import Counter
import csv
import ngram_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
start_time = time.time()
stemmer = PorterStemmer()

###############################################################################
#corrector part
from sklearn.linear_model import BayesianRidge, LinearRegression

def words(text): return re.findall('[a-z]+', text.lower()) 

def correct_train(features):
    model = defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model
'''
file= open("corrector.txt")
f=file.read()
file.close()
NWORDS = correct_train(words(f))
del f
alphabet = 'abcdefghijklmnopqrstuvwxyz'
'''
def edits1(word):
   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes    = [a + b[1:] for a, b in splits if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
   replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
   inserts    = [a + c + b     for a, b in splits for c in alphabet]
   return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words): return set(w for w in words if w in NWORDS)

def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)

###############################################################################
#clean document
def clean_words(words): 
    if isinstance(words, str):
        words = re.sub(r"(\w)([A-Z]+)",r"\1 \2",words)
        words = re.sub(r"(\w)\.([A-Z]+)", r"\1 \2", words) 
        words = words.lower()
        words = re.sub(r"(\d+)( *)(inches|inch|in|')\.?", r"\1in. ", words)
        words = re.sub(r"(\d+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", words)
        words = re.sub(r"(\d+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", words)
        words = re.sub(r"(\d+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", words)
        words = re.sub(r"(\d+)( *)(centimeters|cm)\.?", r"\1cm. ", words)
        words = re.sub(r"(\d+)( *)(milimeters|mm)\.?", r"\1mm. ", words)
        words = re.sub(r"(\d+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", words)
        words = re.sub(r"(\d+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", words)
        words = re.sub(r"(\d+)( *)(degrees|degree)\.?", r"\1deg. ", words)
        words = re.sub(r"(\d+)( *)(volts|volt)\.?", r"\1volt. ", words)
        words = re.sub(r"(\d+)( *)(watts|watt)\.?", r"\1watt. ", words)
        words = re.sub(r"(\d+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", words)
        words = re.sub(r"( +x +|\*| +by +)",r" xby ",words)
        words = re.sub(r"x(\d+)",r" xby \1",words)
        words = re.sub(r"(\d+)x",r"\1 xby ",words)
        words = re.sub(" +"," ",words)
        words = " ".join([stemmer.stem(z) for z in words.split()])
        return words
    else:
        return "null"

###############################################################################
#batch correct function
def words_corrector(words):
    words=str(words)
    words=words.split()
    new_words=list()
    for word in words:
        if word.isalpha() :
            new_words.append(correct(word))
        else :
            new_words.append(word)
    new_words=' '.join(new_words)
    return new_words
    
###############################################################################
#get bullet attribute
def get_Bullet(df_attr,df_pro_desc):
    product_id=df_pro_desc.product_uid.values
    alist=list()
    for i in list(range(1,30)):
        if i<10:
            alist.append('Bullet0'+str(i))
    df_Bullet=df_attr[df_attr.name.isin(alist)][["product_uid", "value"]].rename(columns={"value": "Bullet"})
    Bullets=list()
    Bullets_id=list()
    for i in list(range(0,len(product_id))):
        if product_id[i] in df_Bullet.product_uid.values:
            Bullets_id.append(product_id[i])
            temp_Bullet= df_Bullet.Bullet[df_Bullet.product_uid==product_id[i]].values
            temp_Bullet=[str(x) for x in temp_Bullet]
            temp_Bullet=' '.join(temp_Bullet).lower()
            Bullets.append(temp_Bullet)
        if i%1000==0:
            print(i)
    df_Bullet=pd.DataFrame()
    df_Bullet['product_uid']=Bullets_id
    df_Bullet['Bullets']=Bullets
    return df_Bullet

#get color attribute
def get_color(df_attr):
    colors = ["product_color", "Color Family", "Color/Finish", "Color/Finish Family"]
    df_color = df_attr[df_attr.name.isin(colors)][["product_uid", "value"]].rename(columns={"value": "color"})
    df_color.dropna(how="all", inplace=True)
    all_color = lambda df: " ".join(str(v) for v in list(set(df['color'])))
    df_color = df_color.groupby("product_uid").apply(all_color)
    df_color = df_color.reset_index(name="color")
    df_color["color"] = df_color["color"].values.astype(str)
    return df_color

###############################################################################
#queryExpansion(get an alter query)
class QueryExpansion:
    def __init__(self, df, ngram=3, stopwords_threshold=0.9, base_stopwords=set()):
        self.df = df[["search_term", "product_title"]].copy()
        self.ngram = ngram
        self.stopwords_threshold = stopwords_threshold
        self.stopwords = set(base_stopwords).union(self._get_customized_stopwords())
        
    def _get_customized_stopwords(self):
        words = " ".join(list(self.df["product_title"].values)).split(" ")
        counter = Counter(words)
        num_uniq = len(list(counter.keys()))
        num_stop = int((1.-self.stopwords_threshold)*num_uniq)
        stopwords = set()
        for e,(w,c) in enumerate(sorted(counter.items(), key=lambda x: x[1])):
            if e == num_stop:
                break
            stopwords.add(w)
        return stopwords
    
    def ngrams(self,inputs, n):
        length=len(inputs)
        output = []
        for i in range(length-n+1):
            output.append(tuple(inputs[i:i+n]))
        return output
        
    def _ngram(self, text):
        tokens = text.split(" ")
        tokens = [token for token in tokens if token not in self.stopwords]
        return  ngram_model._ngrams(tokens, self.ngram, " ")

    def _get_alternative_query(self, df):
        res = []
        for v in df:
            res += v
        c = Counter(res)
        value, count = c.most_common()[0]
        return value

    def build(self):
        self.df["title_ngram"] = self.df["product_title"].apply(self._ngram)
        corpus = self.df.groupby("search_term").apply(lambda df: self._get_alternative_query(df["title_ngram"]))
        corpus = corpus.reset_index()
        corpus.columns = ["search_term", "search_term_alt"]
        self.df = pd.merge(self.df, corpus, on="search_term", how="left")
        return self.df["search_term_alt"].values

###############################################################################
#counter
def Words_counter1(string1,string2):
    count=0
    string1=string1.lower().split()
    for w in string1:
        if w.isdigit() or len(w)>=2:
            if len(re.findall(w,string2))>0:
                count=count+1
    return count

def Words_counter2(string1,string2):
    count=0
    string1=string1.lower().split()
    string2=string2.lower()
    for w in string1:
        if w.isdigit() or len(w)>=2:
            w_in_string2=len(re.findall(w,string2))
            count=w_in_string2+count
    return count

    
def word_in_stuff(df_total,query,stuff):
    df=df_total.loc[:,[query,stuff]].values
    counts=[Words_counter1(str1,str2) for str1,str2 in df]
    return counts
    
def query_in_stuff(df_total,query,stuff):
    df=df_total.loc[:,[query,stuff]].values
    counts=[Words_counter2(str1,str2) for str1,str2 in df]
    return counts
    
###############################################################################
#load data
'''
train_path='E:/kaggle2/dataset/train.csv'
test_path='E:/kaggle2/dataset/test.csv'
pro_desc_path='E:/kaggle2/dataset/product_descriptions.csv'
attr_path='E:/kaggle2/dataset/attributes.csv'
df_train = pd.read_csv(train_path,encoding="ISO-8859-1")
df_test = pd.read_csv(test_path,encoding="ISO-8859-1")
df_pro_desc = pd.read_csv(pro_desc_path)
'''
df_attr = pd.read_csv(attr_path)
'''
df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
df_Bullet=get_Bullet(df_attr,df_pro_desc)
'''
df_color=get_color(df_attr)
'''
n=len(df_train)
#merge dataframes
df_total = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_total = pd.merge(df_total, df_pro_desc, how='left', on='product_uid')
df_total = pd.merge(df_total, df_brand, how='left', on='product_uid')
df_total = pd.merge(df_total, df_Bullet, how='left', on='product_uid')
'''
df_total = pd.merge(df_total, df_color, how='left', on='product_uid')
'''
del df_train
del df_test
del  df_pro_desc
del  df_attr
del df_Bullet
#features
df_total['search_term'] = df_total['search_term'].map(lambda x:clean_words(words_corrector(x)))
del NWORDS
print(2)
df_total['product_title'] = df_total['product_title'].map(lambda x:clean_words(x))
df_total['product_description'] = df_total['product_description'].map(lambda x:clean_words(x))
df_total['brand'] = df_total['brand'].map(lambda x:clean_words(x))
df_total['Bullets']=df_total['Bullets'].map(lambda x:clean_words(x))
'''
alter_query=QueryExpansion(df_total)
df_total['alter_query']=alter_query.build()
'''
df_total['len_of_search_term']=df_total['search_term'].map(lambda x:len(x)).astype(np.int64)
df_total['len_of_query'] = df_total['search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_total['len_of_title'] = df_total['product_title'].map(lambda x:len(x.split())).astype(np.int64)
df_total['len_of_description'] = df_total['product_description'].map(lambda x:len(x.split())).astype(np.int64)
df_total['len_of_brand'] = df_total['brand'].map(lambda x:len(x.split())).astype(np.int64)
df_total['len_of_Bullet']=df_total['Bullets'].map(lambda x:len(x.split())).astype(np.int64)
#df_total['len_of_query_title']=df_total['len_of_title']/df_total['len_of_query']
#df_total['len_of_query_desc']=df_total['len_of_description']/df_total['len_of_query']
df_total['query_in_title'] = query_in_stuff(df_total,'search_term','product_title')
df_total['query_in_description'] = query_in_stuff(df_total,'search_term','product_description')
df_total['query_in_Bullet'] = query_in_stuff(df_total,'search_term','Bullets')
df_total['word_in_title'] =  word_in_stuff(df_total,'search_term','product_title')
df_total['word_in_description'] =  word_in_stuff(df_total,'search_term','product_description')
df_total['word_in_brand'] =  word_in_stuff(df_total,'search_term','brand')
df_total['word_in_Bullet'] = word_in_stuff(df_total,'search_term','Bullets')
df_total['word_in_color']=word_in_stuff(df_total,'search_term','color')
#df_total['alter_in_title']=word_in_stuff(df_total,'alter_query','title')
#df_total['alter_in_title']=word_in_stuff(df_total,'alter_query','product_description')
df_total['ratio_title'] = df_total['word_in_title']/df_total['len_of_query']
df_total['ratio_description'] = df_total['word_in_description']/df_total['len_of_query']
df_total['ratio_brand'] = df_total['word_in_brand']/df_total['len_of_query']
#df_total['ratio_Bullet']= df_total['word_in_Bullet']/df_total['len_of_query']
#df_total['rate_title'] = df_total['query_in_title']/df_total['len_of_query']
#df_total['rate_description'] = df_total['query_in_description']/df_total['len_of_query']
#df_total['rate_Bullet']= df_total['query_in_Bullet']/df_total['len_of_query']
df_brand = pd.unique(df_total.brand.ravel())
d={}
i = 1
for s in df_brand:
    d[s]=i
    i+=1
df_total['brand_feature'] = df_total['brand'].map(lambda x:d[x])
'''
class Drop_useless_features(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['id','relevance','search_term','product_title','product_description','brand','Bullets','color','alter_query']
        hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        return hd_searches

class select_feature(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)
        
print(3)
df_train = df_total.iloc[:n]
df_test = df_total.iloc[n:]
del df_total
train_y = df_train['relevance'].values
train_x =df_train[:]
test_x = df_test[:]
'''
#Model of RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 820, n_jobs = -1, max_features=9, max_depth=20, verbose = 1)
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
tsvd = TruncatedSVD(n_components=10, random_state = 1301)
clf = pipeline.Pipeline([
        ('union', FeatureUnion(
                    transformer_list = [
                       clf = pipeline.Pipeline([
        ('union', FeatureUnion(
                    transformer_list = [
                        ('cst',  Drop_useless_features()),  
                        ('t1', pipeline.Pipeline([('s1', select_feature(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                        ('t2', pipeline.Pipeline([('s2', select_feature(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                        ('t3', pipeline.Pipeline([('s3', select_feature(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                        ('t4', pipeline.Pipeline([('s4', select_feature(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)]))
                        ],
                    transformer_weights = {
                        'cst': 0.9,
                        't1': 0.5,
                        't2': 0.25,
                        't3': 0.0,
                        't4': 0.5
                        },
                n_jobs = 1
                )), 
                    transformer_weights = {
                        'cst': 0.9,
                        't1': 0.5,
                        't2': 0.25,
                        't3': 0.0,
                        't4': 0.5
                        },
                n_jobs = 1
                )), 
        ('rfr', rfr)])
clf.fit(train_x, train_y)
y_pred = clf.predict(test_x)
'''
#Model of GradientBoostingRegressor
GB_params1 = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 2, 'min_samples_leaf':15,
              'learning_rate': 0.035, 'loss': 'ls', 'verbose':0, 'random_state':2016}
gbr = GradientBoostingRegressor(**GB_params1)
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
tsvd = TruncatedSVD(n_components=10)
clf = pipeline.Pipeline([
        ('union', FeatureUnion(
                    transformer_list = [
                        ('cst',  Drop_useless_features()),  
                        ('t1', pipeline.Pipeline([('s1', select_feature(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                        ('t2', pipeline.Pipeline([('s2', select_feature(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                        ('t3', pipeline.Pipeline([('s3', select_feature(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                        ('t4', pipeline.Pipeline([('s4', select_feature(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)]))
                        ],
                    transformer_weights = {
                        'cst': 0.9,
                        't1': 0.5,
                        't2': 0.25,
                        't3': 0.0,
                        't4': 0.5
                        },
                n_jobs = -1
                )), 
        ('gbr', gbr)])
clf.fit(train_x, train_y)
y_pred = clf.predict(test_x)

'''
#Model of XGBRegressor
xgb_params={'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 1000, 'reg_lambda': 1, 'colsample_bytree':0.65}
xgb = XGBRegressor(**xgb_params)
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
tsvd = TruncatedSVD(n_components=10)
clf = pipeline.Pipeline([
        ('union', FeatureUnion(
                    transformer_list = [
                        ('cst',  Drop_useless_features()),  
                        ('t1', pipeline.Pipeline([('s1', select_feature(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                        ('t2', pipeline.Pipeline([('s2', select_feature(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                        ('t3', pipeline.Pipeline([('s3', select_feature(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                        ('t4', pipeline.Pipeline([('s4', select_feature(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)]))
                        ],
                    transformer_weights = {
                        'cst': 0.9,
                        't1': 0.5,
                        't2': 0.25,
                        't3': 0.0,
                        't4': 0.5
                        },
                n_jobs = -1
                )),
        ('xgb', xgb)])
clf.fit(train_x, train_y)
y_pred = clf.predict(test_x)


#Model of ExtraTreesRegressor
xtree_params={'n_estimators': 250,  'max_depth': None, 'min_samples_split':12, \
              'verbose': 1, 'random_state':2016, 'n_jobs':-1}
etg = ExtraTreesRegressor(**xtree_params)
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
tsvd = TruncatedSVD(n_components=10)
clf = pipeline.Pipeline([
        ('union', FeatureUnion(
                    transformer_list = [
                        ('cst',  Drop_useless_features()),  
                        ('t1', pipeline.Pipeline([('s1', select_feature(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                        ('t2', pipeline.Pipeline([('s2', select_feature(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                        ('t3', pipeline.Pipeline([('s3', select_feature(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                        ('t4', pipeline.Pipeline([('s4', select_feature(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)]))
                        ],
                    transformer_weights = {
                        'cst': 0.9,
                        't1': 0.5,
                        't2': 0.25,
                        't3': 0.0,
                        't4': 0.5
                        },
                n_jobs = -1
                )), 
        ('etg', etg)])
clf.fit(train_x, train_y)
y_pred = clf.predict(test_x)
'''
#model evaluation
df_sol=pd.read_csv('E:/kaggle2/dataset/solution.csv')
y_true=df_sol.relevance.values
def fmean_squared_error(y_true, predictions):
    sums=0
    j=0
    for i in list(range(0,len(y_true))):
        if y_true[i]!=-1:
            sums=sums+(y_true[i]-predictions[i])**2
            j=j+1
    mean_sq=sums/j
    return mean_sq**0.5
s=fmean_squared_error(y_true,y_pred)
print(s)  
'''
y_pred_all = np.column_stack((y_pred,y_pred_1,y_pred_2,y_pred_3))
l2_train = np.column_stack((y_pred_t,y_pred_t1,y_pred_t2,y_pred_t3))


from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=4)

param_grid = {
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 300, 600, 1000],
    'max_depth': [2, 3, 10],
    #'reg_alpha': [0, 0.01, 0.5, 0.9],
    'reg_lambda': [0.01, 1, 10, 1000],
    'colsample_bylevel': [0.5, 0.75, 1]
}

clf = GridSearchCV(XGBRegressor(nthread=4), param_grid=param_grid, cv=4)
clf.fit(y_pred_all, y_train)
'''
#create csv
pd_id=pd.read_csv('E:/kaggle2/dataset/sample_submission.csv')
ind=pd_id['id'].values
del pd_id
out_file = open("E:/kaggle2/dataset/sol.csv", "w", newline='') 
writer = csv.writer(out_file)
writer.writerow(['id','relevance'])
for i in range(len(y_pred)):
    writer.writerow([ind[i],y_pred[i]])
out_file.close()    
