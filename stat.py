# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 22:38:38 2017

@author: Administrator
"""

import matplotlib_venn as mplv
import pandas as pd
from matplotlib import pyplot as plt
import nltk
from nltk.corpus import stopwords
import re
stops = list(set(stopwords.words('english')))+['','x']
train_path='dataset2/train.csv'
test_path='dataset2/test.csv'
pro_desc_path='dataset2/product_descriptions.csv'
attr_path='dataset2/attributes.csv'
df_train = pd.read_csv(train_path,encoding="ISO-8859-1")
df_test = pd.read_csv(test_path,encoding="ISO-8859-1")
df_pro_desc = pd.read_csv(pro_desc_path)
df_attr = pd.read_csv(attr_path)
df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
train_id=set(df_train.product_uid.values)
test_id=set(df_test.product_uid.values)
attr_id=set(df_attr.product_uid.values)

###############################################################################
#venn chart
plt.figure(1)
mplv.venn3_unweighted([train_id, test_id, attr_id], ('Train_id', 'Test_id', 'Attr_id'))
plt.figure(2)
mplv.venn2([train_id, test_id],('Train_id', 'Test_id'))

#Brand Frequency
plt.figure(3)
count_freq=nltk.FreqDist(df_brand.brand.values)
freq_brand=sorted(count_freq, key=count_freq.get, reverse=True)
x=[w for w in freq_brand[:5]]
y=[count_freq[w] for w in freq_brand[:5]]
plt.bar(range(len(x)),y,tick_label=x)
plt.xlabel("brand")
plt.ylabel("Frequency")

#relevance Frequency
plt.figure(4)
plt.hist(df_train.relevance.values,)
plt.title("relevance")
plt.xlabel("relevance")
plt.ylabel("Frequency")

#high frequency search_term
plt.figure(5)
df_total = pd.concat((df_train, df_test), axis=0, ignore_index=True)
all_query=' '.join(df_total['search_term'].values)
def get_words(text):  
    text=text.lower()
    word_split = re.compile('[^a-zA-Z-]')
    clean_words=[word.strip() for word in word_split.split(text) if word not in stops]
    return clean_words
count_freq=nltk.FreqDist(get_words(all_query))
freq_brand=sorted(count_freq, key=count_freq.get, reverse=True)
x=[w for w in freq_brand[:10]]
y=[count_freq[w] for w in freq_brand[:10]]
plt.bar(range(len(x)),y,tick_label=x,color='#624ea7')

