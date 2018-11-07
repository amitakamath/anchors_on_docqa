
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
from __future__ import print_function
import os
import os.path
import numpy as np
import sklearn
import sklearn.model_selection
#import sklearn.linear_model
import sklearn.ensemble
import spacy
import sys
from sklearn.feature_extraction.text import CountVectorizer
from anchor import anchor_text
import json
import pdb
from docqa.scripts.black_box import load_model, query

# In[2]:
#pdb.set_trace()

# dataset from http://www.cs.cornell.edu/people/pabo/movie-review-data/
# Change this to SQuAD
#def load_polarity(path='/home/marcotcr/phd/datasets/sentiment-sentences'):
"""
def load_polarity(path='/sailhome/kamatha/anchor/rt-polaritydata'):
    data = []
    labels = []
    f_names = ['rt-polarity.neg', 'rt-polarity.pos']
    for (l, f) in enumerate(f_names):
        for line in open(os.path.join(path, f), 'rb'):
            try:
                line.decode('utf8')
            except:
                continue
            data.append(line.strip())
            labels.append(l)
    return data, labels
"""

def load_squad(path='/sailhome/kamatha/data/squad'):
    # Argument is the path to the train and dev JSON files of
    # SQuAD. Returns train_data and dev_data, dictionaries.
    
    context = []  # data
    question = []  # data 
    correct_answer = []  # label
    pdb.set_trace()

    with open(os.path.join(path,'train-v1.1.json'), 'r') as f:
        for line in f:
            train_data = json.loads(line)['data']
    with open(os.path.join(path,'dev-v1.1.json'), 'r') as f:
        for line in f:
            dev_data = json.loads(line)['data']
    
    return train_data, dev_data

# Note: you must have spacy installed. Run:
# 
#         pip install spacy && python -m spacy download en_core_web_lg

# In[3]:


nlp = spacy.load('en_core_web_lg')


# In[4]:

"""
data, labels = load_polarity()
train, test, train_labels, test_labels = sklearn.model_selection.train_test_split(data, labels, test_size=.2, random_state=42)
train, val, train_labels, val_labels = sklearn.model_selection.train_test_split(train, train_labels, test_size=.1, random_state=42)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
val_labels = np.array(val_labels)
"""

train_data, dev_data = load_squad()


# In[5]:

"""
vectorizer = CountVectorizer(min_df=1)
vectorizer.fit(train)
train_vectors = vectorizer.transform(train)
test_vectors = vectorizer.transform(test)
val_vectors = vectorizer.transform(val)
"""

# In[6]:

"""
#c = sklearn.linear_model.LogisticRegression()
c = sklearn.ensemble.RandomForestClassifier(n_estimators=500, n_jobs=10)
c.fit(train_vectors, train_labels)
preds = c.predict(val_vectors)
print('Val accuracy', sklearn.metrics.accuracy_score(val_labels, preds))
def predict_lr(texts):
    return c.predict(vectorizer.transform(texts))
"""

# Loads the pre-trained DocumentQA model
model = load_model('/u/scr/kamatha/document-qa/model-1105-063301')

# Splits context_plus_question into context and question,
# queries the model and returns the result.
def predict_lr(context_plus_question):
    #pdb.set_trace()
    answers = []
    if type(context_plus_question) is list:
        print()
        print()
        print("LEN LIST = "+str(len(list)))
        print()
        print()
        for item in context_plus_question:
            context = item.split(';')[0]
            question = item.split(';')[1]
            answers.append(model, context, question)
        return answers
    context = context_plus_question.split(';')[0]
    question = context_plus_question.split(';')[1]
    return query(model, context, question)



# ### Explaining a prediction
# use_unk_distribution=True means we will perturb examples by replacing words with UNKS

# In[7]:

# Class names are never used
explainer = anchor_text.AnchorText(nlp, ['negative', 'positive'], use_unk_distribution=True)


# In[8]:
# First example taken:
context = dev_data[0]['paragraphs'][5]['context']
question = dev_data[0]['paragraphs'][5]['qas'][3]['question']
context_plus_question = context+';'+question
true_answer = dev_data[0]['paragraphs'][5]['qas'][3]['answers'][0]['text']


#np.random.seed(1)
#text = 'This is a good book .'
#pred = explainer.class_names[predict_lr([text])[0]]

# Find the answer predicted by the model
pred_answer = predict_lr(context_plus_question)

#alternative =  explainer.class_names[1 - predict_lr([text])[0]]

print('Prediction: %s' % pred_answer)
#pdb.set_trace()

text = context+';'+question
exp = explainer.explain_instance(text, predict_lr, threshold=0.95, use_proba=True)


# Let's take a look at the anchor. Note that using this perturbation distribution, having the word 'good' in the text virtually guarantees a positive prediction

# In[9]:

pdb.set_trace()
print('Anchor: %s' % (' AND '.join(exp.names())))
print('Precision: %.2f' % exp.precision())
print()
#print('Examples where anchor applies and model predicts:')
#pdb.set_trace()
#print()
#print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
print()
#print('Examples where anchor applies and model predicts %s:' % alternative)
print()
#print('\n'.join([x[0] for x in exp.examples(partial_index=0, only_different_prediction=True)]))

"""
# ### Changing the distribution
# Let's try this with another perturbation distribution, namely one that replaces words by similar words instead of UNKS

# In[10]:


explainer = anchor_text.AnchorText(nlp, ['negative', 'positive'], use_unk_distribution=False)


# In[11]:


np.random.seed(1)
text = 'This is a good book .'
pred = explainer.class_names[predict_lr([text])[0]]
alternative =  explainer.class_names[1 - predict_lr([text])[0]]
print('Prediction: %s' % pred)
#import pdb
#pdb.set_trace()
exp = explainer.explain_instance(text, predict_lr, threshold=0.95, use_proba=True)


# Let's take a look at the anchor now. Note that with this distribution, we need more to guarantee a prediction of positive.

# In[12]:


print('Anchor: %s' % (' AND '.join(exp.names())))
print('Precision: %.2f' % exp.precision())
print()
print('Examples where anchor applies and model predicts %s:' % pred)
print()
print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
print()
print('Examples where anchor applies and model predicts %s:' % alternative)
print()
print('\n'.join([x[0] for x in exp.examples(only_different_prediction=True)]))


# Let's take a look at the partial anchor 'good' to see why it's not sufficient in this case
# 

# In[13]:


print('Partial anchor: %s' % (' AND '.join(exp.names(0))))
print('Precision: %.2f' % exp.precision(0))
print()
print('Examples where anchor applies and model predicts %s:' % pred)
print()
print('\n'.join([x[0] for x in exp.examples(partial_index=0, only_same_prediction=True)]))
print()
print('Examples where anchor applies and model predicts %s:' % alternative)
print()
print('\n'.join([x[0] for x in exp.examples(partial_index=0, only_different_prediction=True)]))


# ## See a visualization of the anchor with examples and etc (won't work if you're seeing this on github)

# In[ ]:


#exp.show_in_notebook()
"""
