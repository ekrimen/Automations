#!/usr/bin/env python
# coding: utf-8

# In[33]:


from sklearn import model_selection,preprocessing,linear_model,naive_bayes,metrics,svm


# In[34]:


from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer


# In[35]:


from sklearn import decomposition,ensemble


# In[36]:


import pandas,xgboost,numpy,textblob,string


# In[37]:


from keras.preprocessing import text,sequence


# In[38]:


from keras import layers,models,optimizers


# In[39]:


# load the data
trainDF=pandas.read_csv('TaskType.csv',names=['labels','texts'],skiprows=1)


# In[ ]:





# In[40]:


# split data into training and test set
train_x,valid_x,train_y,valid_y=model_selection.train_test_split(trainDF['texts'],trainDF['labels'])


# In[ ]:





# In[41]:


# label encode the target variable 

encoder=preprocessing.LabelEncoder()
train_y=encoder.fit_transform(train_y)
valid_y=encoder.fit_transform(valid_y)


# In[ ]:





# In[43]:


#Create Count Vectorizer Object

count_vect = CountVectorizer(analyzer='word',token_pattern='\w{1,}')
count_vect.fit(trainDF['texts'])


# In[44]:


# Transform the train and validation text into vectorizer object

xtrain_vect = count_vect.transform(train_x)
xvalid_vect = count_vect.transform(valid_x)


# In[ ]:





# In[ ]:





# In[56]:


# word level tf-idf

tfidf_vect = TfidfVectorizer(analyzer='word',token_pattern='\w{1,}',max_features=5000)
tfidf_vect.fit(trainDF['texts'])
xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)


# In[57]:


# ngram level tf-idf

tfidf_vect_ngram = TfidfVectorizer(analyzer='word',token_pattern='\w{1,}',ngram_range=(2,3),max_features=5000)
tfidf_vect_ngram.fit(trainDF['texts'])
xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram = tfidf_vect_ngram.transform(valid_x)


# In[58]:


# charecter level tf-idf

tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char',token_pattern='\w{1,}',ngram_range=(2,3),max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['texts'])
xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(train_x)
xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(valid_x)


# In[59]:


# load the pre-trained word-embedding vectors 

embeddings_index={}
for i,line in enumerate(open('wiki-news-300d-1M.vec',encoding='utf8')):
    values = line.split()
    embeddings_index[values[0]] = numpy.asarray(values[1:],dtype='float32')
    


# In[60]:


# create a tokenizer 
token = text.Tokenizer()
token.fit_on_texts(trainDF['texts'])
word_index = token.word_index


# In[61]:


# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)


# In[62]:


# create token-embedding mapping
embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[63]:


#Create Text based features

trainDF['char_count'] = trainDF['texts'].apply(len)
trainDF['word_count'] = trainDF['texts'].apply(lambda x: len(x.split()))
trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count']+1)
trainDF['punctuation_count'] = trainDF['texts'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
trainDF['title_word_count'] = trainDF['texts'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
trainDF['upper_case_word_count'] = trainDF['texts'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))


# In[64]:


pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}


# In[65]:


# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt


# In[66]:


#nltk.download()


# In[67]:


trainDF['noun_count'] = trainDF['texts'].apply(lambda x: check_pos_tag(x, 'noun'))
trainDF['verb_count'] = trainDF['texts'].apply(lambda x: check_pos_tag(x, 'verb'))
trainDF['adj_count'] = trainDF['texts'].apply(lambda x: check_pos_tag(x, 'adj'))
trainDF['adv_count'] = trainDF['texts'].apply(lambda x: check_pos_tag(x, 'adv'))
trainDF['pron_count'] = trainDF['texts'].apply(lambda x: check_pos_tag(x, 'pron'))


# In[ ]:





# In[71]:


# train a LDA Model
lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)
X_topics = lda_model.fit_transform(xtrain_vect)
topic_word = lda_model.components_ 
vocab = count_vect.get_feature_names()


# In[ ]:





# In[76]:


# view the topic models
n_top_words = 10
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))


# In[78]:


topic_summaries


# In[152]:


def train_model(classifier, feature_vector_train, label, feature_vector_valid,valid_y, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    valid_y_r=encoder.inverse_transform(valid_y)
    #xtrain_r=count_vect.inverse_transform(feature_vector_train)
    valid_x_r=numpy.asarray(valid_x)
    #print(xtrain_r)
    #print(valid_y_r)
    
    predictions_r=encoder.inverse_transform(predictions)
    print(numpy.column_stack((valid_x_r,predictions_r,valid_y_r)))
    #numpy.savetxt("prediction.csv",zip((valid_x_r,predictions_r,valid_y_r)),delimiter=",",fmt='%s')
    numpy.savetxt("valid_x_r.csv",valid_x_r,delimiter=",",fmt='%s')
    numpy.savetxt("predictions_r.csv",predictions_r,delimiter=",",fmt='%s')
    numpy.savetxt("valid_y_r.csv",valid_y_r,delimiter=",",fmt='%s')
    numpy.savetxt("all.csv",zip((valid_x_r,predictions_r,valid_y_r)),delimiter=",",fmt='%s')
    return metrics.accuracy_score(predictions, valid_y)


# In[153]:


# Naive Bayes on Count Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_vect, train_y, xvalid_vect,valid_y)
print ("NB, Count Vectors: ", accuracy)

# Naive Bayes on Word Level TF IDF Vectors
#accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
#print ("NB, WordLevel TF-IDF: ", accuracy)

# Naive Bayes on Ngram Level TF IDF Vectors
#accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
#print ("NB, N-Gram Vectors: ", accuracy)

# Naive Bayes on Character Level TF IDF Vectors
#accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
#print ("NB, CharLevel Vectors: ", accuracy)


# In[103]:


new=pandas.read_csv('NewTask.csv',names=['texts'])


# In[104]:


new['texts']


# In[ ]:




