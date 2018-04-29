# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:04:46 2017

@author: ArunKumar
"""
#Reading the collected tweets
tweets_data_path = 'C:\Users\ArunKumar\Desktop\DataScience\Project1\TotalTweets.txt'
tweets_file = open(tweets_data_path, "r")
import json
# Read each tweet and add it to the list
tweets=[]
for eachLine in tweets_file:
    if eachLine=='\n':
        continue
    tweets.append(json.loads(eachLine))
tweets_file.close()

#Collect only text from this list
tweetTexts=[]
for t in tweets:
    try:
        tweetTexts.append(t['text'])
    except:
        pass
    
#PreProcess tweets appropriate for text mining
import re
dataToBeFed=[]
for tweet in tweetTexts:
    processed=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(http\S+)"," ",str(tweet.encode('utf-8'))).split())
    dataToBeFed.append(processed)
import nltk
defaultStopwords = nltk.corpus.stopwords.words("english")
desiredStopWords = defaultStopwords+['http://','rt','@','RT','rt','gif','Trump','trump','Donald','Melania']
tweetList=[]


'''     Sentimental Analysis  '''
from textblob import TextBlob
pol_list =[]
subj_list = [] 
for eachTweet in dataToBeFed:   
    TweetText=eachTweet.split()
    cleaned_text = filter(lambda x: x not in desiredStopWords, TweetText)
    tweet = ' '.join(cleaned_text)
    tweetList.append(tweet)
    stringUnderTest = TextBlob(tweet)
    pol_list.append(stringUnderTest.sentiment.polarity)
    subj_list.append(stringUnderTest.sentiment.subjectivity)

#constructing histograms based on subjectivity and polarity
import matplotlib.pyplot as plt
aggregatePolarity = sum(i for i in pol_list)
pol_Average = aggregatePolarity/len(pol_list)
aggregateSubjectivity = sum(i for i in subj_list)
subjectivity_Average = aggregateSubjectivity/len(subj_list)   
plt.hist(subj_list, bins=20) #, normed=1, alpha=0.75)
plt.xlabel('subjectivity score')
plt.ylabel('Tweet count')
plt.grid(True)
plt.axvline(subjectivity_Average, color='b', linestyle='dashed', linewidth=2)
plt.savefig('subjectivityHistogram10kTweets.pdf')
plt.show()
plt.hist(pol_list, bins=20) #, normed=1, alpha=0.75)
plt.xlabel('Polarity score')
plt.ylabel('Tweet count')
plt.grid(True)
plt.axvline(pol_Average, color='b', linestyle='dashed', linewidth=2)
plt.savefig('polarityHistogram10kTweets.pdf')
plt.show()
# 
''' Word cloud on collected tweets'''
# # Appropriate tweet pre-processing for wordcloud
data=''.join(dataToBeFed).split()
WordCloudTextList=[]
for word in data:
     if word not in desiredStopWords:
         WordCloudTextList.append(word)
WordCloudText = " ".join(str(x) for x in WordCloudTextList)
 
#Wordcloud module import and wordcloud generation
from wordcloud import WordCloud
wordcloud = WordCloud(max_font_size=40).generate(WordCloudText)
## Display the generated image:
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
plt.savefig('wordCloud.pdf')
#==============================================================================


''' Topic Modelling '''
#==============================================================================
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# #Form document term matrix as below
vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
doc_term_matrix = vectorizer.fit_transform(tweetList)
vocab = vectorizer.get_feature_names()
 
num_topics = 10 #The number of topics that we want to mine the documents aka tweets
from sklearn import decomposition#
clf = decomposition.NMF(n_components=num_topics, random_state=1)
doctopic = clf.fit_transform(doc_term_matrix)
# 
topic_words = []
num_top_words = 5
# 
for topic in clf.components_:
     word_idx = np.argsort(topic)[::-1][:num_top_words]
     for idx in word_idx:
         print(vocab[idx])
     topic_words.append([vocab[i] for i in word_idx])
from sklearn import decomposition
dtm = vectorizer.fit_transform(tweetList).toarray()
for n in range(1, 10):    
    num_topics = 5*n
    num_top_words = 10
    clf = decomposition.NMF(n_components=num_topics, random_state=1)
    doctopic = clf.fit_transform(dtm)
    #print(num_topics, clf.reconstruction_err_)
for t in range(len(topic_words)):
    print("Topic {}: {}".format(t, ' '.join(topic_words[t][:15])))
#==============================================================================

##Gensim Topic modelling
from gensim import corpora, models
docTweets=[]
for i in range(len(tweetList)):
    docTweets.append(tweetList[i].split())
dic = corpora.Dictionary(docTweets)
corpus = [dic.doc2bow(text) for text in docTweets]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
NUM_TOPICS = 10
model = models.ldamodel.LdaModel(corpus_tfidf, 
                                 num_topics=NUM_TOPICS, 
                                 id2word=dic, 
                                 update_every=1, 
                                 passes=100)
print("LDA model")
topics_found = model.print_topics(20)
print(model.log_perplexity(corpus))
#==============================================================================
counter = 1
for t in topics_found:
    print("Topic #{} {}".format(counter, t))
    counter += 1
#==============================================================================
