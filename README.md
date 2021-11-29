# 427final
## Daniel Lindberg

### Overview
The problem faced is the problem that we all face, catorgorizing biomedical research articles pertaining to "acute rheumatic arthritis", "disease, lyme", "abnormalities, cardiovascular", and "knee osteoarthritis". I'm going to do this with bag of words though naive Bayes, SVM, and logistical prediction models to predict the class based on the abstract.

### Libraries used
- biopython
- nltk
- sklearn
- xml
- matplotlib

### The dataset
The dataset will come from pubmed, and as per the instructions, pull all the documents that were published after 2010 that have one of the four classifications as above. Thank fully I heard about BioPython so all of the articles were pulled using that and then storing them in an xml file for storage and later use.

### Method
So like said in the dataset portion, biopython was used to collect articles. After the data was collected though, I used my normalize method from homework 2 to normalize the articles. It takes the the string, tokenizes it removing punctuation, then removes stop words, lower cases the words, port stems, then lemmatizes the words, finally returning a string or normailzed words. After the dataset was split into a training set of 80/20 which seemed to work very well for 3 pipelines.

### Validation and results
#### The first pipeline Naive Bayes
```
pipe  = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,2))),
    ('tfidf', TfidfTransformer( use_idf=True)),
    ('clf', MultinomialNB( alpha=0.1, fit_prior=True, class_prior=None)),
])
```
```
Accuracy: 0.957491
Precision: 0.957
Recall: 0.957
F1: 0.957
```
