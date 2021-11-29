# 427final
## Daniel Lindberg

### Overview
The problem faced is the problem that we all face, categorizing biomedical research articles pertaining to "acute rheumatic arthritis", "disease, lyme", "abnormalities, cardiovascular", and "knee osteoarthritis". I'm going to do this with bag of words though naive Bayes, SVM, and logistical prediction models to predict the class based on the abstract.

### Libraries used
- biopython
- nltk
- sklearn
- xml
- matplotlib

### The dataset
The dataset will come from pubmed, and as per the instructions, pull all the documents that were published after 2010 that have one of the four classifications list above. Thank fully I heard about BioPython so all of the articles were pulled using that making it much simpler, and then storing them in an xml file for later use.

### Method
So like said in the dataset portion, biopython was used to collect articles. After the data was collected though, I used my normalize method from homework 2 to normalize the articles. It takes the string, tokenizes it removing punctuation, then removes stop words, lower cases the words, port stems, then lemmatizes the words, finally returning a normailzed string. After the dataset was split into a training set of 80/20 which seemed to work very well for 3 pipelines.

### Validation and results
#### Naive Bayes
```
pipe  = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,2))),
    ('tfidf', TfidfTransformer( use_idf=True)),
    ('clf', MultinomialNB( alpha=0.1, fit_prior=True, class_prior=None)),
])
```
![plot](/confusion_matrix_Bayes.png)
```
Accuracy: 0.957491
Precision: 0.957
Recall: 0.957
F1: 0.957
```
#### SVM
```
pipe = Pipeline([
    ('vect', TfidfVectorizer(max_df=0.9,ngram_range=(1,2))),
    ('svm', SGDClassifier(loss='hinge', penalty='l1', alpha=1e-5, max_iter=100, random_state=42)),
])
```
![plot](/confusion_matrix_SVM.png)
```
Accuracy: 0.978147
Precision: 0.978
Recall: 0.978
F1: 0.978
```
#### Logistic Regression
```
pipe = Pipeline([   #Pipeline for logistic regression
    ('vect', TfidfVectorizer(max_df=0.8)),
    ('clf', LogisticRegression(penalty='l2', C=3, random_state=42,
                                 max_iter=1000)),
])
```
![plot](/confusion_matrix_LR.png)
```
Accuracy: 0.973208
Precision: 0.973
Recall: 0.973
F1: 0.973
```
### Conclusion
All three of the pipelines created worked exceptionally well. The lowest accuracy was the quickest performing, it was originally 93.4% but by using ngrams I was able to increase accuracy by 2.2% getting 95.7% accuracy with similar precision. The most accurate was the SVM prediction pipeline coming in at 97.8% accurate. Over all the results confirm that the goal set has been reached.

### Discussion
If I were to continue this project, I would give Word2vec a go. I didn’t get to it for this project because of other class projects and the amount of time it would take to run those. The BoW seemed accurate enough to be passable though, with the lowest accuracy 95.7%. I would also be interested in trying to categorize new unseen articles that weren’t in the test data and see how the prediction models work on them.
