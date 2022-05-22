#  data from https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download
#  protocol from https://towardsdatascience.com/a-beginners-guide-to-text-classification-with-scikit-learn-632357e16f3a

# a project from https://towardsdatascience.com/5-solved-end-to-end-data-science-projects-in-python-acdc347f36d0

import pandas as pd
df_review = pd.read_csv('imdb_dataset.csv')
df_review.head()

df_positive = df_review[df_review['sentiment']=='positive'][:9000]
df_negative = df_review[df_review['sentiment']=='negative'][:1000]
df_review_imb = pd.concat([df_positive, df_negative])

from imblearn.under_sampling import  RandomUnderSampler

rus = RandomUnderSampler(random_state=0)
df_review_bal, df_review_bal['sentiment']=rus.fit_resample(df_review_imb[['review']],
                                                           df_review_imb['sentiment'])
df_review_bal

from sklearn.model_selection import train_test_split
train, test = train_test_split(df_review_bal, test_size=0.33, random_state=42)

train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)
train_x_vector

pd.DataFrame.sparse.from_spmatrix(train_x_vector,
                                  index=train_x.index,
                                  columns=tfidf.get_feature_names_out())

test_x_vector = tfidf.transform(test_x)

# support-vector machine learning model (SVM)
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(train_x_vector, train_y)

print(svc.predict(tfidf.transform(['A good movie'])))
print(svc.predict(tfidf.transform(['An excellent movie'])))
print(svc.predict(tfidf.transform(['I did not like this movie at all'])))

# fit a decision tree
from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_x_vector, train_y)

print(dec_tree.predict(tfidf.transform(['A good movie'])))
print(dec_tree.predict(tfidf.transform(['An excellent movie'])))
print(dec_tree.predict(tfidf.transform(['I did not like this movie at all'])))

# fit a naive Bayes model
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(train_x_vector.toarray(), train_y)

print(gnb.predict(tfidf.transform(['A good movie']).toarray()))  # this must be transformed from sparse matrix to dense data via x.toarray()
print(gnb.predict(tfidf.transform(['An excellent movie']).toarray()))  # this must be transformed from sparse matrix to dense data via x.toarray()
print(gnb.predict(tfidf.transform(['I did not like this movie at all']).toarray()))  # this must be transformed from sparse matrix to dense data via x.toarray()

# fit logistic regression model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(train_x_vector, train_y)

print(log_reg.predict(tfidf.transform(['A good movie'])))
print(log_reg.predict(tfidf.transform(['An excellent movie'])))
print(log_reg.predict(tfidf.transform(['I did not like this movie at all'])))

# Evaluate models
# Which model is the most accurate, on average?

# svc.score('Test samples', 'True labels')
svc.score(test_x_vector, test_y)
dec_tree.score(test_x_vector, test_y)
gnb.score(test_x_vector.toarray(), test_y)
log_reg.score(test_x_vector, test_y)


# F1 score for support-vector machine learning model (SVM)
# F1 Score is the weighted average of Precision and Recall. Accuracy is used when the True Positives and True negatives are more important while F1-score is used when the False Negatives and False Positives are crucial. Also, F1 takes into account how the data is distributed, so it’s useful when you have data with imbalance classes.
# https://stackoverflow.com/questions/14117997/what-does-recall-mean-in-machine-learning

from sklearn.metrics import f1_score
f1_score(test_y, svc.predict(test_x_vector),
         labels=['positive', 'negative'],
         average=None)

# classification report
# We can also build a text report showing the main classification metrics that include those calculated before.
from sklearn.metrics import classification_report
print(classification_report(test_y,
                            svc.predict(test_x_vector),
                            labels=['positive', 'negative']))

# Confusion matrix
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(test_y,
                            svc.predict(test_x_vector),
                            labels=['positive', 'negative'])
print(conf_mat)

# Model tuning
# maximize the model performance with GridSearchCV
from sklearn.model_selection import GridSearchCV
#set the parameters
parameters = {'C': [1,4,8,16,32] ,'kernel':['linear', 'rbf']}
svc = SVC()
svc_grid = GridSearchCV(svc,parameters, cv=5)

svc_grid.fit(train_x_vector, train_y)
# GridSearchCV(cv=5, estimator=SVC(),
#              param_grid={'C': [1, 4, 8, 16, 32], 'kernel': ['linear', 'rbf']})
# svc.score(test_x_vector, test_y)
print(svc_grid.best_params_)
print(svc_grid.best_estimator_)
