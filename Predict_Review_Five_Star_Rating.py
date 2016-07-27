
# coding: utf-8

# ## Given a user’s past reviews on Yelp (available from yelp-challenge dataset)
# ## When the user writes a review for a business she hasn't reviewed before
# ## How likely will it be a Five-Star review?
# 
# - Load data
# - Visualize the data
# - Featurize the data
# - Join tables to populate the features
# - Model the data: Logistic regression
# - Evaluate the model

# # Load data

# In[468]:

import pandas as pd

PATH = '/scratch/xun/docs/yelp_dataset_challenge_academic_dataset/'
biz_df = pd.read_csv(PATH + 'yelp_academic_dataset_business.csv')
user_df = pd.read_csv(PATH + 'yelp_academic_dataset_user.csv')
review_df = pd.read_csv(PATH + 'yelp_academic_dataset_review.csv')


# In[469]:

review_df = review_df.set_index('review_id')
user_df = user_df.set_index('user_id')
biz_df = biz_df.set_index('business_id')


# # Visulize the data

# ## Example: Plot distribution of review star ratings

# In[470]:

import seaborn as sns
get_ipython().magic(u'matplotlib inline')

# Set context to "talk" for figure aesthetics 
sns.set_context(context="talk")
# set plot figure size to larger
sns.set(palette='Set2', rc={"figure.figsize": (15, 8)}, style="ticks")


# In[471]:

ax = sns.countplot(x='stars', data=review_df, hue='type')
# Rmoving spines
sns.despine()


# ## Example: Plot review star ratings by year

# In[472]:

review_df['datetime'] = pd.to_datetime(review_df['date'])
review_df['year'] = review_df['datetime'].dt.year
ax = sns.countplot(x='year', data=review_df, hue='stars')
sns.despine()


# # Featurize the data
# 
# - Convert date string to date delta
#   - For example, business_age
# - Convert strings to categorical features
#   - For example, noise level: quiet, loud, very loud.
# - Drop unused features
#   - For example, business_name

# In[474]:

def calculate_date_delta(df, from_column, to_column):
    datetime = pd.to_datetime(df[from_column])
    time_delta = datetime.max() - datetime
    df[to_column] = time_delta.apply(lambda x: x.days)
    df.drop(from_column, axis=1, inplace=True)


# In[475]:

def to_length(df, from_column, to_column):
    df[to_column] = df[from_column].apply(lambda x: len(x))
    df.drop(from_column, axis=1, inplace=True)


# In[476]:

def drop_columns(df, columns):
    for column in columns:
        df.drop(column, axis=1, inplace=True)


# In[477]:

def to_boolean(df, columns):
    for column in columns:
        to_column = column+'_bool'
        df[to_column] = df[column].apply(lambda x: bool(x))
        df.drop(column, axis=1, inplace=True)


# In[478]:

FILL_WITH = 0.0


# In[479]:

def to_category(df, columns):
    for column in columns:
        df[column] = df[column].astype('category')
        # add FILL_WITH category for fillna() to work w/o error
        if (FILL_WITH not in df[column].cat.categories):
            df[column] = df[column].cat.add_categories([FILL_WITH])
        #print 'categories for ', column, ' include ', df[column].cat.categories


# In[480]:

def category_rename_to_int(df, columns):
    for column in columns:
        df[column].cat.remove_unused_categories()
        size = len(df[column].cat.categories)
        #print 'column ', column, ' has ', size, ' columns, include ', df[column].cat.categories
        df[column] = df[column].cat.rename_categories(range(1, size+1))
        #print 'becomes ', df[column].cat.categories


# In[481]:

review_df.columns.values


# In[482]:

calculate_date_delta(df=review_df, from_column='date', to_column='date_delta')


# In[483]:

to_length(df=review_df, from_column='text', to_column='text_len')


# In[484]:

drop_columns(df=review_df, columns=['type', 'year', 'datetime'])


# In[485]:

review_df.fillna(value=0.0, inplace=True)


# In[486]:

calculate_date_delta(df=user_df, from_column='yelping_since', to_column='date_delta')


# In[487]:

to_length(df=user_df, from_column='friends', to_column='friends_count')


# In[488]:

to_length(df=user_df, from_column='elite', to_column='elite_count')


# In[489]:

drop_columns(df=user_df, columns=['name', 'type'])


# In[490]:

user_df.fillna(value=0.0, inplace=True)


# In[491]:

drop_columns(
    df=biz_df,
    columns=[
        'type',
        'name',
        'city',
        'full_address',
        'state',
        'categories',
        'longitude',
        'latitude',
        'neighborhoods',
        'hours.Monday.open',
        'hours.Monday.close',
        'hours.Tuesday.open',
        'hours.Tuesday.close',
        'hours.Wednesday.open',
        'hours.Wednesday.close',
        'hours.Thursday.open',
        'hours.Thursday.close',
        'hours.Friday.open',
        'hours.Friday.close',
        'hours.Saturday.open',
        'hours.Saturday.close',
        'hours.Sunday.open',
        'hours.Sunday.close',
    ]
)


# In[492]:

to_cat_columns = [
    'attributes.Ambience.casual',
    'attributes.Attire',
    'attributes.Alcohol',
    'attributes.Noise Level',
    'attributes.Smoking',
    'attributes.Wi-Fi',
    'attributes.Ages Allowed',
]
to_category(
    df=biz_df,
    columns=to_cat_columns,
)


# In[493]:

biz_df.fillna(value=FILL_WITH, inplace=True)


# In[494]:

category_rename_to_int(
    df=biz_df,
    columns=to_cat_columns,
)


# # Join tables to populate the features
# 
# Join three tables (review, biz, user) to one (review-with-all-info).
# Each join is a many-to-one join.

# In[495]:

# The `user_df` DataFrame is already indexed by the join key (`user_id`). Make sure it's on the right side of join.
review_join_user = review_df.join(user_df, on='user_id', lsuffix='_review', rsuffix='_user')


# In[496]:

review_join_user_join_biz = review_join_user.join(biz_df, on='business_id', rsuffix='_biz')


# In[497]:

drop_columns(df=review_join_user_join_biz, columns=['user_id', 'business_id'])


# In[498]:

#review_join_user_join_biz.columns.values


# # Identify data X and target y

# In[500]:

# target y is whether a review is five-star
y = review_join_user_join_biz.stars.apply(lambda x: x == 5)

# We've already dropped not informative features data X
X = review_join_user_join_biz
review_join_user_join_biz.drop('stars', axis=1, inplace=True)

# get the feature names - this will be useful for the model visualization and feature analysis
features = X.columns.values


# In[501]:

from sklearn import preprocessing

def label_encode(df, columns):
    label_encoders = []
    for column in columns:
        le = preprocessing.LabelEncoder()
        label_encoders.append(le)

        #to convert into numbers
        df[column] = le.fit_transform(df[column])
        
    return label_encoders

def label_decode(df, columns, label_encoders):
    # XXX(xun): columns have to be the same order as label_encode()
    for column, le in zip(columns, label_encoders):
        #to convert back
        df[column] = le.inverse_transform(df[column])


# In[502]:

columns_need_encode = [
    'attributes.BYOB/Corkage',
]
les = label_encode(
    df=X,
    columns=columns_need_encode,
)
#label_decode(df=df, columns=columns_need_encode, label_encoders=les)


# In[504]:

from sklearn.cross_validation import train_test_split

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[505]:

print 'training data shape', X_train.shape
print 'test data shape', X_test.shape
print 'converted label data shape', y_train.shape
print 'features', features


# # Model the data: Logistic regression
# 
# Estimate the probability of a binary response based on one or more features. The probability of a review being five-star.

# In[506]:

# Standardize features by removing the mean and scaling to unit variance
scaler = preprocessing.StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[514]:

from sklearn.cross_validation import cross_val_score
import numpy as np

# Function used to print cross-validation scores
def training_score(est, X, y, cv):
    acc = cross_val_score(est, X, y, cv = cv, scoring='accuracy')
    roc = cross_val_score(est, X, y, cv = cv, scoring='roc_auc')
    print '5-fold Train CV | Accuracy:', round(np.mean(acc), 3),'+/-',     round(np.std(acc), 3),'| ROC AUC:', round(np.mean(roc), 3), '+/-', round(np.std(roc), 3)


# In[515]:

from sklearn import linear_model
from sklearn.cross_validation import StratifiedKFold

# Build model using default parameter values
lrc = linear_model.LogisticRegression()


# In[ ]:

# cross-validation 
cv = StratifiedKFold(y_train, n_folds=5, shuffle=True)


# In[517]:

# print cross-validation scores
training_score(
    est=lrc,
    X=X_train_scaled,
    y=y_train,
    cv=cv,
)


# # Evaluation via Confusion Matrix 
# False positive (upper right); 
# False negative (bottom left)

# In[519]:

# Compute confusion matrix
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Run classifier
lrc_fit = lrc.fit(X_train_scaled, y_train)
y_pred = lrc_fit.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)


# In[521]:

'''
# Compute confusion matrix
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)
'''

# Normalize the confusion matrix by row (i.e by the number of samples in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
plt.show()

