import numpy as np
import pandas as pd
import os
import sklearn
import math

from operator import itemgetter

from sklearn.model_selection import train_test_split


#1.  Loading Metadata Dataset
meta_df= pd.read_csv("./news-portal-user-interactions-by-globocom/articles_metadata.csv")


#2. Loading and regrouping clicks files together
if not os.path.exists('./news-portal-user-interactions-by-globocom/clicks_sum.csv'):
    file_dir = "./news-portal-user-interactions-by-globocom/clicks"
    file_path_list = sorted(
        [
            os.path.join(file_dir, file_name) 
            for file_name in os.listdir(file_dir)
            if file_name.endswith(".csv")
        ]
    )
    
    file_df_list = []
    
    for file_path in file_path_list:
        df = pd.read_csv(file_path)
        file_df_list.append(df)
    
    clicks_sum = pd.concat(file_df_list, ignore_index=True)
    clicks_sum.to_csv('./news-portal-user-interactions-by-globocom/clicks_sum.csv')
else:
    clicks_sum= pd.read_csv('./news-portal-user-interactions-by-globocom/clicks_sum.csv')

# 3. Filtering datas to the ones needed
filtered_data = clicks_sum.merge(meta_df,
                            left_on='click_article_id',
                            right_on='article_id')[["user_id",
                             "article_id",
                             "category_id",
                              "session_size"]]

#4. Loading Pickle file containing embeddings
pickle = pd.read_pickle('./news-portal-user-interactions-by-globocom/articles_embeddings.pickle')

# Function to spread rating properly
def smooth_user_preference(x):
    return math.log(1+x, 2)    
    
#4. Applying Log function to the rating
rating_cat_log = filtered_data.groupby(["user_id", "article_id"])["session_size"].sum()\
                    .apply(smooth_user_preference).reset_index()


#5. Define function to create content base filter based on https://www.kaggle.com/code/gspmoreira/recommender-systems-in-python-101/notebook

###################################
def find_top_n_indices(data, top=5):
    indexed = enumerate(data)
    sorted_data = sorted(indexed, 
                         key=itemgetter(1), 
                         reverse=True) 
    return [d[0] for d in sorted_data[:top]] 

def recommend_Article(user_id, top):
    interactions_indexed_df = interactions_train_df[interactions_train_df['article_id'] \
                                                   .isin(meta_df['article_id'])].set_index('user_id')
    
    user_profiles = build_users_profile(user_id, interactions_indexed_df)
    score = []
    for i in range(0, len(pickle)):
        cos_sim = np.dot(user_profiles, pickle[i])/(np.linalg.norm(user_profiles)*np.linalg.norm(pickle[i]))
        score.append(cos_sim)
    
    _best_scores = find_top_n_indices(score, top)
    print("Recommending " + str(top) + " articles to user " + str(user_id) + "...")   
    print("-------")
    
    for best in _best_scores:
        print("Recommended: " + str(best))

###################################
def get_item_profile(item_id):
    idx = item_ids.index(item_id)
    item_profile = pickle[idx:idx+1]
    return item_profile

def get_item_profiles(ids):
    item_profiles_list = [get_item_profile(x) for x in ids]
    item_profiles = np.vstack(item_profiles_list)
    return item_profiles

def build_users_profile(person_id, interactions_indexed_df):
    interactions_person_df = interactions_indexed_df.loc[[person_id]] #double [[]] pour les index simple qui retourne une valeur et non un df
    user_item_profiles = get_item_profiles(interactions_person_df['article_id'])
    
    user_item_strengths = np.array(interactions_person_df['session_size']).reshape(-1,1)
    #Weighted average of item profiles by the interactions strength
    user_item_strengths_weighted_avg = np.sum(np.multiply(user_item_profiles,user_item_strengths), axis=0) / np.sum(user_item_strengths)
    user_item_strengths_weighted_avg = np.array(user_item_strengths_weighted_avg).reshape(1, -1)
    user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
    return user_profile_norm

def build_users_profiles(): 
    interactions_indexed_df = interactions_train_df[interactions_train_df['article_id'] \
                                                   .isin(meta_df['article_id'])].set_index('user_id')
    user_profiles = {}
    for person_id in interactions_indexed_df.index.unique():
        user_profiles[person_id] = build_users_profile(person_id, interactions_indexed_df)
    return user_profiles

###################################

# Creating Tran and test DF
interactions_train_df, interactions_test_df = sklearn.model_selection.train_test_split(rating_cat_log,
                                   stratify=rating_cat_log['user_id'], 
                                   test_size=0.25,
                                   random_state=0)


item_ids = meta_df['article_id'].tolist()

interactions_indexed_df = interactions_train_df[interactions_train_df['article_id'] \
                                                   .isin(meta_df['article_id'])].set_index('user_id')


# Testing out the script
# recommend_Article(10000,5)