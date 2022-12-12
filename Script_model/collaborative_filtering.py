#0. Packages import
import pandas as pd
import os
import math
import surprise

from collections import defaultdict

from surprise import Reader, Dataset
from surprise import SVD, dump

        
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
 
# Function to spread rating properly
def smooth_user_preference(x):
    return math.log(1+x, 2)    
    
#4. Applying Log function to the rating
rating_cat_log = filtered_data.groupby(["user_id", "article_id"])["session_size"].sum()\
                    .apply(smooth_user_preference).reset_index()

#5. Using Surprise Package
# Initializing Surprise reader
reader = Reader(rating_scale=(1, 10))

# Creating Surprise Dataset
data = Dataset.load_from_df(rating_cat_log[["user_id", "article_id", "session_size"]],
                            reader)

# Split de Dataset in train and test sets
trainset, testset = surprise.model_selection.train_test_split(data, test_size=0.25, random_state=0)

#5. Choose and train algorithm

# Choose the prediction algorithm (matrix factorization SVD here)
algo_best = SVD(n_epochs= 20,lr_all=0.01, reg_all=0.4)

# Fit algorithm
algo_best.fit(trainset)

# Compute predictions of the 'original' algorithm.
predictions = algo_best.test(testset)

# Dump predictions for future use in API
file_name = os.path.expanduser("dump_file") # create the dimp file if not existing
dump.dump("dump_file",
           predictions=predictions,
         )

file_name = os.path.expanduser("dump_file")
loaded_predictions, loaded_algo = dump.load(file_name)


# Function from https://surprise.readthedocs.io/en/stable/FAQ.html
def get_top_n(predictions, n=5):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# Function to display the result. To be changed according to the API display
def recommend(user_id, num):
    print("Recommending " + str(num) + " articles to user " + str(user_id) + "...")   
    print("-------")
    top_n = get_top_n(loaded_predictions, num)   
    recs = top_n[user_id][:num]   
    for rec in recs: 
        print("Recommended: " + str(rec[0]) + " (score:" +      str(rec[1]) + ")")

# Testing out the script
# recommend(10000,5)