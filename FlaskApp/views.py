from flask import Flask, render_template, url_for, request, redirect
from surprise import dump
from collections import defaultdict

# Always use relative import for custom module
from .package.forms import UserForm
from .config import Config

from pathlib import Path

app = Flask(__name__)
app.config.from_object(Config)

BASE_DIR = Path(__file__).resolve(strict=True).parent

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

    # Regroupement des user_id avec leur content_id et l'estimation associée.
    top_n = defaultdict(list)
    
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def recommend(user_id, num=5):
    file_name = f"{BASE_DIR}/dump_file"
    loaded_predictions, _ = dump.load(file_name)
    top_n = get_top_n(loaded_predictions)
    recs = top_n[user_id][:num]
    return recs


@app.route("/")
def sommaire():
    return (
        "Try /user/ for recommendation API.\n"
        "Try /index/ to get to the index"
    )

@app.route('/index/')
def index():
    description = """ Bienvenue sur l'application de recommendation d'articles
    """
    return render_template('index.html',
                            description=description)

@app.route('/user/', methods = ['GET', 'POST'])
def userid():
    form = UserForm()
    art_dict = dict.fromkeys(['article','estimate'])
    outputs = [{'article' :None, 'estimate':None}]
    if form.validate_on_submit():
        outputs = []
        user_id = form.userid.data
        recs = recommend(user_id)
        for rec in recs:

            art_dict = dict.fromkeys(['article','estimate'])
            art_dict['article'] = rec[0]
            art_dict['estimate'] = rec[1]
            outputs.append(art_dict)

        return render_template('User.html', form=form, posts=outputs)

    return render_template('User.html', title='Choose User', form=form, posts=outputs)


@app.route("/recommendation/<int:user_id>/", methods = ['GET'])
def recommend(user_id, num=5):
    file_name = f"{BASE_DIR}/dump_file"
    loaded_predictions, _ = dump.load(file_name)
    top_n = get_top_n(loaded_predictions)
    recs = top_n[user_id][:num]
    list_reco = []   
    for rec in recs:
        list_reco.append(rec[0])
    text = "Articles recommendés à l'utilisateur " + str(user_id)+": " + str(list_reco)
    return recs

if __name__ == "__main__":
    app.run()