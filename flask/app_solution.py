# minimal example from:
# http://flask.pocoo.org/docs/quickstart/

import pickle
import numpy as np
import pandas as pd
import flask
from flask import render_template, request, Flask
import re
from sklearn.feature_extraction.text import CountVectorizer
from corextopic import corextopic as ct

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)

app = Flask(__name__)  # create instance of Flask class


topic_list = ['Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5']


@app.route("/topic", methods=["POST", "GET"])
def predict_final():

    t1 = [request.args.get('topic10'), request.args.get('topic11'), request.args.get('topic12')]
    t1 = [s.lower() for s in t1 if s]
    t2 = [request.args.get('topic20'), request.args.get('topic21'), request.args.get('topic22')]
    t2 = [s.lower() for s in t2 if s]
    t3 = [request.args.get('topic30'), request.args.get('topic31'), request.args.get('topic32')]
    t3 = [s.lower() for s in t3 if s]
    t4 = [request.args.get('topic40'), request.args.get('topic41'), request.args.get('topic42')]
    t4 = [s.lower() for s in t4 if s]
    t5 = [request.args.get('topic50'), request.args.get('topic51'), request.args.get('topic52')]
    t5 = [s.lower() for s in t5 if s]
    anchors = [t1, t2, t3, t4, t5]

    infile = open('stopwords_final', 'rb')
    stopwords = pickle.load(infile)
    infile.close()
    df = pd.read_pickle('cm_19_06')
    df.drop_duplicates(subset='body', keep=False, inplace=True)
    df = df[df['author'] != 'Ilackfocus']
    df = df[df['author'] != '[deleted]']
    df = df[df['author'] != 'AutoModerator']

    token_pattern_no_number = u'(?ui)\\b\\w*[a-zA-Z]+\\w*\\b'
    vectorizer_corex = CountVectorizer(stop_words=stopwords,
                                       binary=True,
                                       token_pattern=token_pattern_no_number,
                                       ngram_range=(1, 2),
                                       max_df=0.5,
                                       min_df=2,
                                       max_features=20000)
    c_word = vectorizer_corex.fit_transform(df['body'])
    vocab = vectorizer_corex.get_feature_names()

    ct_model = ct.Corex(n_hidden=5, seed=42)
    c_model_fitted = ct_model.fit(c_word, words=vocab, anchors=anchors, anchor_strength=4)

    topic_dist = []
    topic_count = np.asarray(c_model_fitted.labels).sum(axis=0)
    for i, topic_ngrams in enumerate(topic_count):
        topic_dist.append(round((topic_count[i] / len(c_model_fitted.labels)) * 100, 2))

    wd = []
    for i, topic_ngrams in enumerate(c_model_fitted.get_topics(n_words=10)):
        wd.append([ngram[0] for ngram in topic_ngrams if ngram[1] > 0])
    top_df = pd.DataFrame(data={'Topic': [1, 2, 3, 4, 5],
                                '% of all comments': topic_dist})
    top_df['Keywords'] = pd.Series(anchors)
    top_df['Top Words'] = pd.Series(wd)
    top_df.set_index('Topic', inplace=True)

    df['Topic1'] = c_model_fitted.labels[:, 0]
    df['Topic2'] = c_model_fitted.labels[:, 1]
    df['Topic3'] = c_model_fitted.labels[:, 2]
    df['Topic4'] = c_model_fitted.labels[:, 3]
    df['Topic5'] = c_model_fitted.labels[:, 4]

    c1 = df[df['Topic1'] == True][['created_utc', 'author', 'score', 'body',
                                   'Topic1', 'Topic2', 'Topic3', 'Topic4', 'Topic5']].sample(5)
    c2 = df[df['Topic2'] == True][['created_utc', 'author', 'score', 'body',
                                   'Topic1', 'Topic2', 'Topic3', 'Topic4', 'Topic5']].sample(5)
    c3 = df[df['Topic3'] == True][['created_utc', 'author', 'score', 'body',
                                   'Topic1', 'Topic2', 'Topic3', 'Topic4', 'Topic5']].sample(5)
    c4 = df[df['Topic4'] == True][['created_utc', 'author', 'score', 'body',
                                   'Topic1', 'Topic2', 'Topic3', 'Topic4', 'Topic5']].sample(5)
    c5 = df[df['Topic5'] == True][['created_utc', 'author', 'score', 'body',
                                   'Topic1', 'Topic2', 'Topic3', 'Topic4', 'Topic5']].sample(5)

    comment_df = pd.concat([c1, c2, c3, c4, c5], ignore_index=True)

    comment_df['date'] = pd.to_datetime(comment_df['created_utc'], unit='s').dt.strftime('%m/%d/%Y')

    comment_df = comment_df[['date', 'author', 'score', 'body',
                             'Topic1', 'Topic2', 'Topic3', 'Topic4', 'Topic5']]


    return flask.render_template('predict_final.html', table1=[top_df.to_html(table_id='cm')],
                                 table2=[comment_df.to_html(table_id='sc')])



if __name__ == '__main__':
    app.run(debug=True)
