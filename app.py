#----------------------------------------------------------------------------#
# Imports
#----------------------------------------------------------------------------#
import flask
from flask import Flask, render_template, request
# from flask.ext.sqlalchemy import SQLAlchemy
import logging
from logging import Formatter, FileHandler
from forms import *
import os
import NetworkModel

import numpy as np
import re
import pandas as pd
import json
import gzip
import sqlite3
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import ssl

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

#----------------------------------------------------------------------------#
# App Config.
#----------------------------------------------------------------------------#

app = Flask(__name__)
app.config.from_object('config')
#db = SQLAlchemy(app)

# Automatically tear down SQLAlchemy.
'''
@app.teardown_request
def shutdown_session(exception=None):
    db_session.remove()
'''

# Login required decorator.
'''
def login_required(test):
    @wraps(test)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return test(*args, **kwargs)
        else:
            flash('You need to login first.')
            return redirect(url_for('login'))
    return wrap
'''
#----------------------------------------------------------------------------#
# Controllers.
#----------------------------------------------------------------------------#


@app.route('/')
def home():
    return render_template('pages/home.html')

@app.route('/article')
def about():
    return render_template('pages/inputpage.html')

@app.route('/analysis', methods= ['POST', 'GET'])
def analysis():
    if request.method == 'POST':
        result = request.form
        val = NetworkModel.analyzeText(result.get('message'))
        taglist = ""
        for x in range(len(val['tag list'])):
            if(x<len(val['tag list'])-1):
                taglist+= val['tag list'][x]+", "
            else:
                taglist+= val['tag list'][x]
        percentile = ""
        if val['percents'][0]=='abv75p':
            percentile = "top 25%"
        elif val['percents'][0]=='abv25p':
            percentile = "bottom 50%"
        elif val['percents'][0]=='abv50p':
            percentile = "top 50%"
        else:
            percentile = "lower 25%"
        ls = ['Tag_Finance', 'Tag_Analytics', 'Tag_Company', 'Tag_Hospitality', 'Tag_National', 'Tag_Healthcare']
        images = []
        for i in ls:
            if i in val['tag list']:
                images.append(i.split('_')[1]+".png")
        tagacc = str(round(val['tag acc']*100, 2))+"%"
        peracc = str(round(val['percentile acc']*100, 2))+"%"
        return render_template("pages/analysis.html",
            images = images,
            similars = val['similars'],
            taglist = taglist,
            perlist=percentile,
            tagacc=tagacc,
            peracc = peracc)

'''
@app.route('/login')
def login():
    form = LoginForm(request.form)
    return render_template('forms/login.html', form=form)
'''

# Error handlers.


@app.errorhandler(500)
def internal_error(error):
    #db_session.rollback()
    return render_template('errors/500.html'), 500


@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

if not app.debug:
    file_handler = FileHandler('error.log')
    file_handler.setFormatter(
        Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    )
    app.logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.info('errors')

#----------------------------------------------------------------------------#
# Launch.
#----------------------------------------------------------------------------#

# Default port:
if __name__ == '__main__':
    app.run()

# Or specify port manually:
'''
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
'''
