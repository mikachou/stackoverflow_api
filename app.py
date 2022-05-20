from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import spacy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin


nlp = spacy.load('en_core_web_sm')
tfidf = joblib.load('./tfidf.joblib')
model = joblib.load('./model.joblib')
tags_binarizer = joblib.load('./tags.joblib')


def lemmatize(s: str) -> iter:
    # tokenize
    doc = nlp(s)

    # remove punct and stopwords
    tokens = filter(lambda token: not token.is_space and not token.is_punct and not token.is_stop and not token.is_digit, doc)

    # lemmatize
    return map(lambda token: token.lemma_.lower(), tokens)


def predict(title: str , post: str, predict_proba: bool):
    text = title + " " + post
    lemmes = np.array([' '.join(list(lemmatize(text)))])

    X = tfidf.transform(lemmes)

    if predict_proba:
        y_proba = model.predict_proba(X)[0]
        tags = list(dict(sorted(tags_binarizer.ts.count.items())).keys())

        result = list(zip(tags, y_proba))
    else:
        y_bin = model.predict(X)
        y_tags = tags_binarizer.inverse_transform(y_bin)

        result = y_tags

    return result


class Request(BaseModel):
    title: str
    post: str


def tfidf_request(request: Request):
    text = request.title + " " + request.post
    lemmes = np.array([' '.join(list(lemmatize(text)))])

    return tfidf.transform(lemmes)


# Declaring our FastAPI instance
app = FastAPI()


@app.get('/')
def main():
    return { "message": "Hello Openclassrooms!" }


@app.post('/predict/')
def predict(request: Request):
    X = tfidf_request(request)
    y_bin = model.predict(X)

    y_tags = tags_binarizer.inverse_transform(y_bin)[0]

    return y_tags


@app.post('/predict_proba/')
def predict_proba(request: Request):
    X = tfidf_request(request)
    y_proba = model.predict_proba(X)[0]
    tags = list(dict(sorted(tags_binarizer.ts.count.items())).keys())

    proba = dict(zip(tags, y_proba))

    return proba
