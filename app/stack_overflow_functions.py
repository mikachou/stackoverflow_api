import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

def top_topics(tags_list: iter, part: float) -> dict:
    cv = CountVectorizer(token_pattern='\S+')
    tags_vect = cv.fit_transform(tags_list)
    tags_vect_sum = np.sum(tags_vect.todense(), axis=0)
    return { k: v for (k, v) in sorted(list(zip(cv.get_feature_names_out(),np.array(tags_vect_sum)[0].tolist())), key=lambda tup: tup[1], reverse=True) if v >= part * len(list(tags_list)) }

def simplified_tags(orig_tags: list, allowed_tags: list, alternative: str = None, only_empty: bool = False) -> list:
    # intersection
    simplified_tags = list(set(orig_tags) & set(allowed_tags))

    # other missing tags = alternative param
    if alternative is not None:
        if (only_empty and len(simplified_tags) == 0) \
        or (not only_empty and len(simplified_tags) < len(orig_tags)):
            simplified_tags.append(alternative) # default = "other"

    return simplified_tags

class TagsSimplifier(BaseEstimator, TransformerMixin):
    def __init__(self, part=0.01):
        self.part = part

    def fit(self, X, y=None):
        self.count = top_topics(X, self.part)
        return self

    def transform(self, X, y=None):
        return X.apply(lambda tags: simplified_tags(tags.split(), self.count.keys())).values

    def inverse_transform(self, X, y=None):
        return X

class TagsBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, part=0.01):
        self.part = part
        self.ts = TagsSimplifier(part=self.part)
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
        simp_X = self.ts.fit_transform(X)
        self.mlb.fit(simp_X)
        return self

    def transform(self, X, y=None):
        simp_X = self.ts.transform(X)
        return self.mlb.transform(simp_X)

    def inverse_transform(self, X, y=None):
        return self.mlb.inverse_transform(X)