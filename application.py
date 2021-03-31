from flask import Flask, request, redirect, url_for, flash, jsonify
import json
import os
import time
import numpy as np
import gensim
from scipy import spatial
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import sent_tokenize
import numpy as np
from nltk.corpus import stopwords 
# TODO - Change prints to logging.x("msg")

app = Flask(__name__)

class TextSimilarity:
    def __init__(self):
        try:
            self.model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')
            self.index2word_set = set(self.model.wv.index2word)
        except FileNotFoundError:
            raise FileNotFoundError
            
    def avg_feature_vector(self, sentence, num_features=300):
        words = WordPunctTokenizer().tokenize(sentence)
        feature_vec = np.zeros((num_features, ), dtype='float32')
        n_words = 0
        for word in words:
            if word in self.index2word_set:
                n_words += 1
                feature_vec = np.add(feature_vec, self.model[word])
        return feature_vec

    def similarity(self, sentence1, sentence2):
        vec1, vec2 = self.avg_feature_vector(sentence1), self.avg_feature_vector(sentence2)
        return self.cosine_similarity(vec1, vec2)

    def cosine_similarity(self, vec1, vec2):
        return 1 - spatial.distance.cosine(vec1, vec2)

sim = TextSimilarity()

def find_similar(sent, blob):
    sentences = np.asarray(sent_tokenize(blob))
    sent_blob_sim = np.asarray([sim.similarity(sent, sentences[i]) for i in range(len(sentences))])
    indices_sorted = np.argsort(sent_blob_sim)[::-1]
    sent_sim_sorted = sentences[indices_sorted]
    return sent_sim_sorted[0]

# Load Knowledge Base
with open('data/corona_info.txt', 'r', encoding="utf8") as f:
    blob = f.read().replace("\n", " ")


@app.route('/api/claimcheck', methods=['POST'])
def claim_check():
    data = request.get_json()   
    claim = data.get('claim')
    result = find_similar(claim, blob)
    return jsonify({"result" : result})      

@app.route('/', methods=['GET'])
def hello():
    return "Fact Claims Checker API"