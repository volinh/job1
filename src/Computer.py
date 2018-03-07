from annoy import AnnoyIndex
from src import setting
from src import Loader as loader
from gensim import corpora,models,matutils
import logging
import re

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# class FeatureExtraction(object):
#     def __init__(self,data):
#         self.data = data
#
#     # def

def preprocess_data(dict_raw_product):
    """
        return van ban tien xu ly
    """
    logging.info("preprocess data")
    dict_product = {}
    for id, content in dict_raw_product.items():
        content1 = re.sub(";", " ", content)
        content2 = re.sub("/", " ", content1)
        content3 = re.sub(",", " ", content2)
        words = content3.split()
        new_content = ""
        for word in words:
            word = word.strip(setting.SPECIAL_CHARACTER).lower()
            if word != '':
                new_content = new_content + " " + word
        dict_product[id] = new_content
    loader.save_dict_product(setting.DICT_PRODUCT_PATH,dict_product)
    return dict_product

def split_text(dict_product):
    dict_bow_product = {}
    for id, content in dict_product.items():
        words = content.split()
        dict_bow_product[id] = words
    return dict_bow_product

def tranform_vecto_tfidf(dict_product):
    """
        tạo từ điển
        return vecto tfidf sản phẩm
    """
    dictionary = corpora.Dictionary(dict_product.values())
    # list_filter_word = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq <= doc_freq]
    # dictionary.filter_tokens(list_filter_word)
    loader.save_dictionary(setting.DICTIONARY_PATH,dictionary)
    dict_vecto_bow = {id: dictionary.doc2bow(doc) for id, doc in dict_product.items()}
    tfidf = models.TfidfModel(dict_vecto_bow.values())
    dict_vecto_tfidf = {id: tfidf[bow] for id, bow in dict_vecto_bow.items()}
    loader.save_dict_vecto_tfidf(setting.DICT_VECTO_TFIDF_PATH,dict_vecto_tfidf)
    return dict_vecto_tfidf

def reduce_dimention():
    pass