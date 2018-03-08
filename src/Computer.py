from annoy import AnnoyIndex
import setting
import Loader as loader
from gensim import corpora,models,matutils
from sklearn.decomposition import IncrementalPCA,TruncatedSVD
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
    logging.info("split text")
    dict_bow_product = {}
    for id, content in dict_product.items():
        words = content.split()
        dict_bow_product[id] = words
    return dict_bow_product

def transform_vecto_tfidf(dict_product):
    """
        tạo từ điển
        return vecto tfidf sản phẩm
    """
    logging.info("transform dictionary of product to dict of tfidf vecto")
    dictionary = corpora.Dictionary(dict_product.values())
    # list_filter_word = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq <= doc_freq]
    # dictionary.filter_tokens(list_filter_word)
    loader.save_dictionary(setting.DICTIONARY_PATH,dictionary)
    dict_vecto_bow = {id: dictionary.doc2bow(doc) for id, doc in dict_product.items()}
    tfidf = models.TfidfModel(dict_vecto_bow.values())
    dict_vecto_tfidf = {id: tfidf[bow] for id, bow in dict_vecto_bow.items()}
    loader.save_dict_vecto_tfidf(setting.DICT_VECTO_TFIDF_PATH,dict_vecto_tfidf)
    return dict_vecto_tfidf,dictionary

def reduce_dimention_pca(dict_vecto_tfidf,n_dictionary,n_components,batch_size=2000):
    logging.info("reduce dimention of matrix by pca")
    sparse_matrix_scipy = matutils.corpus2csc(dict_vecto_tfidf,num_terms=n_dictionary)
    ipca = IncrementalPCA(n_components=n_components,batch_size=batch_size,copy=False)
    ipca.fit(sparse_matrix_scipy.T.toarray())
    logging.info("transform")
    sparse_matrix_scipy = ipca.transform(sparse_matrix_scipy.T.toarray())
    return sparse_matrix_scipy

def reduce_dimension_svd(dict_vecto_tfidf,n_dictionary,n_components):
    logging.info("reduce dimention of matrix by svd")
    sparse_matrix_scipy = matutils.corpus2csc(dict_vecto_tfidf, num_terms=n_dictionary)
    ipca = TruncatedSVD(n_components=n_components)
    ipca.fit(sparse_matrix_scipy.T)
    logging.info("transform")
    sparse_matrix_scipy = ipca.transform(sparse_matrix_scipy.T)
    return sparse_matrix_scipy

def make_tree(filePath,dict,dimension,amount_tree):
    logging.info("make a tree")
    t = AnnoyIndex(dimension)
    for id, vecto in dict.items():
        t.add_item(id, vecto)
    t.build(amount_tree)
    t.save(filePath)
    return t

def make_tree(dict_id,dict_vecto,dimension,amount_tree):
    logging.info("make a tree")
    t = AnnoyIndex(dimension)
    for i in dict_id:
        t.add_item(i,dict_vecto[i])
    t.build(amount_tree)
    loader.save_tree(setting.TREE_PATH,t)
    return t

