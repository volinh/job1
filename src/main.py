import Loader as loader
import Computer as cp
import setting
import time
import logging
import os
import sys,getopt
from gensim import matutils
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def make_folder(folderPath):
    if os.path.exists(folderPath):
        logging.info("folder exists")
        pass
    else:
        logging.info("make a folder")
        os.mkdir(path=folderPath)

def map_id_product(list_id_product):
    logging.info("map list of product's id")
    dict_map_id = {}
    i = 0
    for id in list_id_product:
        dict_map_id[i] = id
        i +=1
    return dict_map_id

def a_to_z():
    logging.info("A -> Z")
    # b1 : make a folder to contain data
    make_folder(setting.FOLDER_DATA_PATH)

    # b2 : return a client connecting to server elasticsearch
    es = loader.get_elasticsearch_client()

    # b3 : scan data
    dict_raw_product = loader.scan_data(es)

    # b4 : preprocess raw data
    dict_product = cp.preprocess_data(dict_raw_product)

    # b5 : transform to tfidf vector
    dict_vecto_tfidf,dictionary = cp.transform_vecto_tfidf(cp.split_text(dict_product))

    # b6 : map id product with nonnegative integer
    list_id_product = dict_vecto_tfidf.keys()
    dict_map_id = map_id_product(list_id_product)

    # b7 : reduce large matrix
    # sparse_matrix = cp.reduce_dimention(dict_vecto_tfidf.values(), len(dictionary), n_components=500,batch_size=20000)
    sparse_matrix = cp.reduce_dimension_svd(dict_vecto_tfidf.values(), len(dictionary), n_components=500)
    shape = sparse_matrix.shape
    logging.info("shape : " + str(shape))
    dense_matrix = list(sparse_matrix)

    # b8 : build tree
    tree = cp.make_tree(dict_id=dict_map_id.keys(),dict_vecto=dense_matrix,dimension=shape[1],amount_tree=10)

    # b9 : search nns in tree
    list_nns = tree.get_nns_by_item(0,20)
    logging.info(list_nns)

def b_to_z(n_components):
    logging.info("B -> Z")
    dictionary = loader.load_dictionary(setting.DICTIONARY_PATH)
    dict_vecto_tfidf = loader.load_dict_vecto_tfidf(setting.DICT_VECTO_TFIDF_PATH)
    # sparse_matrix = cp.reduce_dimention_pca(dict_vecto_tfidf.values(), len(dictionary), n_components=500, batch_size=20000)
    # b6 : map id product with nonnegative integer
    list_id_product = dict_vecto_tfidf.keys()
    dict_map_id = map_id_product(list_id_product)

    # b7 : reduce large matrix
    # sparse_matrix = cp.reduce_dimention(dict_vecto_tfidf.values(), len(dictionary), n_components=500,batch_size=20000)
    sparse_matrix = cp.reduce_dimension_svd(dict_vecto_tfidf.values(), len(dictionary), n_components=n_components)
    shape = sparse_matrix.shape
    logging.info("shape : " + str(shape))
    dense_matrix = list(sparse_matrix)

    # b8 : build tree
    tree = cp.make_tree(dict_id=dict_map_id.keys(), dict_vecto=dense_matrix, dimension=shape[1], amount_tree=10)

    # b9 : search nns in tree
    dict_result = {}
    for i in range(1000):
        list_nns = tree.get_nns_by_item(i, 11)
        dict_result[i] = list_nns
    dict_product = loader.load_dict_product(setting.DICT_PRODUCT_PATH)
    loader.save_result(setting.DICT_RESULT_PATH + "_" + str(n_components),dict_product,dict_map_id,dict_result)

def prepare_data():
    # b1 : make a folder to contain data
    make_folder(setting.FOLDER_DATA_PATH)

    # b2 : return a client connecting to server elasticsearch
    es = loader.get_elasticsearch_client()

    # b3 : scan data
    dict_raw_product = loader.scan_data(es)

    # b4 : preprocess raw data
    dict_product = cp.preprocess_data(dict_raw_product)

    # b5 : transform to tfidf vector
    cp.transform_vecto_tfidf(cp.split_text(dict_product))

def compute_data(n_components,dictionary,dict_vecto_tfidf,items=1000,amount_nns=11):
    # b6 : map id product with nonnegative integer
    list_id_product = dict_vecto_tfidf.keys()
    dict_map_id = map_id_product(list_id_product)

    # b7 : reduce large matrix
    sparse_matrix = cp.reduce_dimension_svd(dict_vecto_tfidf.values(), len(dictionary), n_components=n_components)
    shape = sparse_matrix.shape
    logging.info("shape : " + str(shape))
    dense_matrix = list(sparse_matrix)

    # b8 : build tree
    tree = cp.make_tree(dict_id=dict_map_id.keys(), dict_vecto=dense_matrix, dimension=shape[1], amount_tree=10)

    # b9 : search nns in tree
    dict_result = {}
    for i in range(items):
        list_nns = tree.get_nns_by_item(i, amount_nns)
        dict_result[i] = list_nns
    dict_product = loader.load_dict_product(setting.DICT_PRODUCT_PATH)
    loader.save_result(setting.DICT_RESULT_PATH + "_" + str(n_components), dict_product, dict_map_id, dict_result)

def test_compatible_components(tree,dict_map_id):
    pass

if __name__ == "__main__" :
    time_start = time.time()
    prepare_data()
    dictionary = loader.load_dictionary(setting.DICTIONARY_PATH)
    dict_vecto_tfidf = loader.load_dict_vecto_tfidf(setting.DICT_VECTO_TFIDF_PATH)
    for i in range(101):
        if i == 0:
            continue
        else:
            n_components = i*10
            compute_data(n_components=n_components,dictionary=dictionary,dict_vecto_tfidf=dict_vecto_tfidf)
    time_end = time.time()
    logging.info("thời gian chạy : " + str(time_end - time_start) + "s")
