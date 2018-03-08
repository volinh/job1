import Loader as loader
import Computer as cp
import setting
import time
import logging
import os

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
    sparse_matrix = cp.reduce_dimention(dict_vecto_tfidf.values(), dictionary, n_components=1000,batch_size=10000)
    shape = sparse_matrix.shape
    logging.info("shape : " + str(shape))
    dense_matrix = list(sparse_matrix.todense())

    # b8 : build tree
    tree = cp.build_tree(dict_id=dict_map_id.keys(),dict_vecto=dense_matrix,dimension=shape[1],amount_tree=10)

    # b9 : search nns in tree
    list_nns = tree.get_nns_by_item(0,20)
    logging.info(list_nns)

def test_compatible_components(tree,dict_map_id):
    pass

if __name__ == "__main__" :

    time_start = time.time()

    # make_folder(setting.FOLDER_DATA_PATH)
    # # es = loader.get_elasticsearch_client()
    # # dict_raw_product = loader.scan_data(es)
    # # dict_product = cp.preprocess_data(dict_raw_product)
    # # cp.transform_vecto_tfidf(cp.split_text(dict_product))
    #
    # # for id,content in dict_product.items():
    # #     print(id)
    # #     print(content)
    #
    # dictionary = loader.load_dictionary(setting.DICTIONARY_PATH)
    # dict_vecto_tfidf = loader.load_dict_vecto_tfidf(setting.DICT_VECTO_TFIDF_PATH)
    # # for id,vecto in dict_vecto_tfidf.items():
    # #     print(id)
    # #     print(vecto)
    # # print(len(dict_vecto_tfidf))
    #
    # sparse = cp.reduce_dimention(dict_vecto_tfidf.values(),dictionary,n_components=50)
    # print(sparse)
    # print(sparse.shape)
    # # print("---------------")
    # # print(sparse.T[0:100].T)
    # # print("---------------")

    a_to_z()

    # dict_vecto_tfidf = loader.load_dict_vecto_tfidf(setting.DICT_VECTO_TFIDF_PATH)
    # count = 0
    # for id,vecto in dict_vecto_tfidf.items():
    #     print(id)
    #     print(vecto)
    #     count +=1
    #     if count>=10:
    #         break
    #
    # count = 0
    # for id in dict_vecto_tfidf.keys():
    #     print(id)
    #     count +=1
    #     if count>=10:
    #         break
    # print(type(list(dict_vecto_tfidf.keys)))


    time_end = time.time()
    logging.info("thời gian chạy : " + str(time_end - time_start) + "s")
