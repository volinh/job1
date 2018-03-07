from src import Loader as loader
from src import Computer as cp
from src import setting
import time
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def make_folder(folderPath):
    if os.path.exists(folderPath):
        pass
    else:
        os.mkdir(path=folderPath)

if __name__ == "__main__" :

    time_start = time.time()

    make_folder(setting.FOLDER_DATA_PATH)
    es = loader.get_elasticsearch_client()
    dict_raw_product = loader.scan_data(es)
    dict_product = cp.preprocess_data(dict_raw_product)
    cp.tranform_vecto_tfidf(cp.split_text(dict_product))

    # for id,content in dict_product.items():
    #     print(id)
    #     print(content)


    time_end = time.time()
    logging.info("thời gian chạy : " + str(time_end - time_start) + "s")
