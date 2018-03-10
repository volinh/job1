import sys
import getopt
import Loader as loader

# HOST = "10.5.36.17"
# PORT = 9200
# SPECIAL_CHARACTER = '%@$.,=+-_!;/()[]*"&^:#|\n\t\'\\'
# PARENT_FOLDER_PATH = "../"
# FOLDER_DATA_PATH = "../data"
# DICT_PRODUCT_PATH = "../data/dict_product"
# DICT_RAW_PRODUCT_PATH = "../data/dict_raw_product"
# DICT_VECTO_TFIDF_PATH = "../data/dict_vecto_tfidf"
# DICTIONARY_PATH = "../data/dictionary"
# TREE_PATH = "../data/tree"
# DICT_RESULT_PATH = "../data/dict_result"

class Setting(object):
    def __init__(self,filePath):
        self.file_config_path = filePath

    def assigned_parameter(self):
        dict_param = loader.load_config_file(self.file_config_path)
        self.HOST = dict_param["HOST"]
        self.PORT = int(dict_param["PORT"])
        self.SPECIAL_CHARACTER = '%@$.,=+-_!;/()[]*"&^:#|\n\t\'\\'
        self.PARENT_FOLDER_PATH = dict_param["PARENT_FOLDER_PATH"]
        self.FOLDER_DATA_PATH = dict_param["FOLDER_DATA_PATH"]
        self.DICT_PRODUCT_PATH = dict_param["DICT_PRODUCT_PATH"]
        self.DICT_RAW_PRODUCT_PATH = dict_param["DICT_RAW_PRODUCT_PATH"]
        self.DICT_VECTO_TFIDF_PATH = dict_param["DICT_VECTO_TFIDF_PATH"]
        self.DICTIONARY_PATH = dict_param["DICTIONARY_PATH"]
        self.TREE_PATH = dict_param["TREE_PATH"]
        self.DICT_RESULT_PATH = dict_param["DICT_RESULT_PATH"]
        self.DICT_RESULT_ID_PATH = dict_param["DICT_RESULT_ID_PATH"]
        self.FOLDER_HDFS_PATH = dict_param["FOLDER_HDFS_PATH"]
        self.FILE_NAME_HDFS = dict_param["FILE_NAME_HDFS"]
        self.DOMAIN = dict_param["DOMAIN"]
