from elasticsearch import Elasticsearch,Transport,RequestsHttpConnection,helpers
from annoy import AnnoyIndex
from gensim import corpora
import setting
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def get_elasticsearch_client():
    return Elasticsearch(
        [{'host': setting.HOST, 'port': setting.PORT}],
        transport_class=Transport,
        connection_class=RequestsHttpConnection,
        sniff_on_start=True,
        sniff_on_connection_fail=True,
        sniffer_timeout=60,
        timeout=20
    )

def count_data(es):
    logging.info("count amount of data")
    query_count = {
        "query": {
            "match": {
                "domain": "yes24"
            }
        }
    }
    rs = es.count(index="dynamic_data", doc_type="doc", body=query_count)
    logging.info("số lượng sản phẩm : " + str(rs["count"]))

def scan_data(es):
    logging.info("scan data from sever")
    dict_product = {}
    query_scan = {
        "query": {
            "match": {
                "domain": "yes24"
            }
        }
    }
    rs = helpers.scan(es, index="dynamic_data", doc_type="doc", query=query_scan, scroll="1m")

    for doc in rs:
        data = doc["_source"]
        dict_product[doc["_id"]] = data['category'] + " " + data['title']

    for i, v in dict_product.items():
        print(v)

    return dict_product

def load_dictionary(filePath):
    logging.info("load dictionary")
    dictionary = corpora.Dictionary.load_from_text(filePath)
    return dictionary

def load_dict_product(filePath):
    logging.info("load dictionary of product")
    dict_product = {}
    with open(filePath,"r") as file:
        for line in file.readlines():
            arr = line.split(":")
            dict_product[arr[0].strip()] = arr[1].strip()
    return dict_product

def load_dict_vecto_tfidf(filePath):
    logging.info("load dictionary of tfidf product vectors")
    dict_vecto_tfidf = {}
    with open(filePath,"r") as file:
        for line in file.readlines():
            arr = line.split(" ")
            vecto = []
            for i in range(len(arr)):
                if i==0:
                    id = arr[0]
                else:
                    tuple = arr[i].split(":")
                    id_word = int(tuple[0])
                    val = float(tuple[1])
                    vecto.append((id_word,val))
            dict_vecto_tfidf[id] = vecto
    return dict_vecto_tfidf

def load_tree(filePath,dimension):
    logging.info("load a tree")
    t = AnnoyIndex(dimension)
    t.load(filePath)
    return t

def save_dictionary(filePath,dictionary):
    logging.info("save dictionary")
    dictionary.save_as_text(filePath)

def save_dict_product(filePath,dict_product):
    logging.info("save dictionary of product")
    with open(filePath,"w") as file:
        for id ,content in dict_product.items():
            file.writelines( id + " : " + content + "\n")

def save_dict_vecto_tfidf(filePath,dict_vecto_tfidf):
    logging.info("save dictionary of tfidf product vectors")
    with open(filePath,"w") as file:
        for id, vecto in dict_vecto_tfidf.items():
            text = id
            for k in vecto:
                text = text + " " + str(k[0]) + ":" + str(k[1])
            file.writelines(text + "\n")

def save_tree(filePath,tree):
    logging.info("save tree")
    tree.save(filePath)

