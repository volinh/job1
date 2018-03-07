from elasticsearch import Elasticsearch,Transport,RequestsHttpConnection,helpers
from gensim import corpora
from src import setting
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
    dictionary = corpora.Dictionary.load_from_text(filePath)
    return dictionary

def load_dict_product(filePath):
    dict_product = {}
    with open(filePath,"r") as file:
        for line in file.readlines():
            arr = line.split(":")
            dict_product[arr[0].strip()] = arr[1].strip()
    return dict_product

def save_dictionary(filePath,dictionary):
    dictionary.save_as_text(filePath)
    logging.info("save dictionary")

def save_dict_product(filePath,dict_product):
    with open(filePath,"w") as file:
        for id ,content in dict_product.items():
            file.writelines( id + " : " + content + "\n")
    logging.info("save dictionary of product")

def save_dict_vecto_tfidf(filePath,dict_vecto_tfidf):
    with open(filePath,"w") as file:
        for id, vecto in dict_vecto_tfidf.items():
            text = id
            for k in vecto:
                text = text + " " + str(k[0]) + ":" + str(k[1])
            file.writelines(text + "\n")
    logging.info("save dictionary of tfidf product vectors")



