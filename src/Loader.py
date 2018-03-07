from elasticsearch import Elasticsearch,Transport,RequestsHttpConnection,helpers
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

def counting_data(es):
    query_count = {
        "query": {
            "match": {
                "domain": "yes24"
            }
        }
    }
    rs = es.count(index="dynamic_data", doc_type="doc", body=query_count)
    logging.info("số lượng sản phẩm : " + str(rs["count"]))

def scanning_data(es):
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

def loading_dictionary(filepath):
    pass

def loading_corpus(filepath):
    pass
