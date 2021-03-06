from elasticsearch import Elasticsearch, Transport, RequestsHttpConnection, helpers
from annoy import AnnoyIndex
from gensim import corpora
import logging,subprocess

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_elasticsearch_client(host,port):
    return Elasticsearch(
        [{'host': host, 'port': port}],
        transport_class=Transport,
        connection_class=RequestsHttpConnection,
        sniff_on_start=True,
        sniff_on_connection_fail=True,
        sniffer_timeout=60,
        timeout=20
    )


def count_data(es,domain):
    logging.info("count amount of data")
    query_count = {
        "query": {
            "match": {
                "domain": domain
            }
        }
    }
    rs = es.count(index="dynamic_data", doc_type="doc", body=query_count)
    logging.info("số lượng sản phẩm : " + str(rs["count"]))


def scan_data(es,domain):
    logging.info("scan data from sever")
    dict_product = {}
    query_scan = {
        "query": {
            "match": {
                "domain": domain
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


def load_config_file(filePath):
    dict_param = {}
    with open(filePath) as f:
        list_param = f.readlines()
        for param in list_param:
            if param.startswith("#") or param.startswith("\n"):
                continue
            key, value = param.split("=")
            dict_param[key.strip()] = value.strip()
    return dict_param


def load_dictionary(filePath):
    logging.info("load dictionary")
    dictionary = corpora.Dictionary.load_from_text(filePath)
    return dictionary


def load_dict_product(filePath):
    logging.info("load dictionary of product")
    dict_product = {}
    with open(filePath, "r") as file:
        for line in file.readlines():
            arr = line.split(":")
            dict_product[arr[0].strip()] = arr[1].strip()
    return dict_product


def load_dict_vecto_tfidf(filePath):
    logging.info("load dictionary of tfidf product vectors")
    dict_vecto_tfidf = {}
    with open(filePath, "r") as file:
        for line in file.readlines():
            arr = line.split(" ")
            vecto = []
            for i in range(len(arr)):
                if i == 0:
                    id = arr[0]
                else:
                    tuple = arr[i].split(":")
                    id_word = int(tuple[0])
                    val = float(tuple[1])
                    vecto.append((id_word, val))
            dict_vecto_tfidf[id] = vecto
    return dict_vecto_tfidf


def load_tree(filePath, dimension):
    logging.info("load a tree")
    t = AnnoyIndex(dimension)
    t.load(filePath)
    return t


def save_dictionary(filePath, dictionary):
    logging.info("save dictionary")
    dictionary.save_as_text(filePath)


def save_dict_product(filePath, dict_product):
    logging.info("save dictionary of product")
    with open(filePath, "w") as file:
        for id, content in dict_product.items():
            file.writelines(id + " : " + content + "\n")


def save_dict_vecto_tfidf(filePath, dict_vecto_tfidf):
    logging.info("save dictionary of tfidf product vectors")
    with open(filePath, "w") as file:
        for id, vecto in dict_vecto_tfidf.items():
            text = id
            for k in vecto:
                text = text + " " + str(k[0]) + ":" + str(k[1])
            file.writelines(text + "\n")


def save_tree(filePath, tree):
    logging.info("save tree")
    tree.save(filePath)


def save_result(filePath, dict_product, dict_map_id, dict_result):
    logging.info("save result")
    with open(filePath, "w") as file:
        for id, value in dict_result.items():
            real_id = dict_map_id[id]
            text_id = dict_product[real_id]
            file.writelines("-----------------------------\n")
            file.writelines(real_id + " : " + text_id + "\n")
            print(value)
            for id_nns in value:
                real_id_nns = dict_map_id[id_nns]
                text_id_nns = dict_product[real_id_nns]
                file.writelines(real_id_nns + " : " + text_id_nns + "\n")
            file.writelines("-----------------------------\n")


def save_result_id(filePath, dict_map_id, dict_result_id):
    logging.info("save result id")
    with open(filePath, "w") as file:
        for id, value in dict_result_id.items():
            real_id = dict_map_id[id]
            line = real_id
            stt = 0
            for id_nns in value:
                stt += 1
                real_id_nns = dict_map_id[id_nns]
                if real_id_nns == real_id :
                    continue
                if stt>10:
                    break
                line = line + " " + real_id_nns
            file.writelines(line + "\n")


def save_file_to_hdfs(folderHDFSPath,fileName,fileLocalPath):
    logging.info("save file to hdfs")
    script_check_folder = "/opt/hadoop/bin/hadoop fs -test -d " + folderHDFSPath
    rs1 = subprocess.call(script_check_folder, shell=True)
    if rs1==1:
        script_make_folder = "/opt/hadoop/bin/hadoop fs -mkdir " + folderHDFSPath
        subprocess.call(script_make_folder, shell=True)

    script_check_file = "/opt/hadoop/bin/hadoop fs -test -e " + folderHDFSPath + "/" + fileName
    rs2 = subprocess.call(script_check_file, shell=True)
    if rs2==1:
        script_make_file = "/opt/hadoop/bin/hadoop  fs -copyFromLocal " + fileLocalPath + " " + folderHDFSPath + "/" + fileName
        subprocess.call(script_make_file, shell=True)
    elif rs2==0:
        script_drop_file = "/opt/hadoop/bin/hadoop fs -rm " + folderHDFSPath + "/" + fileName
        subprocess.call(script_drop_file, shell=True)
        script_make_file = "/opt/hadoop/bin/hadoop  fs -copyFromLocal " + fileLocalPath + " " + folderHDFSPath + "/" + fileName
        subprocess.call(script_make_file, shell=True)


