import os
import logging
from annoy import AnnoyIndex

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# path = "../data3"
# if os.path.exists(path):
#     pass
# else :
#     os.mkdir(path=path)
# a = []
# dict = [1,2,3,4,9]
# dict_id = [1,2,3,4,5]
# for i in range(len(dict)):
#     a.append((dict_id[i],dict[i]))
# print(a)from annoy import AnnoyIndex
# import random
#
# f = 40
# t = AnnoyIndex(f)  # Length of item vector that will be indexed
# for i in range(1000):
#     v = [random.gauss(0, 1) for z in range(f)]
#     t.add_item(i, v)
#
# t.build(10) # 10 trees
# t.save('test.ann')
#
# # ...
#
# u = AnnoyIndex(f)
# u.load('test.ann') # super fast, will just mmap the file
# print(u.get_nns_by_item(3, 10)) # will find the 1000 nearest neighbors
