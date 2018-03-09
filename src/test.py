import os
import sys
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


# print('Number of arguments:', len(sys.argv), 'arguments.')
# print('Argument List:', str(sys.argv))
# print(sys.argv[1])

import sys, getopt

def main(argv):
   inputfile = None
   outputfile = None
   try:
      opts, args = getopt.getopt(argv,"i:")
   except getopt.GetoptError:
      print("error")
      print('test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print('Input file is "', inputfile)
   print('Output file is "', outputfile)

if __name__ == "__main__":
   main(sys.argv[1:])
