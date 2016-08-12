import numpy as np

class Data(object):
  """docstring for Data"""
  def __init__(self, train_file,test_file):
    self.train_matrix = self.read_file(train_file)
    self.test_matrix = self.read_file(test_file)
    self.userid2seq = self.id2seq(0)
    self.itemid2seq = self.id2seq(1)
    self.relation2seq = self.set_relation()

  def set_relation(self):
    rel2seq = {}
    rel2seq['0.5'],rel2seq['1'] = [0,0]
    rel2seq['1.5'],rel2seq['2'] = [1,1]
    rel2seq['2.5'],rel2seq['3'] = [2,2]
    rel2seq['3.5'],rel2seq['4'] = [3,3]
    rel2seq['4.5'],rel2seq['5'] = [4,4]
    return rel2seq
    
  def read_file(self,path,delimiter='|'):
    data_list = []
    with open(path,'r') as f:
      for row in f:
        row = row.strip()
        row = row.split(delimiter)[:3]
        data_list.append(row)
    data_matrix = np.array(data_list)
    return data_matrix

  def id2seq(self,column):
    seq = 0
    id_set = set(self.train_matrix[:,column])
    id2seq_map = {}
    for id in id_set:
      id2seq_map[id] = seq
      seq += 1
    return id2seq_map
