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

  def train_user_items(self):
    train_list = self.train_matrix.tolist()
    return self.build_dic(train_list)

  def test_user_items(self):
    test_list = self.test_matrix.tolist()
    return self.build_dic(test_list)

  def build_dic(self,data_list):
    dic = {}
    for record in data_list:
      user,item,rating = record[:3]
      if self.userid2seq[user] not in dic:
        # dic[self.userid2seq(user)] = [(self.itemid2seq[item],self.relation2seq[rating])]
        dic[self.userid2seq[user]] = [self.itemid2seq[item]]
      else:
        # dic[self.userid2seq(user)].append((self.itemid2seq[item],self.relation2seq[rating]))
        dic[self.userid2seq[user]].append(self.itemid2seq[item])
    return dic

# statistic the dataset
class Stat(object):
  """docstring for Stat"""
  def __init__(self, arg):
    self.data = Data()
    
  def count(self,matrix):
    num = matrix.shape[0]
    user_set = set()
    item_set = set()
    for i in range(num):
      record = matrix[i,:]
      user = self.data.userid2seq[record[0]]
      item = self.data.itemid2seq[record[1]]
      relation = self.data.relation2seq[record[2]]
      if user not in user_set: user_set.add(user)
      if item not in item_set: item_set.add(item)
    return len(user_set),len(item_set)

  def rating_distribution(self,rating_list):
    r_dis = [0]*5
    for r in rating_list:
      seq = self.data.relation2seq[r]
      r_dis[seq] += 1
    return r_dis

  def train_info(self):
    m = self.data.train_matrix
    user_num,item_num = self.count(m)
    rating_list = m[:,1]
    rating_dis = self.rating_distribution(rating_list)
    
