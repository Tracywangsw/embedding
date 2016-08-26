import load_data
import math
import numpy as np
import theano
import theano.tensor as T
import pdb
from datetime import *
import csv

data = load_data.Data('data/r1.train','data/r1.test')

class Train(object):
  """docstring for ClassName"""
  def __init__(self,user_d,item_d,relation_d,margin_in,rate_in,reg_param_in):
    self.n = user_d
    self.m = item_d
    self.k = relation_d
    self.margin = margin_in
    self.rate = rate_in
    self.reg_param = reg_param_in
    self.train_num = data.train_matrix.shape[0]
    self.test_num = data.test_matrix.shape[0]
    self.user_num = len(data.userid2seq)
    self.item_num = len(data.itemid2seq)
    self.relation_num = len(data.relation2seq)/2
    self.user_vec = theano.shared(np.asarray(np.random.uniform(-6/math.sqrt(self.n),6/math.sqrt(self.n),(self.user_num,self.n)),dtype='float32'),name='user')
    self.item_vec = theano.shared(np.asarray(np.random.uniform(-6/math.sqrt(self.m),6/math.sqrt(self.m),(self.item_num,self.m)),dtype='float32'),name='item')
    self.relation_vec = theano.shared(np.asarray(np.random.uniform(-6/math.sqrt(self.k),6/math.sqrt(self.k),(self.relation_num,self.k)),dtype='float32'),name='relation')
    self.relatioin_mapping_matrix = theano.shared(np.asarray(self.generate_mapping_matrix(self.n),dtype='float32'),name='map_matrix')
    self.graident_function = self.graident()
    self.loss = self.loss_init()

  def generate_mapping_matrix(self,n):
    m = np.random.rand(self.relation_num,n)
    for i in range(self.relation_num):
      m[i][:] /= np.linalg.norm(m[i][:])
    return m

  def loss_init(self):
    loss = 0
    for i in range(self.train_num):
      record = data.train_matrix[i,:]
      p_user = data.userid2seq[record[0]]
      p_item = data.itemid2seq[record[1]]
      p_relation = data.relation2seq[record[2]]
      n_user = p_user
      n_item = p_item
      n_relation = self.negative_sampling(p_relation)
      p_distance = self.cal_distance(p_user,p_item,p_relation)
      n_distance = self.cal_distance(n_user,n_item,n_relation)
      if p_distance+self.margin-n_distance>0:
        loss += p_distance+self.margin-n_distance
    loss /= self.train_num
    return loss

  def run(self,path):
    nepoch = 100
    predict_init,dis_init = self.predict()
    res_log = [[self.loss]+predict_init+dis_init]
    print('time:'+str(datetime.now())+' epoch:'+str(0)+' loss:'+str(self.loss)+' precision:'+str(predict_init))
    print('hit rating ratio(from 1 to 5):'+str(dis_init))
    for epoch in range(nepoch):
      self.loss = 0
      for i in range(self.train_num):
        record = data.train_matrix[i,:]
        p_user = data.userid2seq[record[0]]
        p_item = data.itemid2seq[record[1]]
        p_relation = data.relation2seq[record[2]]
        n_user = p_user
        n_item = p_item
        n_relation = self.negative_sampling(p_relation)
        p_distance = self.cal_distance(p_user,p_item,p_relation)
        n_distance = self.cal_distance(n_user,n_item,n_relation)
        if p_distance+self.margin-n_distance>0:
          self.loss += p_distance+self.margin-n_distance
          self.SGD(p_user,p_item,p_relation,n_user,n_item,n_relation)
      self.loss /= self.train_num
      precision,dis = self.predict()
      print('time:'+str(datetime.now())+' epoch:'+str(epoch+1)+' loss:'+str(self.loss)+' precision:'+str(precision))
      print('hit rating ratio(from 1 to 5):'+str(dis))
      res_log.append([self.loss]+precision+dis)
    with open(path,'w') as f:
      a = csv.writer(f,delimiter=',')
      a.writerows(res_log)

  def res_relations(self,user,item,top_n):
    sub_relation = {}
    for r in range(self.relation_num):
      sub = self.cal_distance(user,item,r)
      sub_relation[sub] = r
    sort_keys = [k for k in sub_relation.keys()]
    sort_keys.sort()
    rels = [sub_relation[r] for r in sort_keys[:top_n]]
    return rels

  def predict(self,n=1):
    precision = np.array([0]*n,dtype='double')
    hit_relations = [0]*5
    test_relations = [0]*5
    hit = 0
    for i in range(self.test_num):
      test_tuple = data.test_matrix[i,:]
      user = data.userid2seq[test_tuple[0]]
      item = data.itemid2seq[test_tuple[1]]
      relation = data.relation2seq[test_tuple[2]]
      test_relations[relation] += 1
      for top in range(n):
        rels = self.res_relations(user,item,top+1)
        if relation in rels:
          hit += 1
          hit_relations[relation] += 1
          precision[top] += 1
    precision /= self.test_num
    hit_relation_precision = [float(r[0])/r[1] for r in zip(hit_relations,test_relations)]
    # hit_relation_precision = [float(r)/hit for r in hit_relations]
    return precision.tolist(),hit_relation_precision

  def new_items(self,all,old):
    new = set()
    for i in all:
      if i not in old:
        new.add(i)
    return new

  def cal_preference(self,user,item,relation):
    user_vec = self.user_vec[user,:]
    item_vec = self.item_vec[item,:]
    relation_vec = self.relation_vec[relation,:]
    user_mat = self.user_mapping_tensor[relation,:,:]
    item_mat = self.item_mapping_tensor[relation,:,:]
    vec_norm = np.linalg.norm(user_vec.dot(user_mat)-item_vec.dot(item_mat))
    return vec_norm**2

  def top_item_recommend(self,top_n=5):
    precision = 0
    train_user_items = data.train_user_items()
    test_user_items = data.test_user_items()
    all_items = set([i for i in range(self.item_num)])
    test_user_count = 0
    for u in range(self.user_num):
      u_items = set(train_user_items[u])
      items = self.new_items(all_items,u_items)
      items_scores = []
      u_vec = self.user_vec[u,:][np.newaxis]
      for i in items:
        i_vec = self.item_vec[i,:][np.newaxis]
        closest_rating = self.res_relations(u,i,1)
        score = 1/(self.cal_preference(u,i,closest_rating)+1)
        items_scores.append((score,i))
      items_scores.sort(reverse=True)
      recommen_item = set([i[1] for i in items_scores[:top_n]])
      if u not in test_user_items: continue
      test_set = set(test_user_items[u])
      hit = len(test_set)-len(self.new_items(test_set,recommen_item))
      # pdb.set_trace()
      precision += float(hit)/top_n
      test_user_count += 1
    precision /= test_user_count
    return precision

  def negative_sampling(self,p_relation):
    if p_relation<0 or p_relation>4: print('relation is not in range')
    if p_relation == 4:
      n_relation = 0
    elif p_relation == 3:
      n_relation = 1
    elif p_relation == 2:
      n_relation = 0
    elif p_relation == 1:
      n_relation = 3
    else:
      n_relation = 4
    return n_relation

  def cal_distance(self,user,item,relation):
    user_vec = self.user_vec[user,:]
    item_vec = self.item_vec[item,:]
    relation_vec = self.relation_vec[relation,:]
    map_vec = self.relatioin_mapping_matrix[relation,:]
    user_rel = user_vec-(map_vec.T.dot(user_vec))*map_vec
    item_rel = item_vec-(map_vec.T.dot(item_vec))*map_vec
    vec_norm = np.linalg.norm(user_rel+relation_vec-item_rel)
    return vec_norm**2

  def graident(self):
    p_user = T.iscalar('p_user')
    p_item = T.iscalar('p_item')
    p_relation = T.iscalar('p_relation')
    n_relation = T.iscalar('n_relation')
    u_zero = np.zeros((1,self.user_num))
    i_zero = np.zeros((1,self.item_num))
    n_r_zero = np.zeros((1,self.relation_num))
    p_r_zero = np.zeros((1,self.relation_num))
    i_zero[p_item],u_zero[p_user],p_r_zero[p_relation],n_r_zero[n_relation] = [1]*4
    u = u_zero.dot(self.user_vec)
    i = i_zero.dot(self.item_vec)
    r = p_r_zero.dot(self.relation_vec)
    r1 = n_r_zero.dot(self.relation_vec)
    r_map = p_r_zero.dot(self.relatioin_mapping_matrix)
    r1_map = n_r_zero.dot(self.relatioin_mapping_matrix)
    # construct theano expression graph
    ur = u-T.dot(T.transpose(r_map),u)*r_map
    ur1 = u-T.dot(T.transpose(r1_map),u)*r1_map
    ir = i-T.dot(T.transpose(r_map),i)*r_map
    ir1 = i-T.dot(T.transpose(r1_map),i)*r1_map
    distance_part = T.sum((ur+r-ir)**2)+self.margin-T.sum((ur1+r1-ir1)**2)
    regularizatoin = self.reg_param*(T.sum(u**2)+T.sum(i**2)+T.sum(r**2)+T.sum(r1**2)+T.sum(r_map**2)+T.sum(r1_map**2))
    loss = distance_part+regularizatoin
    gu,gi,gr,gr1,gr_map,gr1_map = T.grad(loss,[self.user_vec,self.item_vec,self.relation_vec])
    updates = ((self.user_vec,self.user_vec-self.rate*gu),
              ())
    dloss = theano.function(inputs=[u,i,r,r1,r_map,r1_map,u_s,i_s,r_s,r1_s],outputs=[loss],updates=updates)
    return dloss

  def relation_part_g(self,relation,weight=1):
    drelation = 0
    for r in range(self.relation_num):
      if r == relation: continue
      r_vec = self.relation_vec[r,:]
      rel_vec = self.relation_vec[relation,:]
      x = 1
      sub = (rel_vec**2).sum()-(r_vec**2).sum()
      if r<relation:
        sub = -1*sub
        x = -1
      c = 1/(1+math.exp(-sub))
      dc = math.exp(-sub)/(1+math.exp(-sub))**2
      drelation += 2*weight*x/c*dc*(rel_vec**2).sum()
    return drelation

  def norm(self,v,m):
    v = v[np.newaxis]
    while True:
      vm = v.dot(m)
      n = (np.linalg.norm(vm))**2
      if n>1:
        temp = 2*vm
        m -= self.rate*v.T.dot(temp)
        # v -= self.rate*temp.dot(m.T)
      else:
        return m

  def SGD(self,p_user,p_item,p_relation,n_user,n_item,n_relation):
    u_zero = np.zeros((1,self.user_num))
    i_zero = np.zeros((1,self.item_num))
    n_r_zero = np.zeros((1,self.relation_num))
    p_r_zero = np.zeros((1,self.relation_num))
    i_zero[p_item],u_zero[p_user],p_r_zero[p_relation],n_r_zero[n_relation] = [1]*4
    p_user_v = u_zero.dot(self.user_vec)
    p_item_v = i_zero.dot(self.item_vec)
    p_relation_v = p_r_zero.dot(self.relation_vec)
    n_relation_v = n_r_zero.dot(self.relation_vec)



if __name__ == "__main__":
  user_dem = [[30,20,15],[20,20,20],[30,20,20],[20,20,30]]
  learning_rate = [0.01,0.005,0.001]
  for r in learning_rate:
    for d in user_dem:
      rr = Train(d[0],d[1],d[2],1,r,0.001)
      filename = str(d+[r])+'.csv'
      print(filename)
      rr.run('result/TransR/'+filename)