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
  def __init__(self,n_in,margin_in,rate_in,reg_param_in):
    self.n = n_in
    self.margin = margin_in
    self.rate = rate_in
    self.reg_param = reg_param_in
    self.loss = 0
    self.train_num = data.train_matrix.shape[0]
    self.test_num = data.test_matrix.shape[0]
    self.user_num = len(data.userid2seq)
    self.item_num = len(data.itemid2seq)
    self.relation_num = len(data.relation2seq)/2
    self.user_vec = np.random.uniform(-6/math.sqrt(self.n),6/math.sqrt(self.n),(self.user_num,self.n))
    self.item_vec = np.random.uniform(-6/math.sqrt(self.n),6/math.sqrt(self.n),(self.item_num,self.n))
    self.relation_vec = np.random.uniform(-6/math.sqrt(self.n),6/math.sqrt(self.n),(self.relation_num,self.n))
    self.graident_function = self.graident()
    self.loss = self.loss_init()

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
    rating_precision = ['rating '+str(i)+' precision' for i in range(1,6)]
    rating_recall = ['rating '+str(i)+' recall' for i in range(1,6)]
    rating_f = ['rating '+str(i)+' f' for i in range(1,6)]
    res_log = [['loss','average precision']+rating_precision+rating_recall+rating_f]
    avg_predict_init,r_precision_init,r_recall_init,r_fvalue_init = self.predict()
    res_log.append([self.loss]+avg_predict_init+r_precision_init+r_recall_init+r_fvalue_init)
    print('time:'+str(datetime.now())+' epoch:'+str(0)+' loss:'+str(self.loss)+' precision:'+str(avg_predict_init))
    print('hit rating ratio(from 1 to 5):'+str(r_fvalue_init))
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
      avg_precision,r_precision,r_recall,r_fvalue = self.predict()
      print('time:'+str(datetime.now())+' epoch:'+str(epoch+1)+' loss:'+str(self.loss)+' average precision:'+str(avg_precision))
      print('fvalue(rating from 1 to 5):'+str(r_fvalue))
      res_log.append([self.loss]+avg_precision+r_precision+r_recall+r_fvalue)
    with open(path,'w') as f:
      a = csv.writer(f,delimiter=',')
      a.writerows(res_log)

  def predict(self,n=1):
    precision = np.array([0]*n,dtype='double')
    hit_relations = [0]*5
    test_relations = [0]*5
    predict_dis = [0]*5
    hit = 0
    for i in range(self.test_num):
      test_tuple = data.test_matrix[i,:]
      user = data.userid2seq[test_tuple[0]]
      item = data.itemid2seq[test_tuple[1]]
      relation = data.relation2seq[test_tuple[2]]
      test_relations[relation] += 1
      for top in range(n):
        rels = self.res_relations(user,item,top+1)
        predict_dis[rels[0]] += 1
        if relation in rels:
          hit += 1
          hit_relations[relation] += 1
          precision[top] += 1
    precision /= self.test_num
    hit_relation_recall = [float(r[0])/r[1] for r in zip(hit_relations,test_relations)]
    hit_relation_precision = [float(r[0])/(r[1]) for r in zip(hit_relations,predict_dis)]
    hit_relation_f = [r[0]*r[1] for r in zip(hit_relation_precision,hit_relation_recall)]
    # hit_relation_precision = [float(r)/hit for r in hit_relations]
    return precision.tolist(),hit_relation_precision,hit_relation_recall,hit_relation_f

  def res_relations(self,user,item,top_n):
    user_vec = self.user_vec[user,:]
    item_vec = self.item_vec[item,:]
    sub_relation = {}
    for r in range(self.relation_num):
      sub = np.linalg.norm(user_vec+self.relation_vec[r,:]-item_vec)
      sub_relation[sub] = r
    sort_keys = [k for k in sub_relation.keys()]
    sort_keys.sort()
    rels = [sub_relation[r] for r in sort_keys[:top_n]]
    return rels

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
    vec_norm = np.linalg.norm(self.user_vec[user,:]+self.relation_vec[relation,:]-self.item_vec[item,:])
    return vec_norm*vec_norm

  def graident(self):
    u = T.dvector('u')
    i = T.dvector('i')
    r = T.dvector('r')
    r1 = T.dvector('r1')
    # construct theano expression graph
    distance_part = T.sum((u+r-i)**2)+self.margin-T.sum((u+r1-i)**2)
    regularizatoin = self.reg_param*(T.sum(u**2)+T.sum(i**2)+T.sum(r**2)+T.sum(r1**2))
    loss = distance_part+regularizatoin
    gu,gi,gr,gr1 = T.grad(loss,[u,i,r,r1])
    dloss = theano.function(inputs=[u,i,r,r1],outputs=[gu,gi,gr,gr1])
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


  def SGD(self,p_user,p_item,p_relation,n_user,n_item,n_relation):
    dloss = self.graident_function
    p_user_vec = self.user_vec[p_user,:]
    p_item_vec = self.item_vec[p_item,:]
    p_relation_vec = self.relation_vec[p_relation,:]
    n_relation_vec = self.relation_vec[n_relation,:]
    dp_user,dp_item,dp_relation,dn_relation = dloss(p_user_vec,p_item_vec,p_relation_vec,n_relation_vec)
    dp_relation -= self.relation_part_g(p_relation)
    dn_relation -= self.relation_part_g(n_relation)
    self.user_vec[p_user,:] -= self.rate*dp_user
    self.item_vec[p_item,:] -= self.rate*dp_item
    self.relation_vec[p_relation,:] -= self.rate*dp_relation
    self.relation_vec[n_relation,:] -= self.rate*dn_relation
    self.user_vec[p_user,:] /= np.linalg.norm(self.user_vec[p_user,:])
    self.item_vec[p_item,:] /= np.linalg.norm(self.item_vec[p_item,:])
    self.relation_vec[p_relation,:] /= np.linalg.norm(self.relation_vec[p_relation,:])
    self.relation_vec[n_relation,:] /= np.linalg.norm(self.relation_vec[n_relation,:])

if __name__ == "__main__":
  dem = [20,30]
  learning_rate = [0.05,0.01]
  for d in dem:
    for r in learning_rate:
      rr = Train(d,1,r,0.001)
      filename = str([d,r])+'.csv'
      print(filename)
      rr.run('result/TransE/f/'+filename)