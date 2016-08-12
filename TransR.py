import load_data
import math
import numpy as np
import theano
import theano.tensor as T
import pdb
from datetime import *

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
    self.loss = 0
    self.train_num = data.train_matrix.shape[0]
    self.test_num = data.test_matrix.shape[0]
    self.user_num = len(data.userid2seq)
    self.item_num = len(data.itemid2seq)
    self.relation_num = len(data.relation2seq)/2
    self.user_vec = np.random.uniform(-6/math.sqrt(self.n),6/math.sqrt(self.n),(self.user_num,self.n))
    self.item_vec = np.random.uniform(-6/math.sqrt(self.m),6/math.sqrt(self.m),(self.item_num,self.m))
    self.relation_vec = np.random.uniform(-6/math.sqrt(self.k),6/math.sqrt(self.k),(self.relation_num,self.k))
    self.user_mapping_tensor = self.generate_eye_tensor(self.n,self.k)
    self.item_mapping_tensor = self.generate_eye_tensor(self.m,self.k)
    self.graident_function = self.graident()

  def generate_eye_tensor(self,n,k):
    tensor = np.ones((self.relation_num,n,k))
    for i in range(self.relation_num):
      tensor[i][:][:] = np.eye(n,k)
    return tensor

  def run(self):
    nepoch = 1000
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
      precision = self.predict()
      print('time:'+str(datetime.now())+' epoch:'+str(epoch)+' loss:'+str(self.loss)+' precision:'+str(precision.tolist()))

  def predict(self):
    precision = np.array([0]*3,dtype='double')
    for i in range(self.test_num):
      test_tuple = data.test_matrix[i,:]
      user = data.userid2seq[test_tuple[0]]
      item = data.itemid2seq[test_tuple[1]]
      relation = data.relation2seq[test_tuple[2]]
      for top in range(3):
        rels = self.res_relations(user,item,top+1)
        if relation in rels:
          precision[top] += 1
    precision /= self.test_num
    return precision

  def res_relations(self,user,item,top_n):
    sub_relation = {}
    for r in range(self.relation_num):
      sub = self.cal_distance(user,item,r)
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
    user_vec = self.user_vec[user,:]
    item_vec = self.item_vec[item,:]
    relation_vec = self.relation_vec[relation,:]
    user_mat = self.user_mapping_tensor[relation,:,:]
    item_mat = self.item_mapping_tensor[relation,:,:]
    vec_norm = np.linalg.norm(user_vec.dot(user_mat)+relation_vec-item_vec.dot(item_mat))
    return vec_norm**2

  def graident(self):
    u = T.dvector('u')
    i = T.dvector('i')
    r = T.dvector('r')
    r1 = T.dvector('r1')
    u_rmap = T.dmatrix('u_rmap')
    i_rmap = T.dmatrix('i_rmap')
    u_r1map = T.dmatrix('u_r1map')
    i_r1map = T.dmatrix('i_r1map')
    # construct theano expression graph
    distance_part = T.sum((T.dot(u,u_rmap)+r-T.dot(i,i_rmap))**2)+self.margin-T.sum((T.dot(u,u_r1map)+r1-T.dot(i,i_r1map))**2)
    regularizatoin = self.reg_param*(T.sum(u**2)+T.sum(i**2)+T.sum(r**2)+T.sum(r1**2)+T.sum(u_rmap**2)+T.sum(i_rmap**2)+T.sum(u_r1map**2)+T.sum(i_r1map**2))
    loss = distance_part+regularizatoin
    gu,gi,gr,gr1,gu_rmap,gi_rmap,gu_r1map,gi_r1map = T.grad(loss,[u,i,r,r1,u_rmap,i_rmap,u_r1map,i_r1map])
    dloss = theano.function(inputs=[u,i,r,r1,u_rmap,i_rmap,u_r1map,i_r1map],outputs=[gu,gi,gr,gr1,gu_rmap,gi_rmap,gu_r1map,gi_r1map])
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
    p_user_mat = self.user_mapping_tensor[p_relation,:,:]
    n_user_mat = self.user_mapping_tensor[n_relation,:,:]
    p_item_mat = self.item_mapping_tensor[p_relation,:,:]
    n_item_mat = self.item_mapping_tensor[n_relation,:,:]
    dp_user,dp_item,dp_relation,dn_relation,dp_user_mat,dp_item_mat,dn_user_mat,dn_item_mat = dloss(p_user_vec,p_item_vec,p_relation_vec,n_relation_vec,p_user_mat,p_item_mat,n_user_mat,n_item_mat)
    dp_relation -= self.relation_part_g(p_relation)
    dn_relation -= self.relation_part_g(n_relation)
    self.user_vec[p_user,:] -= self.rate*dp_user
    self.item_vec[p_item,:] -= self.rate*dp_item
    self.relation_vec[p_relation,:] -= self.rate*dp_relation
    self.relation_vec[n_relation,:] -= self.rate*dn_relation
    self.user_mapping_tensor[p_relation,:,:] -= self.rate*dp_user_mat
    self.user_mapping_tensor[n_relation,:,:] -= self.rate*dn_user_mat
    self.item_mapping_tensor[p_relation,:,:] -= self.rate*dp_item_mat
    self.item_mapping_tensor[n_relation,:,:] -= self.rate*dn_item_mat
    self.user_vec[p_user,:] /= np.linalg.norm(self.user_vec[p_user,:])
    self.item_vec[p_item,:] /= np.linalg.norm(self.item_vec[p_item,:])
    self.relation_vec[p_relation,:] /= np.linalg.norm(self.relation_vec[p_relation,:])
    self.relation_vec[n_relation,:] /= np.linalg.norm(self.relation_vec[n_relation,:])
    self.user_mapping_tensor[p_relation,:,:] /= np.linalg.norm(self.user_mapping_tensor[p_relation,:,:])
    self.user_mapping_tensor[n_relation,:,:] /= np.linalg.norm(self.user_mapping_tensor[n_relation,:,:])
    self.item_mapping_tensor[p_relation,:,:] /= np.linalg.norm(self.item_mapping_tensor[p_relation,:,:])
    self.item_mapping_tensor[n_relation,:,:] /= np.linalg.norm(self.item_mapping_tensor[n_relation,:,:])

if __name__ == "__main__":
  rr = Train(30,20,15,1,0.005,0.001)
  rr.run()