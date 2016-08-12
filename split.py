from random import randint
import random
import pdb

data_list = []
with open('data/ratings.dat','r') as f:
  for row in f:
    row = row.strip()
    data_list.append(row.split('|'))
print('data_list complete')

train_data = []
first_add_count = 0
remain_seq = set()
has_add = set()
movie_set = set(); user_set = set()
whole_seq = set([i for i in range(len(data_list))])
for i in range(len(data_list)):
  example = data_list[i]
  user = example[0]
  movie = example[1]
  is_add = False
  if user not in user_set:
    movie_set.add(movie)
    user_set.add(user)
    train_data.append(example)
    has_add.add(i)
    is_add = True
  if movie not in movie_set and not is_add:
    user_set.add(user)
    movie_set.add(movie)
    train_data.append(example)
    has_add.add(i)
print([len(movie_set),len(user_set)])
remain_seq = [i for i in whole_seq if i not in has_add]
print(len(remain_seq))

train_num = int(len(data_list)*0.8)
remain_train_num = train_num - len(train_data)
remain_train_list = set()
print('remain train num:'+str(remain_train_num))
if remain_train_num>0:
  remain_random_seq = random.sample(range(len(remain_seq)), remain_train_num)

  for s in remain_random_seq:
    remain_train_list.add(remain_seq[s])
  print(len(remain_train_list))
else:
  print('first_add_count is too many!')


for i in remain_train_list:
  train_data.append(data_list[i])
print('train_data complete')

test_seq = set()
for i in remain_seq:
  if i not in remain_train_list:
    test_seq.add(i)
print('test_data num is '+str(len(test_seq)))

test_data = []
for i in test_seq:
  test_data.append(data_list[i])
print('test_data complete')
print('total:'+str(len(data_list)))
print('train_data:'+str(len(train_data)))
print('test_data:'+str(len(test_data)))

def write_file(path,write_list):
  with open(path,'w') as f:
    for d in write_list:
      string = '|'.join(d)
      f.write(string+'\n')

write_file('data/r1.test',test_data)
write_file('data/r1.train',train_data)