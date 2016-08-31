data_list = []
with open('ml-10M100K/ratings.dat','r') as f:
  for row in f:
    row = row.strip()
    data_list.append(row.split('|'))
print('data_list complete')

rating_dis = {}
user_set = set()
item_set = set()
for data in data_list:
  user = data[0]
  item = data[1]
  rating = data[2]
  if user not in user_set: user_set.add(user)
  if item not in item_set: item_set.add(item)
  if rating not in rating_dis: rating_dis[rating] = 1
  else:
    rating_dis[rating] += 1

rating_seq_arr = [0]*5
for i in range(len(rating_seq_arr)):
  j,k = i+0.5,i+1
  rating_seq_arr[i] = rating_dis[str(j)]+rating_dis[str(k)]

rating_seq_dis = [float(r)/sum(rating_seq_arr) for r in rating_seq_arr]
print('user number:'+str(len(user_set)))
print('item number:'+str(len(item_set)))
print('rating number:'+str(len(data_list)))
print('rating distribution:'+str(rating_seq_dis))




