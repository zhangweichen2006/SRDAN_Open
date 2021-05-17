id = torch.tensor([0,1,2,3,0,1,6,9])
#array([0, 4, 1, 5, 2, 3, 6, 7]) perm
#aa = np.argsort(aa) perm.argsort()
[04152367]
[004152367]
[00112367]

[04]
[004152367]
[00112367]

#new_perm = zeros(8) [0,0,1,1,2,3,6,7]
for i in range(8):
  if flag[i]:
    new_id = perm[i]
  new_perm[i] = new_id
new_perm[perm.argsort()] # 0,1,2,3,0,1,6,7 
final_perm = new_perm[perm.argsort()]


[4., 1., 2., 3., 0., 4., 3., 1., 6., 9.]
0, 1,2,3,4,0,3,1,8,9
rray([0., 1., 2., 3., 4., 0., 3., 1., 8., 9.])


use high res feat


indices_index_joint = torch.tensor([9,1,6,3,3,2,0,4,2,6,9]).cuda()

# [ 0,  1,  7,  3,  3,  5,  6,  7,  5,  9, 10]
# [ 0,  1,  9,  3,  3,  5,  6,  7,  5,  9, 10]
# [ 0, 1, 9, 3, 3, 5, 6, 7, 5, 9, 0]

# torch.cat((indices_index, indices_index2)) # 405314, 60000
# indices_index_len = len(indices_index_joint)
# print("indices_index_joint", indices_index_joint.shape) # 463646

# unique_index, unique_index_inds = torch.unique(indices_index_joint, return_inverse=True) # 465314

perm = indices_index_joint.argsort()
# perm = torch.argsort(ab)
aux = indices_index_joint[perm]

rightperm = torch.cat((perm[0].unsqueeze(-1),perm))[:-1]

flag = torch.cat((torch.tensor([True]).cuda(), aux[1:] != aux[:-1]))

new_perm = torch.where(flag==1,perm,rightperm)

new_indices_index = new_perm[perm.argsort()] 
# print("inds", indices_index.shape, indices_index)
# print("inds2", indices_index2.shape, indices_index2)
print("new_indices_index", new_indices_index.shape, new_indices_index)

# print("unique_index", unique_index.shape, unique_index)
# print("unique_index_inds", unique_index_inds.shape, unique_index_inds)
