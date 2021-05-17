from tsnecuda import TSNE
import numpy as np
from categorical_scatter import categorical_scatter_2d

import argparse

# import sys
# import sklearn.manifold as manifold
# from sklearn.decomposition import PCA as PCA
# sys.path.insert(0, './bhtsne')
# import bhtsne


parser = argparse.ArgumentParser(description='arg parser')

parser.add_argument('--out_dir', type=str, default='/home/wzha8158/Lidar_Outputs/(78.9)25-10-2020_second_da_nounet_bost2sing_novelo_PMA-0.1lr0.5_RCD-0.1lr0.5_FPN_Up1Down1_NoShare_Large_nusc_4GPU_8Batch_50Ep_Falsepseudo_0.5ratio_False2Test_0.25SELECT_PROP_10L_POW_Falsemcd_curve_1.0L_Ratio/eval/eval_last/', help='specify the config for training')

args = parser.parse_args()

X_tsne_ST_x = np.load(args.out_dir+'/Draw_set_ST_x.npy').squeeze()
X_tsne_ST_fpn3 = np.load(args.out_dir+"/Draw_set_ST_x_fpn3.npy").squeeze()

X_tsne_ST_fpn4 = np.load(args.out_dir+"/Draw_set_ST_x_fpn4.npy").squeeze()
X_tsne_ST_fpn5 = np.load(args.out_dir+"/Draw_set_ST_x_fpn5.npy").squeeze()

# X_tsne_S_Ori4 = np.load(args.out_dir+"/Draw_set_S_x_ori5.npy").squeeze()

# X_tsne_T_Ori4 = np.load(args.out_dir+"/Draw_set_T_x_ori5.npy").squeeze()


# print("X_tsne_ST_x", X_tsne_ST_x.shape)
# print("X_tsne_S_Ori4", X_tsne_S_Ori4.shape)

# X_tsne_ST_x = np.load("Draw_set_ST_near_x.npy")
# X_tsne_ST_fpn3 = np.load("Draw_set_ST_far_x.npy")

# X_tsne_ST_fpn4 = np.load("Draw_set_ST_near_y.npy")
# X_tsne_ST_fpn5 = np.load("Draw_set_ST_far_y.npy")

# X_tsne_S_Ori4 = np.load("Draw_set_S_nearfar_x.npy")
# X_tsne_S_nearfar_y = np.load("Draw_set_S_nearfar_y.npy")

# X_tsne_T_Ori4 = np.load("Draw_set_T_nearfar_x.npy")
# X_tsne_T_nearfar_y = np.load("Draw_set_T_nearfar_y.npy")

#################

# X_tsne_ST_x_bhtsne = bhtsne.run_bh_tsne(X_tsne_ST_x, randseed=0, initial_dims=X_tsne_ST_x.shape[1])
# X_tsne_ST_fpn3_bhtsne = bhtsne.run_bh_tsne(X_tsne_ST_fpn3, randseed=0, initial_dims=X_tsne_ST_fpn3.shape[1])

# X_tsne_ST_fpn4_bhtsne = bhtsne.run_bh_tsne(X_tsne_ST_fpn4, randseed=0, initial_dims=X_tsne_ST_fpn4.shape[1])
# X_tsne_ST_fpn5_bhtsne = bhtsne.run_bh_tsne(X_tsne_ST_fpn5, randseed=0, initial_dims=X_tsne_ST_fpn5.shape[1])

# X_tsne_S_Ori4_bhtsne = bhtsne.run_bh_tsne(X_tsne_S_Ori4, randseed=0, initial_dims=X_tsne_S_Ori4.shape[1])
# # X_tsne_S_nearfar_y_bhtsne = bhtsne.run_bh_tsne(X_tsne_S_nearfar_y, randseed=0, initial_dims=X_tsne_S_nearfar_y.shape[1])

# X_tsne_T_Ori4_bhtsne = bhtsne.run_bh_tsne(X_tsne_T_Ori4, randseed=0, initial_dims=X_tsne_T_Ori4.shape[1])
# # X_tsne_T_nearfar_y_bhtsne = bhtsne.run_bh_tsne(X_tsne_T_nearfar_y, randseed=0, initial_dims=X_tsne_T_nearfar_y.shape[1])

# print("finish bhtsne")

#############
X_tsne_ST_x = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X_tsne_ST_x.astype(np.float))
X_tsne_ST_fpn3 = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X_tsne_ST_fpn3.astype(np.float))
X_tsne_ST_fpn4 = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X_tsne_ST_fpn4.astype(np.float))
X_tsne_ST_fpn5 = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X_tsne_ST_fpn5.astype(np.float))

# X_tsne_S_Ori4 = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X_tsne_S_Ori4.astype(np.float))
# # X_tsne_S_nearfar_y = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X_tsne_S_nearfar_y.astype(np.float))

# X_tsne_T_Ori4 = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X_tsne_T_Ori4.astype(np.float))
# X_tsne_T_nearfar_y = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X_tsne_T_nearfar_y.astype(np.float))

# Y_ST = 1545*[0]+1465*[1]
# YS_x = 1545*[0]+1545*[1]
# YT_x = 1465*[0]+1465*[1]

Y_ST = 9393*[0]+7682*[1]
YS_x = 9393*[0]+9393*[1]
YT_x = 7682*[0]+7682*[1]

# 18785 15364


epoch_id = 0

categorical_scatter_2d(X_tsne_ST_x, Y_ST, alpha=1.0, ms=6,
                                show=False, figsize=(9, 6), savename=args.out_dir+f'/X_tsne_ST_x_{epoch_id}.png')
categorical_scatter_2d(X_tsne_ST_fpn3, Y_ST, alpha=1.0, ms=6,
                                show=False, figsize=(9, 6), savename=args.out_dir+f'/X_tsne_ST_fpn3_{epoch_id}.png')
categorical_scatter_2d(X_tsne_ST_fpn4, Y_ST, alpha=1.0, ms=6,
                        show=False, figsize=(9, 6), savename=args.out_dir+f'/X_tsne_ST_fpn4_{epoch_id}.png')
categorical_scatter_2d(X_tsne_ST_fpn5, Y_ST, alpha=1.0, ms=6,
                        show=False, figsize=(9, 6), savename=args.out_dir+f'/X_tsne_ST_fpn5_{epoch_id}.png')

# categorical_scatter_2d(X_tsne_S_Ori4, YS_x, alpha=1.0, ms=6,
#                         show=False, figsize=(9, 6), savename=args.out_dir+f'/X_tsne_S_Ori4_{epoch_id}.png')
# categorical_scatter_2d(X_tsne_T_Ori4, YT_x, alpha=1.0, ms=6,
#                         show=False, figsize=(9, 6), savename=args.out_dir+f'/X_tsne_T_Ori4_{epoch_id}.png')

#####################
# categorical_scatter_2d(X_tsne_ST_x_bhtsne, Y_ST, alpha=1.0, ms=6,
#                                 show=False, figsize=(9, 6), savename=args.out_dir+f'/X_tsne_ST_x_{epoch_id}_bhtsne.png')
# categorical_scatter_2d(X_tsne_ST_fpn3_bhtsne, Y_ST, alpha=1.0, ms=6,
#                                 show=False, figsize=(9, 6), savename=args.out_dir+f'/X_tsne_ST_fpn3_{epoch_id}_bhtsne.png')
# categorical_scatter_2d(X_tsne_ST_fpn4_bhtsne, Y_ST, alpha=1.0, ms=6,
#                         show=False, figsize=(9, 6), savename=args.out_dir+f'/X_tsne_ST_fpn4_{epoch_id}_bhtsne.png')
# categorical_scatter_2d(X_tsne_ST_fpn5_bhtsne, Y_ST, alpha=1.0, ms=6,
#                         show=False, figsize=(9, 6), savename=args.out_dir+f'/X_tsne_ST_fpn5_{epoch_id}_bhtsne.png')

# categorical_scatter_2d(X_tsne_S_Ori4_bhtsne, YS_nearfar, alpha=1.0, ms=6,
#                         show=False, figsize=(9, 6), savename=args.out_dir+f'/X_tsne_S_Ori4_{epoch_id}_bhtsne.png')
# # categorical_scatter_2d(X_tsne_S_nearfar_y_bhtsne, YS_nearfar, alpha=1.0, ms=6,
#                         # show=False, figsize=(9, 6), savename=args.out_dir+f'/X_tsne_S_nearfar_y_{epoch_id}_bhtsne.png')
# categorical_scatter_2d(X_tsne_T_Ori4_bhtsne, YT_nearfar, alpha=1.0, ms=6,
#                         show=False, figsize=(9, 6), savename=args.out_dir+f'/X_tsne_T_Ori4_{epoch_id}_bhtsne.png')
# # categorical_scatter_2d(X_tsne_T_nearfar_y_bhtsne, YT_nearfar, alpha=1.0, ms=6,
#                         # show=False, figsize=(9, 6), savename=args.out_dir+f'/X_tsne_T_nearfar_y_{epoch_id}_bhtsne.png')
