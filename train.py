from utils import *
from tqdm import tqdm
from torch import optim
from model import my_model
from model import ContrastiveLoss
import torch.nn.functional as F
import numpy as np


for args.dataset in ["amap"]:
    print("Using {} dataset".format(args.dataset))
    file = open("result_baseline.csv", "a+")
    print(args.dataset, file=file)
    file.close()

    if args.dataset == 'cora':
        args.cluster_num = 7
        args.gnnlayers_l = 3
        args.gnnlayers_h = 7
        args.sigma_X = 0.01
        args.sigma = 0.001
        args.gama = 0.5
        args.lr = 1e-3
        args.dims = [1300]
    elif args.dataset == 'citeseer':
        args.cluster_num = 6
        args.gnnlayers_l = 5
        args.gnnlayers_h = 2
        args.sigma_X = 0.01
        args.sigma = 0.1
        args.gama = 0.01
        args.lr = 2e-4
        args.dims = [1000]
    elif args.dataset == 'amap':
        args.cluster_num = 8
        args.gnnlayers_l = 7
        args.gnnlayers_h = 4
        args.sigma_X = 0.0001
        args.sigma = 0.1
        args.gama = 0.5
        args.lr = 2e-3
        args.dims = [800]
    elif args.dataset == 'bat':
        args.cluster_num = 4
        args.gnnlayers_l = 60
        args.gnnlayers_h = 8
        args.sigma_X = 0.01
        args.sigma = 0.001
        args.gama = 0.5
        args.lr = 5e-3
        args.dims = [800]
    elif args.dataset == 'eat':
        args.cluster_num = 4
        args.gnnlayers_l = 25
        args.gnnlayers_h = 4
        args.sigma_X = 0.001
        args.sigma = 0.1
        args.gama = 0.5
        args.lr = 1e-3
        args.dims = [1000]
    elif args.dataset == 'uat':
        args.cluster_num = 4
        args.gnnlayers_l = 1
        args.gnnlayers_h = 2
        args.sigma_X = 0.01
        args.sigma = 0.1
        args.gama = 1
        args.lr = 1e-3
        args.dims = [600]
    elif args.dataset == 'others':
        args.cluster_num = 7
        args.gnnlayers_l = 3
        args.gnnlayers_h = 2
        args.sigma_X = 0.01
        args.sigma = 0.1
        args.gama = 0.5
        args.lr = 1e-3
        args.dims = [800]

    # load data
    X, y, A, node_num = load_graph_data(args.dataset, show_details=False)
    features = X
    true_labels = y
    adj = sp.csr_matrix(A)
    sm_fea_s_i = sp.csr_matrix(features).toarray()
    sm_fea_s_i = torch.FloatTensor(sm_fea_s_i)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    print('Laplacian Smoothing...')
    sm_fea_s_list = preprocess_graph(features, adj, args.gnnlayers_l, args.gnnlayers_h, norm='sym', renorm=True)
    adj_1st = (adj + sp.eye(adj.shape[0])).toarray()

    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []

    for seed in range(10):
        setup_seed(seed)
        model = my_model([features.shape[1]] + args.dims)
        sm_fea_s_noise = add_gaussian_noise(features, args.sigma_X)
        inx_lh = calculate_weighted_features_z(model.alpha, model.beta, sm_fea_s_list, sm_fea_s_noise)
        best_acc, best_nmi, best_ari, best_f1, prediect_labels = clustering(inx_lh, true_labels, args.cluster_num)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = model.to(args.device)
        sm_fea_s_noise = sm_fea_s_noise.to(args.device)
        inx_lh = inx_lh.to(args.device)
        target = torch.FloatTensor(adj_1st).to(args.device)
        cc_loss_fn = ContrastiveLoss()

        print('Start Training...')
        for epoch in tqdm(range(args.epochs)):
            model.train()
            inx_lh = calculate_weighted_features_z(model.alpha, model.beta, sm_fea_s_list, sm_fea_s_noise)
            z1, z2, z2_s = model(inx_lh, is_train=True)
            z = (z1 + z2) / 2

            A_r = (z1 @ z2.T) / 2
            c_loss = F.mse_loss(A_r, target)

            cc_loss = cc_loss_fn(z1, z2, z2_s)

            loss = args.gama * c_loss + cc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 4 == 0:
                model.eval()
                z1, z2, _ = model(inx_lh, is_train=True)
                hidden_emb = (z1 + z2) / 2
                acc, nmi, ari, f1, predict_labels = clustering(hidden_emb, true_labels, args.cluster_num)
                if acc >= best_acc:
                    best_acc = acc
                    best_nmi = nmi
                    best_ari = ari
                    best_f1 = f1

        tqdm.write('acc: {}, nmi: {}, ari: {}, f1: {}'.format(best_acc, best_nmi, best_ari, best_f1))
        file = open("result_baseline.csv", "a+")
        print(best_acc, best_nmi, best_ari, best_f1, file=file)
        file.close()
        acc_list.append(best_acc)
        nmi_list.append(best_nmi)
        ari_list.append(best_ari)
        f1_list.append(best_f1)

    acc_list = np.array(acc_list)
    nmi_list = np.array(nmi_list)
    ari_list = np.array(ari_list)
    f1_list = np.array(f1_list)
    file = open("result_baseline.csv", "a+")
    print(args.gnnlayers_l, args.gnnlayers_h, args.lr, args.dims, args.sigma_X, args.sigma, args.gama, file=file)
    print(round(acc_list.mean(), 2), round(acc_list.std(), 2), file=file)
    print(round(nmi_list.mean(), 2), round(nmi_list.std(), 2), file=file)
    print(round(ari_list.mean(), 2), round(ari_list.std(), 2), file=file)
    print(round(f1_list.mean(), 2), round(f1_list.std(), 2), file=file)
    file.close()