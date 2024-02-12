import torch
import random
import numpy as np
from time import time
from parse import args
from data import load_dataset
from client import FedRecClient
from server import FedRecServer
from tqdm import tqdm

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class ClientConfig(object):
    def __init__(self, user_id, num_users, num_items, latent_dim, layers, device_id,
                 local_epoch, local_lr, local_bs, local_pc, model_type, use_privacy):
        # model settings
        self.user_id = user_id
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.layers = layers
        self.device_id = device_id
        self.model_type = model_type

        # train setting
        self.local_epoch = local_epoch
        self.lr = local_lr
        self.local_bs = local_bs
        self.local_pc = local_pc  # privacy control

        # privacy protection
        self.use_privacy = use_privacy

class ServerConfig(object):
    def __init__(self, num_users, num_items, latent_dim, layers, model_type, device_id,
                 global_lr, data_treat, server_bs, server_epoch, transmit_num,
                 lightgcn_layers, use_privacy):
        # model settings
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.layers = layers
        self.model_type = model_type
        self.device_id = device_id

        # train setting
        self.lr = global_lr
        self.data_treat = data_treat  # "conf" -> use confidence as label; "pos" -> treat as positive (1.0)
        self.server_bs = server_bs
        self.server_epoch = server_epoch
        self.transmit_num = transmit_num # e.g. 30

        self.lightGCN_n_layers = lightgcn_layers

        # privacy protection
        self.use_privacy = use_privacy



def main():
    args_str = ",".join([("%s=%s" % (k, v)) for k, v in args.__dict__.items()])
    print("Arguments: %s " % args_str)

    t0 = time()
    m_item, all_train_ind, all_test_ind, items_popularity, all_user = load_dataset(args.path + args.dataset)

    # setup server
    server_config = ServerConfig(len(all_user), m_item, args.server_dim,
                                 eval(args.server_layers),
                                 args.server_model_type, args.device,
                                 args.global_lr, args.data_treat, args.server_bs,
                                 args.server_epoch, args.transmit_num, args.lightgcn_layers, args.use_privacy)
    server = FedRecServer(server_config).to(args.device)

    clients = []
    # setup client
    for user_id, train_ind, test_ind in tqdm(zip(all_user, all_train_ind, all_test_ind)):
        client_config = ClientConfig(user_id, 1, m_item, args.client_dim,
                                     eval(args.user_layers), args.device,
                                     args.local_epoch, args.local_lr, args.local_bs,
                                     args.local_pc, args.local_model_type, args.use_privacy)
        # train_ind, test_ind, target_items, m_item, args.dim
        clients.append(FedRecClient(client_config, train_ind, test_ind).to(args.device))

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" %
          (time() - t0, len(clients), m_item,
           sum([len(i) for i in all_train_ind]),
           sum([len(i) for i in all_test_ind])))
    print("output format: (server: {Recall@20, NDCG@20}), {client: Recall@20, NDCG@20}), ensemble: {Recall@20, NDCG@20}))")

    for epoch in range(1, args.epochs + 1):
        t1 = time()
        rand_clients = np.arange(len(clients))
        np.random.shuffle(rand_clients)

        client_loss, server_loss = server.train_(clients, rand_clients, epoch)

        t2 = time()
        server_result, client_result, ensemble_result = server.eval_(clients)
        print("Iteration %d, server loss = %.5f, client loss = %.5f [%.1fs]" % (epoch, sum(server_loss) / len(server_loss), sum(client_loss) / len(client_loss), t2 - t1) +
              ", server: (%.7f, %.7f)" % tuple(server_result) +
              ", client: (%.7f, %.7f)" % tuple(client_result) +
              ", ensemble: (%.7f, %.7f)." % tuple(ensemble_result) +
              " [%.1fs]" % (time() - t2))


if __name__ == "__main__":
    setup_seed(42)
    main()
