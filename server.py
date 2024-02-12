import copy

import torch
import torch.nn as nn
from parse import args
import pickle
from sklearn import preprocessing
from tqdm import tqdm
from models import *
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from torch.utils.data import DataLoader, Dataset


class ServerDataset(Dataset):
    def __init__(self, users, items, labels):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        return self.users[item], self.items[item], self.labels[item]

class FedRecServer(nn.Module):
    def __init__(self, config):  # server config
        super().__init__()
        if config.model_type == "NCF":
            self.model = NCF(config).to(config.device_id)
        elif config.model_type == "NGCF":
            self.model = NGCF(config).to(config.device_id)
        elif config.model_type == "LightGCN":
            self.model = LightGCN(config).to(config.device_id)
        else:
            raise ImportError(f"the model type should be in [NCF, LightGCN, ...]")
        self.config = config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.criterion = torch.nn.BCELoss(reduction="sum")
        # self.criterion = torch.nn.BCELoss()
        self.train_frequency = torch.zeros(self.config.num_items)
        self.client_to_items = {}
        self.graph = None

    def _get_graph(self, userList, itemList, ratings):  # user, item list and corresponding ratings (edge weight)
        def _convert_sp_mat_to_sp_tensor(X):
            coo = X.tocoo().astype(np.float32)
            row = torch.Tensor(coo.row).long()
            col = torch.Tensor(coo.col).long()
            index = torch.stack([row, col])
            data = torch.FloatTensor(coo.data)
            return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        trainUser = np.array(userList)
        trainItem = np.array(itemList)
        adj_mat = sp.dok_matrix((self.config.num_users + self.config.num_items, self.config.num_users + self.config.num_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        UserItemNet = csr_matrix((np.array(ratings), (trainUser, trainItem)),
                                 shape=(self.config.num_users, self.config.num_items))
        R = UserItemNet.tolil()
        adj_mat[:self.config.num_users, self.config.num_users:] = R
        adj_mat[self.config.num_users:, :self.config.num_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        graph = _convert_sp_mat_to_sp_tensor(norm_adj)
        graph = graph.coalesce().to(self.config.device_id)
        return graph

    def train_single_batch(self, users, items, labels, graph=None):
        users, items, ratings = users.to(self.config.device_id), items.to(self.config.device_id), labels.to(self.config.device_id)
        self.optimizer.zero_grad()
        if graph is None:
            ratings_pred = self.model(users, items)
        else:
            ratings_pred = self.model(users, items, graph)
        loss = self.criterion(ratings_pred.view(-1), ratings)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    def output_knowledge(self, user_id, evaluation=False):
        user = [user_id] * self.config.num_items
        items = [i for i in range(self.config.num_items)]
        user = torch.LongTensor(user).to(self.config.device_id)
        items = torch.LongTensor(items).to(self.config.device_id)
        # print(f"user size:{user.size()}")
        # print(f"item size:{items.size()}")
        if self.graph is not None:
            predictions = self.model(user, items, copy.deepcopy(self.graph))
        else:
            predictions = self.model(user, items)
        predictions = predictions.squeeze()
        if evaluation:
            return predictions
        else:
            # select half of frequent update samples
            received_items = self.client_to_items[user_id]
            frequent_items = []
            for it in self.sorted_frequency.tolist():
                if len(frequent_items) >= self.config.transmit_num / 2: break
                if it not in received_items: frequent_items.append(it)

            # select half of hard samples
            _, rating_k = torch.topk(predictions.squeeze(), self.config.transmit_num)
            hard_items = []
            for it in rating_k.tolist():
                if len(hard_items) >= self.config.transmit_num / 2: break
                if it not in received_items and it not in frequent_items: hard_items.append(it)
            # merge two list
            selected_items = frequent_items + hard_items
            confidence = predictions[selected_items].cpu().tolist()

            return [selected_items, confidence]

    def train_(self, clients, batch_clients_idx, epoch_id):
        batch_loss = []
        users = []
        items = []
        ratings = []
        all_ratio = []

        f_score_top_before = []
        f_score_top_after = []
        f_score_order_before = []
        f_score_order_after = []
        fbb_list = []
        ftopbb_list = []

        for idx in batch_clients_idx:
            client = clients[idx]
            if epoch_id == 1:
                if self.config.use_privacy:
                    client_confidence, client_items, client_loss, ratio, f_before, f_order_before, f_after, f_order_after = client.train_(None)
                else:
                    client_confidence, client_items, client_loss, ratio, fbb, ftopbb = client.train_(None)
            else:

                server_data = self.output_knowledge(client.client_id)
                if len(server_data[0]) > len(client._train_):
                    server_data = [server_data[0][:len(client._train_)], server_data[1][:len(client._train_)]]
                if self.config.use_privacy:
                    client_confidence, client_items, client_loss, ratio, f_before, f_order_before, f_after, f_order_after = client.train_(server_data)
                else:
                    client_confidence, client_items, client_loss, ratio, fbb, ftopbb = client.train_(server_data)
            if self.config.use_privacy:
                f_score_top_before.append(f_before)
                f_score_top_after.append(f_after)
                f_score_order_before.append(f_order_before)
                f_score_order_after.append(f_order_after)
            else:
                fbb_list.append(fbb)
                ftopbb_list.append(ftopbb)
            batch_loss.append(client_loss)
            all_ratio.append(ratio)
            self.client_to_items[client.client_id] = copy.deepcopy(client_items)
            for it, rt in zip(client_items, client_confidence):
                users.append(client.client_id)
                items.append(it)
                ratings.append(rt)
                self.train_frequency[it] += 1
        if self.config.use_privacy:
            print(f"{sum(f_score_top_before) / len(f_score_top_before)}, {sum(f_score_top_after) / len(f_score_top_after)}")
            if len(f_score_order_before) != 0 and len(f_score_order_after) != 0:
                print(f"{sum(f_score_order_before) / len(f_score_order_before)}, {sum(f_score_order_after) / len(f_score_order_after)}")
        else:
            print(f"{sum(fbb_list) / len(fbb_list)}, {sum(ftopbb_list) / len(ftopbb_list)}")
        _, self.sorted_frequency = torch.sort(self.train_frequency, descending=True)
        dataset = ServerDataset(users, items, ratings)
        import time
        start_time = time.time()
        if self.config.model_type != "NCF":
            self.graph = self._get_graph(users, items, ratings)
        end_time = time.time()
        print(f"construct the graph cost: {end_time - start_time}")
        print(f"epoch {epoch_id}, server has {len(dataset)} data.")
        trainloader = DataLoader(dataset, batch_size=self.config.server_bs, shuffle=True)

        server_loss = []
        self.model.train()
        for _ in range(self.config.server_epoch):
            server_batch_loss = []
            for _, batch in enumerate(trainloader):
                user, item, rating = batch[0], batch[1], batch[2]
                loss = self.train_single_batch(user, item, rating, copy.deepcopy(self.graph))
                server_batch_loss.append(loss)
            server_loss.append(sum(server_batch_loss) / len(server_batch_loss))


        return batch_loss, server_loss

    def eval_(self, clients):

        server_test_cnt, server_test_results = 0, 0.
        local_test_cnt, local_test_results = 0, 0.
        mix_test_cnt, mix_test_results = 0, 0.
        with torch.no_grad():
            for client in clients:
                server_prediction = self.output_knowledge(client.client_id, evaluation=True)
                server_test_result, local_test_result, mix_test_result = client.eval_(server_prediction)
                if server_test_result is not None:
                    server_test_cnt += 1
                    server_test_results += server_test_result
                if local_test_result is not None:
                    local_test_cnt += 1
                    local_test_results += local_test_result
                if mix_test_result is not None:
                    mix_test_cnt += 1
                    mix_test_results += mix_test_result

        return server_test_results / server_test_cnt, local_test_results / local_test_cnt, mix_test_results / mix_test_cnt

