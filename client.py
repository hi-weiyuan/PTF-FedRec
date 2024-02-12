import copy
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from models import *
from parse import args
from evaluate import evaluate_precision, evaluate_recall, evaluate_ndcg, evaluate_hr
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from torch.autograd import Variable

def to_gpu(var):
    return var.to(args.device)

class LocalDataset(Dataset):
    def __init__(self, items, labels):
        self.users = torch.LongTensor([0] * len(items))
        self.items = torch.LongTensor(items)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        return self.users[item], self.items[item], self.labels[item]

class FedRecClient(nn.Module):
    def __init__(self, config, train_ind, test_ind):
        super().__init__()
        # initialize client
        if config.model_type == "NCF":
            self.model = NCF(config).to(config.device_id)
        else:
            raise ImportError(f"the model type of clients should be in [NCF, ...]")
        # construct data
        self.config = config
        self.client_id = self.config.user_id
        self._train_ = train_ind
        self._test_ = test_ind
        self.m_item = config.num_items
        self.swat_limit = max(int(len(self._train_) * args.privacy_rate), 1)

        self.negative_items = []
        items, labels = [], []
        for pos_item in train_ind:
            items.append(pos_item)
            labels.append(1.)

            for _ in range(args.num_neg):
                neg_item = np.random.randint(self.m_item)
                while neg_item in train_ind:
                    neg_item = np.random.randint(self.m_item)
                items.append(neg_item)
                labels.append(0.)

        self.dataset = LocalDataset(items, labels)
        self.original_items = self.dataset.items.cpu().squeeze().tolist()
        self.previous_order = {}
        self.trainloader = DataLoader(self.dataset,
                                 batch_size=config.local_bs, shuffle=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.criterion = torch.nn.BCELoss()

    def server_data_clean(self, server_data):
        cleaned_item = []
        cleaned_label = []
        for it, lb in zip(server_data[0], server_data[1]):
            if it in self._train_:
                continue
            cleaned_item.append(it)
            cleaned_label.append(lb)
        return [cleaned_item, cleaned_label]

    def train_single_batch(self, users, items, labels):
        """train a model for a single batch"""
        users, items, ratings = users.to(self.config.device_id), items.to(self.config.device_id), labels.to(self.config.device_id)
        self.optimizer.zero_grad()
        ratings_pred = self.model(users, items)
        loss = self.criterion(ratings_pred.view(-1), ratings)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    def swap_protection(self, selected_confidence, indices):
        pos_item = copy.deepcopy(self._train_)
        item_conf = list(zip(indices, selected_confidence))
        item_conf_dict = {i: c for i, c in zip(indices, selected_confidence)}
        sorted_item_conf = sorted(item_conf, key=lambda x:x[1], reverse=True)

        top_pos_item = [(i, c) for i, c in sorted_item_conf if i in pos_item][:self.swat_limit]
        second_top_neg = [(i, c) for i, c in sorted_item_conf if i not in pos_item][:self.swat_limit]

        swap_count = min(len(top_pos_item), self.swat_limit)
        for i in range(swap_count):
            pos_i_id, pos_i_conf = top_pos_item[i]
            neg_i_id, neg_i_conf = second_top_neg[i]
            item_conf_dict[pos_i_id] = neg_i_conf
            item_conf_dict[neg_i_id] = pos_i_conf
        new_conf = []
        for i in indices:
            new_conf.append(item_conf_dict[i])

        return new_conf, indices
    def inference_attack(self, predictions, topk=10):
        _, guess_pos = torch.topk(predictions.squeeze(), topk if len(predictions.squeeze()) > topk else len(predictions.squeeze()))
        guess_pos = guess_pos.cpu().tolist()
        p = len(set(guess_pos).intersection(set(self._train_))) / len(guess_pos)

        r = len(set(guess_pos).intersection(set(self._train_))) / len(self._train_)
        if p + r == 0:
            f = 0
        else:
            f = 2 * p * r / (p + r)

        order_guess = []
        f_order = 0.

        _, order = torch.topk(predictions, len(list(set(self.original_items))))
        order = order.cpu().tolist()
        current_order = {o: i for i, o in enumerate(order)}
        if len(self.previous_order) > 0:
            for ii in self.original_items:
                if ii in current_order.keys() and ii in self.previous_order.keys():
                    if current_order[ii] < self.previous_order[ii]:
                        order_guess.append(ii)
                if len(order_guess) >= len(self._train_): break
            if len(order_guess) == 0:
                f_order = 0.
            else:
                p = len(set(order_guess).intersection(set(self._train_))) / len(order_guess)
                r = len(set(order_guess).intersection(set(self._train_))) / len(self._train_)
                if p + r == 0:
                    f_order = 0
                else:
                    f_order = 2 * p * r / (p + r)
        self.previous_order = copy.deepcopy(current_order)
        return f, f_order

    def train_(self, server_data):
        if args.use_partial:
            if not args.use_privacy:
                raise Variable(f"If use partial is True, use_privacy should be True")
            self.select_ratio = random.uniform(args.select_ratio, 1.0)
        else:
            self.select_ratio = 1.0
        if args.random_neg_ratio >=4:
            self.random_ratio = 4
        else:
            self.random_ratio = random.uniform(args.random_neg_ratio, 4.)
        if server_data is not None:
            server_data = self.server_data_clean(server_data)
            items = self.dataset.items.cpu().tolist()
            labels = self.dataset.labels.cpu().tolist()

            items.extend(server_data[0])
            labels.extend(server_data[1])

            self.dataset = LocalDataset(items, labels)

            self.trainloader = DataLoader(self.dataset,
                                 batch_size=self.config.local_bs, shuffle=True)

        epoch_loss = []
        self.model.train()
        for _ in range(self.config.local_epoch):
            batch_loss = []
            for batch_id, batch in enumerate(self.trainloader):
                user, item, rating = batch[0], batch[1], batch[2]
                loss = self.train_single_batch(user, item, rating)
                batch_loss.append(loss)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        self.model.eval()
        with torch.no_grad():
            items = [i for i in range(self.config.num_items)]
            user = [0] * len(items)
            user = torch.LongTensor(user).to(self.config.device_id)
            items = torch.LongTensor(items).to(self.config.device_id)
            predictions = self.model(user, items)
            predictions = predictions.squeeze()
            predictions[list(set(items.tolist()) - set(self.dataset.items.tolist()))] = - (1 << 10)

            indeces = self.dataset.items.tolist()
            predictions[list(set(items.tolist()) - set(indeces))] = - (1 << 10)
            fbb, ftopbb = self.inference_attack(predictions, topk=len(self._train_))

            pos_indeces = np.random.choice(np.array(self._train_), max(int(len(self._train_) * self.select_ratio), 1), replace=False).tolist()
            neg_num = max(int(len(pos_indeces) * self.random_ratio), 1)
            self.swat_limit = max(int(len(pos_indeces) * args.privacy_rate), 1)
            neg_indeces = np.random.choice(np.array(list(set(indeces) - set(self._train_))), neg_num, replace=True).tolist()
            indeces = pos_indeces + neg_indeces

            selected_confidence = predictions[indeces].cpu().tolist()
            selected_ratingk = indeces
            if self.config.use_privacy:
                predictions[list(set(items.tolist()) - set(indeces))] = - (1 << 10)
                f_before, f_order_before = self.inference_attack(predictions, topk=max(int(len(selected_ratingk)* 0.2), 1))
                if args.use_swap:
                    selected_confidence, selected_ratingk = self.swap_protection(selected_confidence, selected_ratingk)
                predictions = predictions.cpu()
                predictions[selected_ratingk] = torch.tensor(selected_confidence)
                f_after, f_order_after = self.inference_attack(predictions, topk=max(int(len(selected_ratingk)* 0.2), 1))

                return selected_confidence, selected_ratingk, sum(epoch_loss) / len(epoch_loss), len(set(selected_ratingk).intersection(set(self._train_))) / len(list(set(self._train_))), f_before, f_order_before, f_after, f_order_after
        return selected_confidence, selected_ratingk, sum(epoch_loss) / len(epoch_loss), len(set(selected_ratingk).intersection(set(self._train_))) / len(self._train_), fbb, ftopbb

    def noise(self, shape, std):
        noise = np.random.multivariate_normal(
            mean=np.zeros(shape[1]), cov=np.eye(shape[1]) * std, size=shape[0]
        )
        return torch.Tensor(noise).to(args.device)
    def eval_(self, server_prediction):
        server_prediction = server_prediction.squeeze()
        server_prediction[self._train_] = - (1 << 10)
        if self._test_:
            hr_at_20 = evaluate_recall(server_prediction, self._test_, 20)
            ndcg_at_20 = evaluate_ndcg(server_prediction, self._test_, 20)
            server_test_result = np.array([hr_at_20, ndcg_at_20])
        else:
            server_test_result = None

        self.model.eval()
        with torch.no_grad():
            user = [0] * self.config.num_items
            items = [i for i in range(self.config.num_items)]
            user = torch.LongTensor(user).to(self.config.device_id)
            items = torch.LongTensor(items).to(self.config.device_id)
            predictions = self.model(user, items)
            predictions = predictions.squeeze()
            predictions[self._train_] = - (1 << 10)
        if self._test_:
            hr_at_20 = evaluate_recall(predictions, self._test_, 20)
            ndcg_at_20 = evaluate_ndcg(predictions, self._test_, 20)
            local_test_result = np.array([hr_at_20, ndcg_at_20])
        else:
            local_test_result = None

        new_prediction = copy.deepcopy(server_prediction)
        new_prediction[self.dataset.items] = predictions[self.dataset.items]
        mix_prediction = (server_prediction + new_prediction) / 2

        if self._test_:
            hr_at_20 = evaluate_recall(mix_prediction, self._test_, 20)
            ndcg_at_20 = evaluate_ndcg(mix_prediction, self._test_, 20)
            mix_test_result = np.array([hr_at_20, ndcg_at_20])
        else:
            mix_test_result = None
        return server_test_result, local_test_result, mix_test_result



