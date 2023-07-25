# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from . import SASRecP, BERT4Rec
import random
import numpy as np

class ECL(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(ECL, self).__init__(config, dataset)

        # load parameters info
        self.initializer_range = config['initializer_range']

        self.mask_ratio = config['mask_ratio']
        self.generate_method = config['generate_method']

        self.mask_token = self.n_items
        self.hidden_size = config['hidden_size']  # same as embedding_size

        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        self.contras_loss_temp = config['contras_loss_temp']
        self.contras_target = config['contras_target']
        self.contras_k = config['contras_k']
        self.con_sim = Similarity(temp=self.contras_loss_temp)
        self.con_loss_fct = nn.CrossEntropyLoss()

        self.encode_loss_weight = config['encoder_loss_weight']
        self.discrim_loss_weight = config['discriminator_loss_weight']
        self.con_loss_weight = config['contrastive_loss_weight']
        self.con_m_loss_weight = config['contrastive_mask_loss_weight']
        self.generate_loss_weight = config['generate_loss_weight']

        self.pre_model_path = config['pre_model_path']

        self.loss_fct = nn.CrossEntropyLoss()
        self.disable_aug = config['disable_aug']

        self.discriminator_combine = config['discriminator_combine']
        self.discriminator_score = config['discriminator_score']
        self.discriminator_score_weight = config['discriminator_score_weight']
        self.discriminator_bidirectional = config['discriminator_bidirectional']
        self.ridl_type = config["ridl_type"]

        self.share_param = config['share_param']
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)  # mask token add 1

        # Always need encoder
        self.encoder = SASRecP(config, dataset, self.item_embedding).to(config['device'])

        # Always load generator with fixed parameters
        self.generator = BERT4Rec(config, dataset, self.item_embedding).to(config['device'])

        # Init discriminator
        # Set param sharing, the emb table is shared by default
        if self.share_param == 'all':
            self.discriminator = self.encoder
        else:
            self.discriminator = SASRecP(config, dataset).to(config['device'])



        self.discriminator_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.discriminator_linear = nn.Linear(self.hidden_size, 1, bias=False)

        # @todo Pls remember to set this var
        self.state = config['ecl_state'] # Discriminator task

        self.always_con = config['always_con'] # Discriminator task
        self.mask_strategy = config['mask_strategy']
        self.contras_forward = config['contras_forward']
        self.perturb_eps = config['perturb_eps']



        self.auto_weight = config['auto_weight']
        if self.auto_weight:
            self.generate_loss_weight = nn.Parameter(torch.tensor(self.generate_loss_weight))
            self.con_loss_weight = nn.Parameter(torch.tensor(self.con_loss_weight))
            self.discrim_loss_weight = nn.Parameter(torch.tensor(self.discrim_loss_weight))


        # parameters initialization
        self.apply(self._init_weights)
        self.encoder.apply(self._init_weights)
        if self.discrim_loss_weight != 0:
            self.discriminator.apply(self._init_weights)
        if self.generate_loss_weight != 0:
            self.generator.apply(self._init_weights)

        self.recall, self.recall_n = 0, 0

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward_discriminator(self, encode_output, generate_seq, item_seq_len):
        encode_output_last = encode_output[:,-1,:]
        discrim_output, flat_discrim_output = None, None
        if self.discriminator_combine == 'add':
            condition = encode_output_last.unsqueeze(1).repeat(1, self.max_seq_length, 1)
            item_emb = self.item_embedding(generate_seq)
            item_emb = item_emb + condition
            discrim_output = self.discriminator.forwardLogits(generate_seq, item_seq_len, item_emb, bidirectional=self.discriminator_bidirectional)
            flat_discrim_output = discrim_output.view(-1, self.hidden_size)

        if self.discriminator_combine == 'atten':
            condition = encode_output_last.unsqueeze(1)
            discrim_output = self.discriminator.forwardCLS(generate_seq, item_seq_len, condition, bidirectional=self.discriminator_bidirectional)
            flat_discrim_output = discrim_output.reshape(-1, self.hidden_size)


        if self.discriminator_combine == 'cat':
            item_emb = self.item_embedding(generate_seq)
            condition = encode_output_last.unsqueeze(1).repeat(1, self.max_seq_length, 1)
            condition_output = torch.cat((condition, item_emb), -1)
            condition_output = self.discriminator_transform(condition_output)
            discrim_output = self.discriminator.forwardLogits(generate_seq, item_seq_len, condition_output, bidirectional=self.discriminator_bidirectional)

            flat_discrim_output = discrim_output.view(-1, self.hidden_size)

        if self.discriminator_combine == 'addall':
            item_emb = self.item_embedding(generate_seq)
            item_emb = item_emb + encode_output
            discrim_output = self.discriminator.forwardLogits(generate_seq, item_seq_len, item_emb, bidirectional=self.discriminator_bidirectional)
            flat_discrim_output = discrim_output.view(-1, self.hidden_size)
        return discrim_output, flat_discrim_output

    def forward(self, item_seq, item_seq_len):
        encode_output = self.encoder.forward(item_seq, item_seq_len)
        # print('encoder time', time() - t)
        encode_output = encode_output
        # encode_output = self.gather_indexes(encode_output, item_seq_len - 1)

        masked_item_seq, ori_items, rep_items, pos_items, masked_index = self.reconstruct_train_data(item_seq,type=self.ridl_type)
        generate_seq, generate_loss = self.generator.predictSeq(masked_item_seq, masked_index, pos_items)

        # To monitor it
        all_rep = torch.sum(rep_items)
        all_ori = torch.sum(ori_items)
        all_pad = (item_seq == 0).long().sum()
        eql_items = (item_seq == generate_seq).long().sum() - all_pad
        recall = (eql_items.item() - all_ori.item()) / all_rep.item()
        self.recall += recall
        self.recall_n += 1

        if self.discrim_loss_weight == 0:
            return encode_output, None, None, None, generate_loss, None


        new_ori = (item_seq == generate_seq).long()
        new_ori = new_ori - (item_seq == 0).long()
        new_rep = (item_seq != 0).long() - new_ori

        discrim_output, flat_discrim_output = self.forward_discriminator(encode_output, generate_seq, item_seq_len)

        discrim = self.discriminator_linear(flat_discrim_output)
        replace_prob = torch.sigmoid(discrim).view(-1)

        return encode_output, replace_prob, new_ori.view(-1), new_rep.view(-1), generate_loss, discrim_output




    def forwardMaskCSE(self, item_seq, item_seq_len):
        encode_output = self.encoder.forward(item_seq, item_seq_len)
        masked_item_seq, ori_items, rep_items, pos_items, masked_index = self.reconstruct_train_data(item_seq)
        generate_seq, generate_loss = self.generator.predictSeq(masked_item_seq, masked_index, pos_items)
        return encode_output, generate_seq, generate_loss


    def calculate_con_loss(self, seq_output, seq_output_1):
        # use avg seq hidden to calculate con loss
        k = self.contras_k
        if self.contras_target == 'avg':
            logits_0 = seq_output.mean(dim=1).unsqueeze(1)
            logits_1 = seq_output_1.mean(dim=1).unsqueeze(0)
            cos_sim = self.con_sim(logits_0, logits_1).view(-1, logits_0.size(0))
            labels = torch.arange(logits_0.size(0)).long().to(self.device)
            con_loss = self.con_loss_fct(cos_sim, labels)

        # use lask k avg seq hidden to calculate con loss
        if self.contras_target == 'avgk':
            logits_0 = seq_output[:, -k:, :].mean(dim=1).unsqueeze(1)
            logits_1 = seq_output_1[:, -k:, :].mean(dim=1).unsqueeze(0)
            cos_sim = self.con_sim(logits_0, logits_1).view(-1, seq_output_1.size(0))
            labels = torch.arange(logits_0.size(0)).long().to(self.device)
            con_loss = self.con_loss_fct(cos_sim, labels)

        # use inter seq hidden for each position to calculate con loss
        if self.contras_target == 'inter':
            logits_0 = seq_output.transpose(0, 1).unsqueeze(2)
            logits_1 = seq_output_1.transpose(0, 1).unsqueeze(1)
            cos_sim = self.con_sim(logits_0, logits_1).view(-1, seq_output.size(0))
            labels = torch.arange(seq_output.size(0)).long().to(self.device)
            labels_n = labels.repeat(seq_output.size(1))
            con_loss = self.con_loss_fct(cos_sim, labels_n)

        if self.contras_target == 'interk':
            logits_0 = seq_output[:, -k:, :].transpose(0, 1).unsqueeze(2)
            logits_1 = seq_output_1[:, -k:, :].transpose(0, 1).unsqueeze(1)
            cos_sim = self.con_sim(logits_0, logits_1).view(-1, seq_output.size(0))
            labels = torch.arange(seq_output.size(0)).long().to(self.device)
            labels_n = labels.repeat(k)
            con_loss = self.con_loss_fct(cos_sim, labels_n)

        # use all seq hidden to calculate con loss
        if self.contras_target == 'all':
            logits_0 = seq_output[:, -k:, :].reshape(-1, seq_output.size(-1)).unsqueeze(1)
            logits_1 = seq_output_1[:, -k:, :].reshape(-1, seq_output.size(-1)).unsqueeze(0)
            cos_sim = self.con_sim(logits_0, logits_1).view(-1, logits_0.size(0))
            labels = torch.arange(logits_0.size(0)).long().to(self.device)
            con_loss = self.con_loss_fct(cos_sim, labels)

        return con_loss


    def calculate_mask_cse_loss(self, interaction):
        # @todo pos or neg
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, new_seq, gen_loss = self.forwardMaskCSE(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        item_label = item_seq[:, 1:]
        pad = pos_items.unsqueeze(-1)
        item_labeln = torch.cat((item_label, pad), dim=-1).long().to(self.device)

        seq_emb = seq_output.view(-1, self.hidden_size)  # [batch*seq_len hidden_size]
        test_item_emb = self.item_embedding.weight[:-1, :]
        logits = torch.matmul(seq_emb, test_item_emb.transpose(0, 1))
        pos_ids_l = torch.squeeze(item_labeln.view(-1))
        encode_loss = self.loss_fct(logits, pos_ids_l)

        con_loss = 0
        # Used to calculate contrastive loss

        return self.encode_loss_weight * encode_loss, self.con_loss_weight * con_loss, self.generate_loss_weight * gen_loss

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, replace_prob, ori_items, rep_items, generate_loss, discrim_output = self.forward(item_seq, item_seq_len)
        # print('all forward time', time() - t)

        pos_items = interaction[self.POS_ITEM_ID]


        if self.disable_aug == 'y' or self.disable_aug == 'h' or self.disable_aug == 'hm':
            item_label = item_seq[:, 1:]
            pad = pos_items.unsqueeze(-1)
            item_labeln = torch.cat((item_label, pad), dim=-1).long().to(self.device)
            seq_emb = seq_output.view(-1, self.hidden_size)  # [batch*seq_len hidden_size]
            test_item_emb = self.item_embedding.weight[:-1,:]
            logits = torch.matmul(seq_emb, test_item_emb.transpose(0, 1))
            pos_ids_l = torch.squeeze(item_labeln.view(-1))
            encode_loss = self.loss_fct(logits, pos_ids_l)
        else:
            seq_emb = seq_output[:, -1, :].squeeze(1)  # [batch hidden_size]
            test_item_emb = self.item_embedding.weight[:-1, :]
            logits = torch.matmul(seq_emb, test_item_emb.transpose(0, 1))
            encode_loss = self.loss_fct(logits, pos_items)




        discrim_loss = torch.tensor(0)
        # by current step
        if self.discrim_loss_weight != 0:
            pos_loss = torch.log(replace_prob.clamp(min=1e-8, max=1-1e-8)) * rep_items
            neg_loss = torch.log((1-replace_prob).clamp(min=1e-8, max=1-1e-8)) * ori_items
            rep_n, ori_n = rep_items.sum(), ori_items.sum()
            discrim_loss = torch.sum(-pos_loss-neg_loss)/torch.sum(rep_n + ori_n).clamp(min=1)


        con_loss = torch.tensor(0)
        if self.con_loss_weight != 0 and (self.state == 'sr' or self.always_con):
            # Used to calculate contrastive loss
            seq_output_1 = self.encoder.forward(item_seq, item_seq_len)
            con_loss = self.calculate_con_loss(seq_output, seq_output_1)

        return self.encode_loss_weight * encode_loss, self.con_loss_weight * con_loss, self.discrim_loss_weight * discrim_loss, self.generate_loss_weight * generate_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output_raw = self.encoder.forward(item_seq, item_seq_len)
        # seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        seq_output = seq_output_raw[:, -1, :].squeeze(1)

        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]

        if self.discriminator_score_weight != 0:
            discrim_output, flat_discrim_output = self.forward_discriminator(seq_output_raw, item_seq, item_seq_len)
            discrim_output = discrim_output[:, -1, :].squeeze(1)
            discrim_scores = torch.mul(discrim_output, test_item_emb).sum(dim=1)  # [B]

            if self.discriminator_score == 'add':
                scores += self.discriminator_score_weight * discrim_scores
            if self.discriminator_score == 'mul':
                scores = scores * ( 1 - (1 - discrim_scores ) * self.discriminator_score_weight)

        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output_raw = self.encoder.forward(item_seq, item_seq_len)
        # seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        seq_output = seq_output_raw[:, -1, :].squeeze(1)

        test_items_emb = self.item_embedding.weight[:-1, :]
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]

        if self.discriminator_score_weight != 0:
            discrim_output, flat_discrim_output = self.forward_discriminator(seq_output_raw, item_seq, item_seq_len)
            discrim_output = discrim_output[:, -1, :].squeeze(1)
            discrim_scores = torch.matmul(discrim_output, test_items_emb.transpose(0, 1))  # [B n_items]

            if self.discriminator_score == 'add':
                scores += self.discriminator_score_weight * discrim_scores
            if self.discriminator_score == 'mul':
                scores = scores * (1 - (1 - discrim_scores) * self.discriminator_score_weight)

        return scores

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence
    def _neg_sample(self, item_set):
        item = random.randint(1, self.n_items - 1)
        while item in item_set:
            item = random.randint(1, self.n_items - 1)
        return item

    def generate_train_data_random(self, item_seq):
        device = item_seq.device
        batch_size = item_seq.size(0)

        sequence_instances = item_seq.cpu().numpy().tolist()
        masked_item_sequence = []

        for instance in sequence_instances:
            # WE MUST USE 'copy()' HERE!
            masked_sequence = instance.copy()
            cnt = 0
            for index_id, item in enumerate(instance):
                # padding is 0, the sequence is not begin
                if item == 0:
                    continue

                prob = random.random()
                if prob < self.mask_ratio:
                    masked_sequence[index_id] = self._neg_sample(instance)
                    cnt += 1

                if cnt >= self.mask_ratio * len(instance) + 1:
                    # Don't mask too much
                    break

            if cnt == 0:
                # At least one
                masked_sequence[-1] = self._neg_sample(instance)

            masked_item_sequence.append(masked_sequence)

        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        return masked_item_sequence

    def generate_train_data_random_insert(self, item_seq):
        device = item_seq.device
        batch_size = item_seq.size(0)

        sequence_instances = item_seq.cpu().numpy().tolist()
        masked_item_sequence = []

        for instance in sequence_instances:
            # WE MUST USE 'copy()' HERE!
            masked_sequence = instance.copy()
            cnt = 0
            for index_id, item in enumerate(instance):
                # padding is 0, the sequence is not begin
                if item == 0 or len(masked_sequence) >= self.max_seq_length:
                    continue

                prob = random.random()
                if prob < self.mask_ratio:
                    masked_sequence.insert(index_id + cnt, self._neg_sample(instance))
                    cnt += 1

                if cnt >= self.mask_ratio * len(instance) + 1:
                    # Don't mask too much
                    break

            if cnt == 0:
                # At least one
                masked_sequence.insert(1, self._neg_sample(instance))

            masked_item_sequence.append(self._padding_sequence(masked_sequence, len(instance)))

        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        return masked_item_sequence

    def generate_train_data_random_delete(self, item_seq):
        device = item_seq.device
        batch_size = item_seq.size(0)

        sequence_instances = item_seq.cpu().numpy().tolist()
        masked_item_sequence = []

        for instance in sequence_instances:
            # WE MUST USE 'copy()' HERE!
            masked_sequence = instance.copy()
            cnt = 0
            for index_id, item in enumerate(instance):
                # padding is 0, the sequence is not begin
                if item == 0:
                    continue

                prob = random.random()
                if prob < self.mask_ratio:
                    masked_sequence.pop(index_id-cnt)
                    cnt += 1

                if cnt >= self.mask_ratio * len(instance) + 1:
                    # Don't mask too much
                    break

            if cnt == 0:
                # At least one
                masked_sequence.pop(1)

            masked_item_sequence.append(self._padding_sequence(masked_sequence, len(instance)))

        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        return masked_item_sequence

    def generate_train_data_random_crop(self, item_seq):
        device = item_seq.device
        batch_size = item_seq.size(0)

        sequence_instances = item_seq.cpu().numpy().tolist()
        masked_item_sequence = []

        for instance in sequence_instances:
            # WE MUST USE 'copy()' HERE!
            copied_sequence = instance.copy()
            sub_seq_length = int(self.mask_ratio * len(copied_sequence))
            # randint generate int x in range: a <= x <= b
            start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
            if sub_seq_length < 1:
                copied_sequence = [copied_sequence[start_index]]
            else:
                copied_sequence = copied_sequence[start_index:start_index + sub_seq_length]

            masked_item_sequence.append(self._padding_sequence(copied_sequence, len(instance)))

        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        return masked_item_sequence

    def generate_train_data_random_reorder(self, item_seq):
        device = item_seq.device
        batch_size = item_seq.size(0)

        sequence_instances = item_seq.cpu().numpy().tolist()
        masked_item_sequence = []

        for instance in sequence_instances:
            # WE MUST USE 'copy()' HERE!
            copied_sequence = instance.copy()
            sub_seq_length = int(self.mask_ratio * len(copied_sequence))
            start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
            sub_seq = copied_sequence[start_index:start_index + sub_seq_length]
            random.shuffle(sub_seq)
            reordered_seq = copied_sequence[:start_index] + sub_seq + \
                            copied_sequence[start_index + sub_seq_length:]
            assert len(copied_sequence) == len(reordered_seq)

            masked_item_sequence.append(self._padding_sequence(reordered_seq, len(instance)))

        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        return masked_item_sequence


    def reconstruct_train_data(self, item_seq, type=None):
        """
        Mask item sequence for training.
        """
        device = item_seq.device
        batch_size = item_seq.size(0)

        sequence_instances = item_seq.cpu().numpy().tolist()

        # Masked Item Prediction
        # [B * Len]
        masked_item_sequence = []
        ori_items = []
        rep_items = []
        pos_items = []
        masked_index = []
        for instance in sequence_instances:
            # WE MUST USE 'copy()' HERE!
            masked_sequence = instance.copy()
            ori_item = []
            rep_item = []
            pos_item = []
            index_ids = []
            # too slow here
            if self.mask_strategy == 'sample':
                # sample uniformly
                l = len(instance)
                s = len(np.nonzero(instance)[0])
                pad_n = l-s
                min_l = int(self.mask_ratio * s + 1)
                ids = [i for i in range(pad_n+1, l)]
                my_idxs = random.sample(ids, min_l)
                ori_item, rep_item = [1]*s, [0]*s
                for index_id in my_idxs:
                    pos_item.append(instance[index_id])
                    ori_item[index_id - pad_n] = 0
                    rep_item[index_id - pad_n] = 1
                    if type == None:
                        masked_sequence[index_id] = self.mask_token
                    if type == "insert":
                        masked_sequence.insert(index_id, self._neg_sample(instance))
                        masked_sequence = masked_sequence[:-1]
                    if type == "delete":
                        masked_sequence[index_id] = 0
                    if type == "subs":
                        masked_sequence[index_id] = self._neg_sample(instance)

                    index_ids.append(index_id)

                # l = len(instance)
                # s = l-len(np.nonzero(instance)[0])
                # index_id = random.randint(s+1, l-1)
                # pos_item.append(instance[index_id])
                # ori_item, rep_item = [1]*(l-s), [0]*(l-s)
                # ori_item[index_id-s] = 0
                # rep_item[index_id-s] = 1
                # masked_sequence[index_id] = self.mask_token
                # index_ids.append(index_id)

            else:
                # cal prob one by one
                for index_id, item in enumerate(instance):
                    # padding is 0, the sequence is end
                    if item == 0:
                        continue

                    if index_id == 0:
                        ori_item.append(1)
                        rep_item.append(0)
                        continue

                    prob = random.random()
                    if prob < self.mask_ratio:
                        pos_item.append(item)
                        ori_item.append(0)
                        rep_item.append(1)
                        if type == None:
                            masked_sequence[index_id] = self.mask_token
                        if type == "insert":
                            masked_sequence.insert(index_id, self._neg_sample(instance))
                            masked_sequence = masked_sequence[:-1]
                        if type == "delete":
                            masked_sequence[index_id] = 0
                        if type == "subs":
                            masked_sequence[index_id] = self._neg_sample(instance)

                        index_ids.append(index_id)
                    else:
                        ori_item.append(1)
                        rep_item.append(0)

                    if sum(rep_item) >= self.mask_ratio*len(instance)+1:
                        # Don't mask too much
                        break

                if sum(rep_item) == 0:
                    # At least one
                    pos_item.append(masked_sequence[-1])
                    masked_sequence[-1] = self.mask_token
                    ori_item[-1] = 0
                    rep_item[-1] = 1
                    index_ids.append(len(instance)-1)


            masked_item_sequence.append(masked_sequence)
            ori_items.append(self._padding_sequence(ori_item, self.max_seq_length))
            rep_items.append(self._padding_sequence(rep_item, self.max_seq_length))
            pos_items.append(self._padding_sequence(pos_item, self.mask_item_length))
            masked_index.append(self._padding_sequence(index_ids, self.mask_item_length))

        # [B Len]
        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        ori_items = torch.tensor(ori_items, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        rep_items = torch.tensor(rep_items, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        masked_index = torch.tensor(masked_index, dtype=torch.long, device=device).view(batch_size, -1)
        return masked_item_sequence, ori_items, rep_items, pos_items, masked_index

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
