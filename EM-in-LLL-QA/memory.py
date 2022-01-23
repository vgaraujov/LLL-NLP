from transformers import BertModel
from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
import pickle
import os
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)


from settings import model_classes
from utils import pad_to_max_len


class Memory:
    def __init__(self, args):
        self.n_neighbors = args.n_neighbors
        self.mem_capacity = args.mem_capacity
        with torch.no_grad():
            logger.info("Initializing memory {} model".format(args.model_name))
            self.model = BertModel.from_pretrained(args.model_name).cuda()
            self.model.eval()
        self.hidden_size = self.model.config.hidden_size
        self.max_len = self.model.config.max_position_embeddings
        self.keys, self.input_ids, self.token_type_ids, self.labelss, self.labelse = [], [], [], [], []
        self.tree = NearestNeighbors(n_jobs=args.n_workers)
        self.built_tree = False

    def add(self, input_ids, masks, token_type_ids, labelss, labelse):
        if self.built_tree:
            logging.warning("Tree already build! Ignore add.")
            return
        
        prob = torch.rand(input_ids.size(0))
        input_ids = input_ids[prob<=self.mem_capacity]
        masks = masks[prob<=self.mem_capacity]
        token_type_ids = token_type_ids[prob<=self.mem_capacity]
        labelss = labelss[prob<=self.mem_capacity]
        labelse = labelse[prob<=self.mem_capacity]

        outputs = self.model(input_ids=input_ids, attention_mask=masks, token_type_ids=token_type_ids)
        self.keys.extend(outputs[0][:, 0, :].detach().cpu().tolist())
        for input_id, mask, token_type_id in zip(input_ids.cpu().tolist(), masks.cpu().tolist(), token_type_ids.cpu().tolist()):
            min_zero_id = len(mask)
            while mask[min_zero_id-1] == 0:
                min_zero_id -= 1
            self.input_ids.append(input_id[:min_zero_id])
            self.token_type_ids.append(token_type_id[:min_zero_id])
        self.labelss.extend(labelss.cpu().tolist())
        self.labelse.extend(labelse.cpu().tolist())
        del outputs


    def sample(self, n_samples):
        if self.built_tree:
            logging.warning("Tree already build! Ignore sample.")
            return
        inds = np.random.randint(len(self.labelss), size=n_samples)
        input_ids = [self.input_ids[ind] for ind in inds]
        token_type_ids = [self.token_type_ids[ind] for ind in inds]
        labelss = [self.labelss[ind] for ind in inds]
        labelse = [self.labelse[ind] for ind in inds]

        input_ids, masks = pad_to_max_len(input_ids)
        token_type_ids, _ = pad_to_max_len(token_type_ids)
        labelss = torch.tensor(labelss, dtype=torch.long)
        labelse = torch.tensor(labelse, dtype=torch.long)
        return input_ids.cuda(), masks.cuda(), token_type_ids.cuda(), labelss.cuda(), labelse.cuda()


    def build_tree(self):
        if self.built_tree:
            logging.warning("Tree already build! Ignore build.")
            return
        self.built_tree = True
        self.keys = np.array(self.keys)
        self.tree.fit(self.keys)
        self.input_ids = np.array(self.input_ids)
        self.token_type_ids = np.array(self.token_type_ids)
        self.labelss = np.array(self.labelss)
        self.labelse = np.array(self.labelse)

    def query(self, input_ids, masks, token_type_ids):
        if not self.built_tree:
            logging.warning("Tree not built! Ignore query.")
            return
        outputs = self.model(input_ids=input_ids, attention_mask=masks, token_type_ids=token_type_ids)
        queries = outputs[0][:, 0, :].cpu().numpy()
        inds = self.tree.kneighbors(queries, n_neighbors=self.n_neighbors, return_distance=False)
        input_ids, masks = list(zip(*[pad_to_max_len(input_id) for input_id in self.input_ids[inds]]))
        token_type_ids, _ = list(zip(*[pad_to_max_len(token_type_id) for token_type_id in self.token_type_ids[inds]]))
        labelss = [torch.tensor(label, dtype=torch.long) for label in self.labelss[inds]]
        labelse = [torch.tensor(label, dtype=torch.long) for label in self.labelse[inds]]
        return input_ids, masks, token_type_ids, labelss, labelse
