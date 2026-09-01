import io
import os
import os.path
import random

import h5py
import numpy as np
import torch
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from codes.core.data.functions import load_jsonl_file

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(DataModule, self).__init__()
        self.args = args
        self.tokenizer = CLIPProcessor.from_pretrained(self.args.pretrained_model).tokenizer
        self.image_processor = CLIPProcessor.from_pretrained(self.args.pretrained_model).feature_extractor
        self.clip_processor = _transform(224)

    def load_image_from_h5py(self, key_list, is_mention):
        if is_mention:
            image_bytes = [self.mention_image_ds.get(key, None) for key in key_list]
        else:
            image_bytes = [self.entity_image_ds.get(key, None) for key in key_list]
        image_objs = [Image.open(io.BytesIO(np.array(_))) if _ is not None else Image.new('RGB', (224, 224), 'white') for _ in image_bytes]
        pixel_values = torch.stack([self.clip_processor(_) for _ in image_objs])
        return pixel_values

    def select_candidates(self, candidate_list):
        gt_qid = candidate_list[0]
        random_candidates = random.sample(candidate_list[1:], self.args.data.num_train_candidate)
        return [gt_qid] + random_candidates

    def build_enhance_texts(self, batch):
        gen_lens = [len(b.get('generated') or []) for b in batch]
        max_gen = max(gen_lens) if gen_lens else 0
        K = max_gen if max_gen > 0 else 1
        flat_list = []

        for s in batch:
            mention = s.get('mentions') or ""
            gens = s.get('generated') or []

            if gens:
                answers = []
                for g in gens:
                    raw = g.get('answer')
                    txt = (mention + ". " if mention else "") + str(raw)
                    answers.append(txt)

                if len(answers) >= K:
                    answers = answers[:K]
                else:
                    pad = s.get('sentence')
                    pad_txt = (mention + ". " if mention else "") + str(pad)
                    answers += [pad_txt] * (K - len(answers))
            else:
                fill = s.get('sentence')
                fill_txt = (mention + ". " if mention else "") + str(fill)
                answers = [fill_txt] * K
            flat_list.extend(answers)
        return flat_list

    def setup(self, stage: str):
        data_args = self.args.data
        self.data_args = data_args

        self.data_args.entity = os.path.join(self.data_args.folder, self.data_args.entity)
        self.data_args.train_data = os.path.join(self.data_args.folder, self.data_args.train_data)
        self.data_args.valid_data = os.path.join(self.data_args.folder, self.data_args.valid_data)
        self.data_args.test_data = os.path.join(self.data_args.folder, self.data_args.test_data)
        self.data_args.image_h5 = os.path.join(self.data_args.folder, self.data_args.image_h5)

        if not hasattr(self, 'entity'):
            self.entity = load_jsonl_file(self.data_args.entity, desc='Entity', key='qid')  # qid -> entity information
            self.qid2id = {d['qid']: d['id'] for _, d in self.entity.items()}  # qid -> id
            image_h5py_file = h5py.File(data_args.image_h5, 'r')
            self.entity_image_ds = image_h5py_file['entity_image']
            self.mention_image_ds = image_h5py_file['mention_image']

        if stage == 'fit':
            if not hasattr(self, 'train_data') and not hasattr(self, 'valid_data'):
                self.train_data = load_jsonl_file(self.data_args.train_data, desc='Train')
                if getattr(self.args.data, 'percentage', 1) < 1:
                    self.train_data = self.train_data[
                                      :int(len(self.train_data) * getattr(self.args.data, 'percentage', 1))]

                for idx in range(len(self.train_data)):
                    self.train_data[idx].update({'idx': idx})
                self.valid_data = load_jsonl_file(self.data_args.valid_data, desc='Valid')
                for idx in range(len(self.valid_data)):
                    self.valid_data[idx].update({'idx': idx})

        elif stage == 'test':
            if not hasattr(self, 'test_data'):
                self.test_data = load_jsonl_file(self.data_args.test_data, desc='Test')
                for idx in range(len(self.test_data)):
                    self.test_data[idx].update({'idx': idx})

    def train_collator(self, batch, is_eval):
        mention_text = [_['mentions'] + '. ' + _['sentence'] for _ in batch]
        mention_idx = torch.tensor([_['idx'] for _ in batch], dtype=torch.int)
        mention_id = [_['mention_id'] for _ in batch]
        mention_image_file = [_['imgPath'].split('/')[-1].split('.')[0] for _ in batch]
        entity_candidates_qid = sum(
            [_['candidates'] if is_eval else self.select_candidates(_['candidates']) for _ in batch],
            [])
        entity_candidates_dict = [self.entity[qid] for qid in entity_candidates_qid]
        entity_candidates_text = [d['name'] + '. ' + d.get('desc', '') for d in
                                  entity_candidates_dict]

        mention_input_dict = self.tokenizer(mention_text, truncation=True, padding='max_length',
                                            max_length=self.args.data.text_max_length, return_tensors='pt')

        mention_pixel_values = self.load_image_from_h5py(mention_image_file, is_mention=True)
        mention_input_dict['pixel_values'] = mention_pixel_values

        enhance_tokenized = []
        if self.args.use_enhance:
            enhance_txt = self.build_enhance_texts(batch)
            enhance_tokenized = self.tokenizer(
                enhance_txt,
                truncation=True,
                padding='max_length',
                max_length=self.args.data.text_max_length,
                return_tensors='pt'
            )

        entity_input_dict = self.tokenizer(entity_candidates_text, truncation=True, padding='max_length',
                                           max_length=self.args.data.text_max_length, return_tensors='pt')
        entity_pixel_values = self.load_image_from_h5py(entity_candidates_qid, is_mention=False)
        entity_input_dict['pixel_values'] = entity_pixel_values

        return {
            'mention_idx': mention_idx,
            'mention_id': mention_id,
            'mention_input_dict': mention_input_dict,
            'enhance_txt': enhance_tokenized,
            'entity_input_dict': entity_input_dict,
            'candidate_qids': entity_candidates_qid
        }

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.args.data.batch_size,
                          num_workers=self.args.data.num_workers,
                          shuffle=True,
                          collate_fn=lambda x: self.train_collator(x, is_eval=False))

    def val_dataloader(self):
        return DataLoader(self.valid_data,
                          batch_size=self.args.data.eval_batch_size,
                          num_workers=self.args.data.num_workers,
                          shuffle=False,
                          collate_fn=lambda x: self.train_collator(x, is_eval=True))

    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.args.data.eval_batch_size,
                          num_workers=self.args.data.num_workers,
                          shuffle=False,
                          collate_fn=lambda x: self.train_collator(x, is_eval=True))