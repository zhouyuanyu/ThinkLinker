import os
from datetime import datetime

import numpy as np
import torch
import pytorch_lightning as pl
from PIL import Image

from transformers import CLIPProcessor

from codes.core.model.modeling_thinklinker import ThinkLinkerEncoder, ThinkLinkerMatcher
from codes.core.data.mention_enhance import MentionEnhancer


class ThinkLinkerLitModule(pl.LightningModule):
    def __init__(self, args):
        super(ThinkLinkerLitModule, self).__init__()
        self.args = args
        self.save_hyperparameters(args)   
        self.time = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.encoder = ThinkLinkerEncoder(args)
        self.matcher = ThinkLinkerMatcher(args)

        # 第二阶段引入提及增强模块
        if self.args.use_enhance:
            self.mention_enhance = MentionEnhancer(args)

        self.loss_fct = torch.nn.CrossEntropyLoss()

        self.tokenizer = CLIPProcessor.from_pretrained(self.args.pretrained_model).tokenizer
        self.image_processor = CLIPProcessor.from_pretrained(self.args.pretrained_model).feature_extractor

        self.eval_info = []


    def encode_entity_batch(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        pixel_values = batch.get('pixel_values', None)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        if pixel_values is None:
            empty_img_flag = torch.ones(input_ids.size(0), dtype=torch.bool, device=self.device)
        else:
            empty_img_flag = torch.all(pixel_values == 0, dim=(1, 2, 3))

        text_cls, image_cls, text_seq, image_patch = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )

        return text_cls, image_cls, text_seq, image_patch, empty_img_flag

    def encode_mention_batch(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        pixel_values = batch.get('pixel_values', None)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        if pixel_values is None:
            empty_img_flag = torch.ones(input_ids.size(0), dtype=torch.bool, device=self.device)
        else:
            empty_img_flag = torch.all(pixel_values == 0, dim=(1, 2, 3))

        text_cls, image_cls, text_seq, image_patch = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )

        return text_cls, image_cls, text_seq, image_patch

    def process_mention(self, data):
        input_dict_list = []
        pixel_values = []

        for sample_dict in data:
            if sample_dict['answer'] == 'nil':  # ignore the sample without ground truth
                continue
            mention = sample_dict['mentions']
            text = sample_dict['sentence']
            input_text = mention + ' [SEP] ' + text  # concat entity and text
            input_dict = self.tokenizer(input_text, padding='max_length', max_length=77,
                                        truncation=True)
            input_dict_list.append(input_dict)

            if sample_dict['img_path'] != "":
                try:
                    img_path = os.path.join(self.args.data.mention_img_folder, sample_dict['img_path'])
                    img = Image.open(img_path).resize((224, 224), Image.Resampling.LANCZOS)
                    pixel_value = self.image_processor(img, return_tensors='pt')['pixel_values'].squeeze()
                except:
                    pixel_value = torch.zeros((3, 224, 224))
            else:
                pixel_value = torch.zeros((3, 224, 224))

            pixel_values.append(pixel_value)

        input_dict = self.tokenizer.pad(input_dict_list, padding='max_length', max_length=77, return_tensors='pt')
        input_dict['pixel_values'] = torch.stack(pixel_values)

        return self.encode_mention_batch(input_dict)


    def training_step(self, batch):
        mention_batch = batch['mention_input_dict']
        entity_batch = batch['entity_input_dict']
        enhance_txt = batch['enhance_txt']

        # [bs, dim]
        mention_text_embeds, mention_image_embeds, mention_text_seq_tokens, mention_image_patch_tokens = \
            self.encoder(**mention_batch)
        entity_text_cls, entity_image_embeds, entity_text_seq_tokens, entity_image_patch_tokens = \
            self.encoder(**entity_batch)

        batch_size = mention_text_embeds.shape[0]
        total_entity = entity_text_cls.shape[0]
        entity_text_cls = entity_text_cls.reshape(batch_size, total_entity // batch_size,
                                                  -1)
        entity_image_embeds = entity_image_embeds.reshape(batch_size, total_entity // batch_size,
                                                          -1)
        length, dim = entity_text_seq_tokens.shape[-2:]
        entity_text_seq_tokens = entity_text_seq_tokens.reshape(batch_size, total_entity // batch_size, length,
                                                                dim)
        length, dim = mention_image_patch_tokens.shape[-2:]
        entity_image_patch_tokens = entity_image_patch_tokens.reshape(batch_size, total_entity // batch_size, length,
                                                                      dim)

        # 第二阶段使用增强提及文本
        if self.args.use_enhance:
            mention_enhance_embeds, _, mention_enhance_seq_tokens, _ = self.encoder(**enhance_txt)

            enhance_txt_size = mention_enhance_embeds.shape[0] // batch_size
            mention_enhance_embeds = mention_enhance_embeds.reshape(batch_size, enhance_txt_size, -1) # (batch_size, 生成的增强文本的个数, dim)
            length, dim = mention_enhance_seq_tokens.shape[-2:]
            mention_enhance_seq_tokens = mention_enhance_seq_tokens.reshape(batch_size, enhance_txt_size, length, dim) # (batch_size, 生成的增强文本的个数, length, dim)

            mention_text_embeds, mention_text_seq_tokens = \
                self.mention_enhance(mention_text_embeds, mention_enhance_embeds, mention_text_seq_tokens, mention_enhance_seq_tokens)



        logits, (text_logits, image_logits, image_text_logits) = self.matcher(entity_text_cls,
                                                                              entity_text_seq_tokens,
                                                                              mention_text_embeds,
                                                                              mention_text_seq_tokens,
                                                                              entity_image_embeds,
                                                                              entity_image_patch_tokens,
                                                                              mention_image_embeds,
                                                                              mention_image_patch_tokens)

        labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)

        text_loss = self.loss_fct(text_logits, labels)
        image_loss = self.loss_fct(image_logits, labels)
        image_text_loss = self.loss_fct(image_text_logits, labels)
        overall_loss = self.loss_fct(logits, labels)

        loss = overall_loss + text_loss + image_loss + image_text_loss
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mention_idx = batch['mention_idx']
        mention_input_dict = batch['mention_input_dict']
        entity_input_dict = batch['entity_input_dict']
        enhance_txt = batch['enhance_txt']

        mention_text_embeds, mention_image_embeds, mention_text_seq_tokens, mention_image_patch_tokens = \
            self.encoder(**mention_input_dict)
        entity_text_cls, entity_image_embeds, entity_text_seq_tokens, entity_image_patch_tokens = \
            self.encoder(**entity_input_dict)

        batch_size = mention_text_embeds.shape[0]
        total_entity = entity_text_cls.shape[0]
        entity_text_cls = entity_text_cls.reshape(batch_size, total_entity // batch_size,
                                                  -1)
        entity_image_embeds = entity_image_embeds.reshape(batch_size, total_entity // batch_size,
                                                          -1)
        length, dim = entity_text_seq_tokens.shape[-2:]
        entity_text_seq_tokens = entity_text_seq_tokens.reshape(batch_size, total_entity // batch_size, length,
                                                                dim)
        length, dim = mention_image_patch_tokens.shape[-2:]
        entity_image_patch_tokens = entity_image_patch_tokens.reshape(batch_size, total_entity // batch_size, length,
                                                                      dim)

        if self.args.use_enhance:
            mention_enhance_embeds, _, mention_enhance_seq_tokens, _ = self.encoder(**enhance_txt)

            enchance_txt_size = mention_enhance_embeds.shape[0] // batch_size
            mention_enhance_embeds = mention_enhance_embeds.reshape(batch_size, enchance_txt_size,
                                                                    -1)  # (batch_size, 生成的增强文本的个数, dim)
            length, dim = mention_enhance_seq_tokens.shape[-2:]
            mention_enhance_seq_tokens = mention_enhance_seq_tokens.reshape(batch_size, enchance_txt_size, length,
                                                                            dim)  # (batch_size, 生成的增强文本的个数, length, dim)

            mention_text_embeds, mention_text_seq_tokens = \
                self.mention_enhance(mention_text_embeds, mention_enhance_embeds, mention_text_seq_tokens,
                                     mention_enhance_seq_tokens)

        score, _ = self.matcher(entity_text_cls,
                                entity_text_seq_tokens,
                                mention_text_embeds,
                                mention_text_seq_tokens,
                                entity_image_embeds,
                                entity_image_patch_tokens,
                                mention_image_embeds,
                                mention_image_patch_tokens)
        rank = torch.argsort(torch.argsort(score, dim=-1, descending=True), dim=-1, descending=False) + 1
        tgt_rank = rank[torch.arange(score.shape[0]), 0]
        self.eval_info.append(dict(rank=tgt_rank.cpu(), all_rank=rank.cpu(), idx=mention_idx.cpu()))


    def validation_epoch_end(self, outputs):
        all_eval_info = self.all_gather(self.eval_info)
        save_folder = os.path.join('./rank_save', self.args.task, self.args.run_name + f'-{self.time}')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        all_rank = []
        mention_idx = []
        for _ in all_eval_info:
            batch_all_rank = _['all_rank'].cpu()
            batch_mention_idx = _['idx'].cpu()
            all_rank.append(batch_all_rank.reshape(-1, batch_all_rank.shape[-1]))
            mention_idx.append(batch_mention_idx.reshape(-1))
        all_rank = torch.concat(all_rank, dim=0).numpy()
        mention_idx = torch.concat(mention_idx, dim=0).numpy()

        ranks = torch.concat([_['rank'].cpu().flatten() for _ in all_eval_info]).numpy()
        hits20 = (ranks <= 20).mean()
        hits10 = (ranks <= 10).mean()
        hits5 = (ranks <= 5).mean()
        hits3 = (ranks <= 3).mean()
        hits2 = (ranks <= 2).mean()
        hits1 = (ranks <= 1).mean()

        self.log("Val/hits20", hits20, sync_dist=True, rank_zero_only=True)
        self.log("Val/hits10", hits10, sync_dist=True, rank_zero_only=True)
        self.log("Val/hits5", hits5, sync_dist=True, rank_zero_only=True)
        self.log("Val/hits3", hits3, sync_dist=True, rank_zero_only=True)
        self.log("Val/hits2", hits2, sync_dist=True, rank_zero_only=True)
        self.log("Val/hits1", hits1, sync_dist=True, rank_zero_only=True)
        self.log("Val/mr", ranks.mean(), sync_dist=True, rank_zero_only=True)
        self.log("Val/mrr", (1. / ranks).mean(), sync_dist=True, rank_zero_only=True)
        self.eval_info.clear()

        if self.trainer.is_global_zero:
            print('Saving {}'.format(os.path.join(save_folder, 'valid_all_rank.npy')))
            print('Saving {}'.format(os.path.join(save_folder, 'valid_mention_idx.npy')))
            np.save(os.path.join(save_folder, 'valid_all_rank.npy'), all_rank)
            np.save(os.path.join(save_folder, 'valid_mention_idx.npy'), mention_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        mention_idx = batch['mention_idx']
        mention_id = batch['mention_id']
        mention_input_dict = batch['mention_input_dict']
        entity_input_dict = batch['entity_input_dict']
        enhance_txt = batch['enhance_txt']

        mention_text_embeds, mention_image_embeds, mention_text_seq_tokens, mention_image_patch_tokens = \
            self.encoder(**mention_input_dict)
        entity_text_cls, entity_image_embeds, entity_text_seq_tokens, entity_image_patch_tokens = \
            self.encoder(**entity_input_dict)

        batch_size = mention_text_embeds.shape[0]
        total_entity = entity_text_cls.shape[0]
        entity_text_cls = entity_text_cls.reshape(batch_size, total_entity // batch_size,
                                                  -1)
        entity_image_embeds = entity_image_embeds.reshape(batch_size, total_entity // batch_size,
                                                          -1)
        length, dim = entity_text_seq_tokens.shape[-2:]
        entity_text_seq_tokens = entity_text_seq_tokens.reshape(batch_size, total_entity // batch_size, length,
                                                                dim)
        length, dim = mention_image_patch_tokens.shape[-2:]
        entity_image_patch_tokens = entity_image_patch_tokens.reshape(batch_size, total_entity // batch_size, length,
                                                                      dim)

        if self.args.use_enhance:
            mention_enhance_embeds, _, mention_enhance_seq_tokens, _ = self.encoder(**enhance_txt)

            enchance_txt_size = mention_enhance_embeds.shape[0] // batch_size

            mention_enhance_embeds = mention_enhance_embeds.reshape(batch_size, enchance_txt_size,
                                                                    -1)
            length, dim = mention_enhance_seq_tokens.shape[-2:]
            mention_enhance_seq_tokens = mention_enhance_seq_tokens.reshape(batch_size, enchance_txt_size, length,
                                                                            dim)

            mention_text_embeds, mention_text_seq_tokens = \
                self.mention_enhance(mention_text_embeds, mention_enhance_embeds, mention_text_seq_tokens,
                                     mention_enhance_seq_tokens)

        score, _ = self.matcher(entity_text_cls,
                                entity_text_seq_tokens,
                                mention_text_embeds,
                                mention_text_seq_tokens,
                                entity_image_embeds,
                                entity_image_patch_tokens,
                                mention_image_embeds,
                                mention_image_patch_tokens)
        rank = torch.argsort(torch.argsort(score, dim=-1, descending=True), dim=-1, descending=False) + 1
        tgt_rank = rank[torch.arange(score.shape[0]), 0]
        self.eval_info.append(dict(rank=tgt_rank.cpu(), all_rank=rank.cpu(), idx=mention_idx.cpu()))


    def test_epoch_end(self, outputs):
        all_eval_info = self.all_gather(self.eval_info)
        save_folder = os.path.join('./rank_save', self.args.task, self.args.run_name + f'-{self.time}')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        all_rank = []
        mention_idx = []
        for _ in all_eval_info:
            batch_all_rank = _['all_rank'].cpu()
            batch_mention_idx = _['idx'].cpu()
            all_rank.append(batch_all_rank.reshape(-1, batch_all_rank.shape[-1]))
            mention_idx.append(batch_mention_idx.reshape(-1))
        all_rank = torch.concat(all_rank, dim=0).numpy()
        mention_idx = torch.concat(mention_idx, dim=0).numpy()

        ranks = torch.concat([_['rank'].cpu().flatten() for _ in all_eval_info]).numpy()
        hits20 = (ranks <= 20).mean()
        hits10 = (ranks <= 10).mean()
        hits5 = (ranks <= 5).mean()
        hits3 = (ranks <= 3).mean()
        hits2 = (ranks <= 2).mean()
        hits1 = (ranks <= 1).mean()

        self.log("Test/hits20", hits20, sync_dist=True, rank_zero_only=True)
        self.log("Test/hits10", hits10, sync_dist=True, rank_zero_only=True)
        self.log("Test/hits5", hits5, sync_dist=True, rank_zero_only=True)
        self.log("Test/hits3", hits3, sync_dist=True, rank_zero_only=True)
        self.log("Test/hits2", hits2, sync_dist=True, rank_zero_only=True)
        self.log("Test/hits1", hits1, sync_dist=True, rank_zero_only=True)
        self.log("Test/mr", ranks.mean(), sync_dist=True, rank_zero_only=True)
        self.log("Test/mrr", (1. / ranks).mean(), sync_dist=True, rank_zero_only=True)
        self.eval_info.clear()

        if self.trainer.is_global_zero:
            print('Saving {}'.format(os.path.join(save_folder, 'test_all_rank.npy')))
            print('Saving {}'.format(os.path.join(save_folder, 'test_mention_idx.npy')))
            np.save(os.path.join(save_folder, 'test_all_rank.npy'), all_rank)
            np.save(os.path.join(save_folder, 'test_mention_idx.npy'), mention_idx)



    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_params = [
            {
                'params': [
                    p for n, p in self.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0001
            },
            {
                'params': [
                    p for n, p in self.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0
            }
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_params,
            lr=self.args.lr,
            betas=(0.9, 0.999),
            eps=1e-4
        )
        return [optimizer]
