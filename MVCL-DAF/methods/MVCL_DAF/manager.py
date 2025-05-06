import torch
import torch.nn.functional as F
import logging
from torch import nn
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from data.utils import get_dataloader
from utils.metrics import AverageMeter, Metrics
from transformers import AdamW, get_linear_schedule_with_warmup
from .model import MVCL_DAF
from .loss import SupConLoss, InfoNCE, Multi_infoNCE, Multi_SupCon
import numpy as np
import os
import pandas as pd

__all__ = ['MVCL_DAF_manager']

class MVCL_DAF_manager:

    def __init__(self, args, data, labels_weight):
             
        self.logger = logging.getLogger(args.logger_name)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        args.device = self.device
        self.model = MVCL_DAF(args)
        self.model.to(self.device)
        self.optimizer, self.scheduler = self._set_optimizer(args, self.model)

        mm_dataloader = get_dataloader(args, data.mm_data)
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            mm_dataloader['train'], mm_dataloader['dev'], mm_dataloader['test']
            
        self.args = args
        self.labels_weight = labels_weight.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        print('manager_re_multi_view')
        print('loss', args.loss)
        print('align_method',args.aligned_method)
        if args.loss == 'InfoNCE':
            self.cons_criterion = Multi_infoNCE(temperature=args.temperature, reduction='mean', negative_mode='unpaired')
        if args.loss == 'SupCon':
            self.cons_criterion = Multi_SupCon(temperature=args.temperature)
        self.metrics = Metrics(args)
        
        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path, self.device)

    
    def _set_optimizer(self, args, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)
        
        if args.learning_rate_method == 'Cosine annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0.0, T_max=args.num_train_epochs//5)
        else:  # default scheduler if not using Cosine annealing
            num_train_optimization_steps = int(args.num_train_examples / args.train_batch_size) * args.num_train_epochs
            num_warmup_steps = int(args.num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_batch_size)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_optimization_steps)
        
        return optimizer, scheduler



    def _train(self, args): 
        
        early_stopping = EarlyStopping(args)
        no_improve_epochs = 0
        self.best_eval_score = 0
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            # self.model_attention.train() # !!
            loss_record = AverageMeter()
            cons_loss_record = AverageMeter()
            cls_loss_record = AverageMeter()
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)
                cons_text_feats = batch['cons_text_feats'].to(self.device) 
                condition_idx = batch['condition_idx'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)              
                label_ids = batch['label_ids'].to(self.device)

                

                with torch.set_grad_enabled(True):

                    logits, _, condition, cons_condition, text_condition, visual_condition, acoustic_condition = self.model(text_feats, video_feats, audio_feats, cons_text_feats, condition_idx)
                    cons_feature = torch.cat((condition.unsqueeze(1), cons_condition.unsqueeze(1)), dim=1)
                    text_feature = torch.cat((text_condition.unsqueeze(1), cons_condition.unsqueeze(1)), dim=1)
                    visual_feature = torch.cat((visual_condition.unsqueeze(1), cons_condition.unsqueeze(1)), dim=1)
                    acoustic_feature = torch.cat((acoustic_condition.unsqueeze(1), cons_condition.unsqueeze(1)), dim=1)
                    if args.loss == 'InfoNCE':
                        cons_loss = self.cons_criterion.compute_loss(text_anchor=cons_condition, 
                                                                     text_view = text_condition,
                                                                     visual_view = visual_condition,
                                                                     acoustic_view = acoustic_condition,
                                                                     global_view=condition).to(self.device)
                    elif args.loss == 'SupCon':
                        cons_loss = self.cons_criterion.compute_loss(cons_feature, text_feature, visual_feature, acoustic_feature).to(self.device)
                    cls_loss = self.criterion(logits, label_ids).to(self.device)
                    loss = cons_loss + cls_loss 
                    self.optimizer.zero_grad()

                    
                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))
                    cons_loss_record.update(cons_loss.item(), label_ids.size(0))
                    cls_loss_record.update(cls_loss.item(), label_ids.size(0))

                    if args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                    self.optimizer.step()
                    self.scheduler.step()

            
            outputs = self._get_outputs(args, self.eval_dataloader)
            eval_score = outputs[args.eval_monitor]

            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'cons_loss': round(cons_loss_record.avg, 4),
                'cls_loss': round(cls_loss_record.avg, 4),
                'eval_score': round(eval_score, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in eval_results.keys():
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            early_stopping(eval_score, self.model)

            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break


            if eval_score > self.best_eval_score:
                self.best_eval_score = eval_score
                no_improve_epochs = 0  # Reset no improve count
                save_path = '/home/freyr/MVCL-DAF/methods/MVCL_DAF/Models'
                torch.save(self.model.state_dict(), os.path.join(save_path, 'best_model.pth'))
                self.logger.info('The Best Model is Saved')
            else:
                no_improve_epochs += 1
                if args.learning_rate_method == 'decay' and no_improve_epochs >= 4:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] / 2
                    self.logger.info(f'Learning rate decayed to {self.optimizer.param_groups[0]["lr"]}')
                    no_improve_epochs = 0  # Optionally reset no improve count here  
            
            # early_stopping(eval_score, self.model)
            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

            if args.learning_rate_method == 'Cosine annealing':
                self.scheduler.step()

            # Print current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f'Current learning rate: {current_lr:.6f}')
        
        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 


    def _get_outputs(self, args, dataloader, show_results = False):

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        total_features = torch.empty((0, args.feat_size)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):

            text_feats = batch['text_feats'].to(self.device)
            cons_text_feats = batch['cons_text_feats'].to(self.device)
            condition_idx = batch['condition_idx'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
                
            with torch.set_grad_enabled(False):
                logits, features, condition, cons_condition, text_condition, visual_condition, acoustic_condition \
                    = self.model(text_feats, video_feats, audio_feats, cons_text_feats, condition_idx)
                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, features))

        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)

        y_logit = total_logits.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        y_prob = total_maxprobs.cpu().numpy()
        y_feat = total_features.cpu().numpy()

        
        outputs = self.metrics(y_true, y_pred, show_results = show_results)
        
        if args.save_pred and show_results:
            np.save('y_true_' + str(args.seed) + '.npy', y_true)
            np.save('y_pred_' + str(args.seed) + '.npy', y_pred)

        outputs.update(
            {
                'y_prob': y_prob,
                'y_logit': y_logit,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_feat': y_feat
            }
        )

        return outputs

    def _test(self, args):
        save_path = '/home/freyr/MVCL-DAF/methods/MVCL_DAF/Models'
        self.model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pth')))
        self.model.to(self.device)
        self.model.eval()

        test_results = {}
        ind_outputs = self._get_outputs(args, self.test_dataloader, show_results=True)
        ind_outputs['best_eval_score'] = round(self.best_eval_score, 4)
        test_results.update(ind_outputs)
        
        return test_results


    





