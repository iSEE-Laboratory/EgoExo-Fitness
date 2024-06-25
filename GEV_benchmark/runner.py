import torch
import torch.optim as optim
import torch.nn as nn
from datasets import Execution_Verification_Dataset_Pair
from models import Video_backbone, MLP_block, Modal_Fuser, Video_Encoder
from torch.utils.tensorboard import SummaryWriter
import os
import utils
from tqdm import tqdm
from yacs.config import CfgNode as CN
import yaml
from logger import write_logs
import metrics
import torch.nn.functional as F

class Runner(object):
    def __init__(self, args=None):
        self.args = args
        # m-odel
        self.get_model()
        # optimizer
        self.optimizer = self.get_optimizer()
        # scheduler 
        self.scheduler = self.get_scheduler()
        # data loader
        self.get_data()
        # log recorder
        self.writer = self.get_log_recorder()
        # loss_fn
        self.get_loss_fns()
        

    def get_loss_fns(self):
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCELoss(reduction='none')
        self.weighted_mse = utils.weighted_mse_loss
        self.weighted_bce = utils.weighted_bce_loss

    def compute_loss(self, y, y_hat, auxiliary_data):
        """
        loss_type: ['mse', 'weighted_mse', 'bce', 'weighted_bce']
        y: ground_truth label
        y_hat: predicted results
        """
        l_t = self.args.loss_type
        action_ids = auxiliary_data['action_ids']
        # print(y)
        # print('1', y_hat)
        
        if l_t == 'mse':
            loss = self.mse(y_hat, y)
        elif l_t == 'weighted_mse':
            loss = self.weighted_mse(y_hat, y, self.loss_weight, action_ids)
        elif l_t == 'bce':
            loss = self.bce(y_hat, y)
        elif l_t == 'weighted_bce':
            loss = self.weighted_bce(y_hat, y, self.loss_weight, action_ids)
        # print(loss)
        
        return loss


    def get_data(self):
        # close-set action
        open_set_actions = [i for i in range(12) if str(i) in self.args.open_set_action.split(',')]
        close_set_actions = [i for i in range(12) if not str(i) in self.args.open_set_action.split(',')]
        
        train_dataset = Execution_Verification_Dataset_Pair(root_path=self.args.data_path, annotations_path=self.args.list_file_train, 
                                                   view=self.args.view, vid_enc=self.args.vid_enc, lan_enc=self.args.lan_enc, sample_ego=self.args.sample_ego, sample_rate=self.args.sample_rate, action_set=close_set_actions)
        test_dataset_exp = Execution_Verification_Dataset_Pair(root_path=self.args.data_path, annotations_path=self.args.list_file_test, test_mode=True, view=self.args.view, vid_enc=self.args.vid_enc, lan_enc=self.args.lan_enc, action_set=close_set_actions)
        test_dataset_all = Execution_Verification_Dataset_Pair(root_path=self.args.data_path, annotations_path=self.args.list_file_test, test_mode=True, view='all', vid_enc=self.args.vid_enc, lan_enc=self.args.lan_enc, action_set=close_set_actions)
        test_dataset_go3 = Execution_Verification_Dataset_Pair(root_path=self.args.data_path, annotations_path=self.args.list_file_test, test_mode=True, view='go3', vid_enc=self.args.vid_enc, lan_enc=self.args.lan_enc, action_set=close_set_actions)
        test_dataset_logit = Execution_Verification_Dataset_Pair(root_path=self.args.data_path, annotations_path=self.args.list_file_test, test_mode=True, view='logit', vid_enc=self.args.vid_enc, lan_enc=self.args.lan_enc, action_set=close_set_actions)
        if len(open_set_actions) != 0:
            open_test_dataset_go3 = Execution_Verification_Dataset_Pair(root_path=self.args.data_path, annotations_path=self.args.list_file_test, test_mode=True, view='go3', vid_enc=self.args.vid_enc, lan_enc=self.args.lan_enc, action_set=open_set_actions)
            open_test_dataset_logit = Execution_Verification_Dataset_Pair(root_path=self.args.data_path, annotations_path=self.args.list_file_test, test_mode=True, view='logit', vid_enc=self.args.vid_enc, lan_enc=self.args.lan_enc, action_set=open_set_actions)
            self.wo_open_set = False
        else:
            self.wo_open_set = True
        self.ce_weight = train_dataset.weight

        # set mse weight
        len_dict = len(train_dataset.action_count_dict)
        self.loss_weight = torch.zeros(len_dict, 2)
        self.loss_weight = utils.set_loss_weight(train_dataset.action_count_dict, self.loss_weight)

        print(train_dataset.__len__())
        print(test_dataset_all.__len__())
        print(test_dataset_go3.__len__())
        print(test_dataset_logit.__len__())
        if len(open_set_actions) != 0:
            print(open_test_dataset_go3.__len__())
            print(open_test_dataset_logit.__len__())
        # print(open_dataset.__len__())
        print('weight: ', self.ce_weight)
        print('loss_weight: ', self.loss_weight)
        print('training:',train_dataset.action_count_dict)
        print('testing_exp:',test_dataset_exp.action_count_dict)
        print('testing_all:',test_dataset_all.action_count_dict)
        print('testing_go3:',test_dataset_go3.action_count_dict)
        print('testing_logit:',test_dataset_logit.action_count_dict)
        if len(open_set_actions) != 0:
            print('open_testing_go3:',open_test_dataset_go3.action_count_dict)
            print('open_testing_logit:',open_test_dataset_logit.action_count_dict)
        # print('opening:',open_dataset.action_count_dict)
        print()

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.bs_train,
                                               shuffle=True,
                                               num_workers=int(self.args.workers), 
                                               pin_memory=True)
        self.test_loader_exp = torch.utils.data.DataLoader(test_dataset_exp, batch_size=self.args.bs_test,
                                              num_workers=int(self.args.workers),
                                              pin_memory=True, 
                                              shuffle=False)
        self.test_loader_all = torch.utils.data.DataLoader(test_dataset_all, batch_size=self.args.bs_test,
                                              num_workers=int(self.args.workers),
                                              pin_memory=True, 
                                              shuffle=False)
        self.test_loader_go3 = torch.utils.data.DataLoader(test_dataset_go3, batch_size=self.args.bs_test,
                                              num_workers=int(self.args.workers),
                                              pin_memory=True, 
                                              shuffle=False)
        self.test_loader_logit = torch.utils.data.DataLoader(test_dataset_logit, batch_size=self.args.bs_test,
                                              num_workers=int(self.args.workers),
                                              pin_memory=True, 
                                              shuffle=False)
        if len(open_set_actions) != 0:
            self.open_test_loader_go3 = torch.utils.data.DataLoader(open_test_dataset_go3, batch_size=self.args.bs_test,
                                                num_workers=int(self.args.workers),
                                                pin_memory=True, 
                                                shuffle=False)
            self.open_test_loader_logit = torch.utils.data.DataLoader(open_test_dataset_logit, batch_size=self.args.bs_test,
                                                num_workers=int(self.args.workers),
                                                pin_memory=True, 
                                                shuffle=False)
            # assert 1==2
        
        if self.args.test_on_training_set:
            self.test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.bs_test,
                                              num_workers=int(self.args.workers),
                                              pin_memory=True,
                                              shuffle=False)
        
    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return CN(yaml.safe_load(file))

    def get_model(self):
        # load backbone
        # base_model = Video_backbone(backbone_arch=self.args.backbone_arch, num_class=400)
        # base_model.load_pretrain(self.args.pretrained_backbone)
        fuser_cfg = None
        encoder_cfg = None
        if os.path.exists(self.args.cfg_path):
            print('load config from {}'.format(self.args.cfg_path))
            fuser_cfg_path = os.path.join(self.args.cfg_path, 'fuser.yaml')
            encoder_cfg_path = os.path.join(self.args.cfg_path, 'encoder.yaml')
            fuser_cfg = self.load_config(fuser_cfg_path)
            encoder_cfg = self.load_config(encoder_cfg_path)
        print('-'*100)
        print('encoder config:')
        print(encoder_cfg)
        print('-'*100)
        print('fuser config:')
        print(fuser_cfg)
        print('-'*100)
        # load video encoder
        vid_encoder = Video_Encoder(arch=self.args.video_enc_arch, cfg=encoder_cfg)
        # load modal fuser
        modal_fuser = Modal_Fuser(cfg=fuser_cfg, arch=self.args.fuser_arch)
        # load classifier
        classifier = MLP_block(in_dim=self.args.classifier_in_dim, out_dim=1)
        self.vid_encoder = vid_encoder
        self.modal_fuser = modal_fuser
        # self.base_model = base_model
        self.classifier = classifier
        return

    def get_optimizer(self):
        if not self.args.fix_backbone:
            optimizer = torch.optim.Adam(
                                    [
                                        # {'params': self.base_model.parameters()},
                                        {'params': self.modal_fuser.parameters()},
                                        {'params': self.classifier.parameters()},
                                        {'params': self.vid_encoder.parameters()}],
                                    self.args.base_lr,
                                    weight_decay=self.args.weight_decay)
        else:
            optimizer = torch.optim.Adam(
                                    [
                                    {'params': self.modal_fuser.parameters()},
                                    {'params': self.classifier.parameters()},
                                    {'params': self.vid_encoder.parameters()}],
                                    self.args.base_lr,
                                    weight_decay=self.args.weight_decay)
            utils.freeze_model(self.base_model)
        print('lr:', self.args.base_lr)
        print('wd: ', self.args.weight_decay)
        # assert 1==2
        return optimizer

    def get_scheduler(self):
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[25, 50, 75, 100, 125, 150, 175], gamma=0.5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                               T_max=self.args.num_epochs,
                                                               eta_min=self.args.base_lr * 0.01)
        return scheduler

    def get_log_recorder(self):
        writer = SummaryWriter(os.path.join(self.args.experiment_path, 'tensorboard'), comment='self.args.exp_name')
        return writer

    def to_cuda(self):
        self.loss_weight = self.loss_weight.cuda()

        # self.base_model = self.base_model.cuda()
        self.modal_fuser = self.modal_fuser.cuda()
        self.classifier = self.classifier.cuda()
        self.vid_encoder = self.vid_encoder.cuda()

        # self.base_model = nn.DataParallel(self.base_model)
        self.modal_fuser = nn.DataParallel(self.modal_fuser)
        self.classifier = nn.DataParallel(self.classifier)
        self.vid_encoder = nn.DataParallel(self.vid_encoder)

        self.ce = self.ce.cuda()
        self.mse = self.mse.cuda()
        self.bce = self.bce.cuda()
        return
    
    def save_ckpt(self, epoch_best, acc_best, ckpt_name='best_ckpt.pth'):
        torch.save({
                    # 'base_model' : self.base_model.state_dict(),
                    'modal_fuser' : self.modal_fuser.state_dict(),
                    'classifier' : self.classifier.state_dict(),
                    'vid_encoder' : self.vid_encoder.state_dict(),
                    'epoch_best': epoch_best,
                    'acc_best' : acc_best,
                    'experiment_name' : self.args.experiment_path.split('/')[-1]
                    }, os.path.join(self.args.experiment_path, ckpt_name))
        return

    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        # base_model_sd = ckpt['base_model']
        modal_fuser_sd = ckpt['modal_fuser']
        classifier_sd = ckpt['classifier']
        vid_encoder_sd = ckpt['vid_encoder']

        # self.base_model.load_state_dict({k.replace("module.", ""): v for k, v in base_model_sd.items()})
        self.modal_fuser.load_state_dict({k.replace("module.", ""): v for k, v in modal_fuser_sd.items()})
        self.classifier.load_state_dict({k.replace("module.", ""): v for k, v in classifier_sd.items()})
        self.vid_encoder.load_state_dict({k.replace("module.", ""): v for k, v in vid_encoder_sd.items()})
        
        print('loading ckpt: acc-{} @ epoch-{}'.format(ckpt['acc_best'], ckpt['epoch_best']))

    def network_forward(self, data):
        # video = data['video'].cuda()
        lan_feat = data['language_feat'].cuda()     # B, L, D
        lan_mask = data['language_mask'].cuda()
        l_len = data['language_len'].cuda()
        vid_feat = data['video_feat'].cuda()
        vid_mask = data['video_mask'].cuda()
        v_len = data['video_len'].cuda()
        y = data['label'][:,:,0].cuda().float()
        action_ids = data['action_id'].cuda()
        
        vid_feat = self.vid_encoder(vid_feat, src_key_padding_mask=vid_mask, v_len=v_len)
        fused_feat = self.modal_fuser(vid_feat, lan_feat, v_mask=vid_mask, t_mask=lan_mask, t_len=l_len, v_len=v_len) # B, L, hidden_dim
        y_hat = self.classifier(fused_feat)[:,:,0] # B, L
        # print(y_hat)
        return vid_feat, fused_feat, y_hat, y, lan_mask, vid_mask
        

    def train(self):
        # to cuda
        self.to_cuda()

        self.best_test_t1 = 0
        self.best_test_w_error_f1 = 0
        self.best_test_correct_instances = 0
        self.best_test_epoch = 0
        self.best_test_t1_epoch = 0
        self.best_test_w_error_f1_epoch = 0
        self.best_test_correct_instances_epoch = 0
        # TODO: Resume
        for epoch in range(self.args.num_epochs):
            print('Epoch {}:'.format(epoch))
            total_loss = 0
            total_bce_loss = 0
            total_nce_loss = 0
            total_acc = 0
            total_y_hat = torch.Tensor([])
            total_y = torch.Tensor([])

            # self.base_model.train()
            self.modal_fuser.train()
            self.classifier.train()
            self.vid_encoder.train()
            torch.set_grad_enabled(True)
            # for batch_idx, (data, exemplar) in enumerate(tqdm(self.train_loader, leave=False)):
            for batch_idx, (data, exemplar) in enumerate(self.train_loader):
                # video = data['video'].cuda()
                # assert 1==2
                vid_feat, fused_feat, y_hat, y, lan_mask, vid_mask = self.network_forward(data)
                vid_feat_exemplar, fused_feat_exemplar, y_hat_exemplar, y_exemplar, lan_mask_exemplar, vid_mask_exemplar = self.network_forward(exemplar)
                
                # BCE Loss
                bce_loss = self.bce(y_hat, y) # B, L
                lan_mask_ = (~lan_mask).int().expand(bce_loss.shape)
                bce_loss = bce_loss * lan_mask_
                bce_loss = torch.sum(bce_loss) / torch.sum(lan_mask_)
                # InfoNCE Loss
                vid_mask_feat_ = (vid_mask).int().unsqueeze(-1).expand(-1, -1, vid_feat.shape[-1])   # B,L,D
                avg_vid_feat = torch.sum(vid_feat * vid_mask_feat_, dim=1) / torch.sum(vid_mask_feat_, dim=1) # B, D
                avg_vid_feat_exemplar = torch.sum(vid_feat_exemplar * vid_mask_feat_, dim=1) / torch.sum(vid_mask_feat_, dim=1) # B, D
                temperature = 0.5  
                avg_vid_feat = avg_vid_feat / (avg_vid_feat.norm(dim=-1, keepdim=True) + 1e-6)
                avg_vid_feat_exemplar = avg_vid_feat_exemplar / (avg_vid_feat_exemplar.norm(dim=-1, keepdim=True) + 1e-6)
                logits_per_data = temperature * avg_vid_feat @ avg_vid_feat_exemplar.t()  # img -> text [b,t,l]
                logits_per_exemplar = temperature * avg_vid_feat_exemplar @ avg_vid_feat.t()  # text -> img  [b,l,t]

                nce_labels = torch.arange(len(logits_per_data)).cuda()
                loss_data = F.cross_entropy(logits_per_data, nce_labels)
                loss_exemplar = F.cross_entropy(logits_per_exemplar, nce_labels)

                info_nce_loss = loss_data + loss_exemplar
                
                loss = bce_loss + self.args.nce_factor * info_nce_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print(loss.item())
                
                # for name, param in self.classifier.named_parameters():
                #     print(param.grad)
                total_loss += loss.item()
                total_bce_loss += bce_loss.item()
                total_nce_loss += (self.args.nce_factor * info_nce_loss).item()
                
            if self.args.lr_decay:
                self.scheduler.step()
            
            train_results= self.eval(self.train_loader, epoch, True, subset='train')
            test_results_exp = self.eval(self.test_loader_exp, epoch, True, subset='test_exp') 
            test_results_all = self.eval(self.test_loader_all, epoch, True, subset='test_all')  
            test_results_go3 = self.eval(self.test_loader_go3, epoch, True, subset='test_go3')  
            test_results_logit = self.eval(self.test_loader_logit, epoch, True, subset='test_logit')  
            if not self.wo_open_set:
                open_test_results_go3 = self.eval(self.open_test_loader_go3, epoch, True, subset='open_test_go3')  
                open_test_results_logit = self.eval(self.open_test_loader_logit, epoch, True, subset='open_test_logit') 
            else:
                open_test_results_go3 = None
                open_test_results_logit = None
            # open_results= self.eval(self.open_loader, epoch, False, subset='open')  
            current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            write_logs(self.writer, epoch, current_lr, train_results, test_results_exp ,test_results_all, test_results_go3, test_results_logit,
               open_test_results_go3, open_test_results_logit)
            self.writer.add_scalar('Train_Loss/total_loss', total_loss/len(self.train_loader), epoch)
            self.writer.add_scalar('Train_Loss/bce_loss', total_bce_loss/len(self.train_loader), epoch)
            self.writer.add_scalar('Train_Loss/nce_loss', total_nce_loss/len(self.train_loader), epoch)
            
            # if self.args.view == 'all':
            #     test_results = test_results_all
            # elif self.args.view == 'ego':
            #     test_results = test_results_ego
            # else:
            #     test_results = test_results_exo
            test_results = test_results_exp
            if test_results['acc'] > self.best_test_t1:
                self.best_test_t1 = test_results['acc']
                self.best_test_t1_epoch = epoch
                self.writer.add_scalar('Best_test_t1:', self.best_test_t1, epoch)
                self.save_ckpt(epoch, self.best_test_t1, 'best_t1_ckpt.pth')
                if self.args.exp_name == 'debug':
                    print('save_best_t1_ckpt')
            if test_results['w_error_f1_score'] > self.best_test_w_error_f1:
                self.best_test_w_error_f1 = test_results['w_error_f1_score']
                self.best_test_w_error_f1_epoch = epoch
                self.writer.add_scalar('Best_test_w_error_f1:', self.best_test_w_error_f1, epoch)
                self.save_ckpt(epoch, self.best_test_w_error_f1, 'best_w_error_f1_ckpt.pth')
                if self.args.exp_name == 'debug':
                    print('save_best_f1s_ckpt')
            if test_results['correct_instances'] > self.best_test_correct_instances:
                self.best_test_correct_instances = test_results['correct_instances']
                self.best_test_correct_instances_epoch = epoch
                self.writer.add_scalar('Best_test_correct_instances:', self.best_test_correct_instances, epoch)
                self.save_ckpt(epoch, self.best_test_correct_instances, 'best_correct_instances_ckpt.pth')
                if self.args.exp_name == 'debug':
                    print('save_best_correct_instance_ckpt')
            print('  best test T1: {} @ {} epoch'.format(self.best_test_t1, self.best_test_t1_epoch))
            print('  best test F1: {} @ {} epoch'.format(self.best_test_w_error_f1, self.best_test_w_error_f1_epoch))
            print('  best test Correct Instances: {} @ {} epoch\n'.format(self.best_test_correct_instances, self.best_test_correct_instances_epoch))
            # 在这里写一个save_ckpt是为了方便resume
            self.save_ckpt(epoch, test_results['acc'], 'last_ckpt.pth')
        return

    def eval(self, loader, epoch=-1, print_per_action=False, subset='close'):
        total_loss = 0
        total_acc = 0
        correct_instances = 0
        total_instances = 0
        total_y_hat = torch.Tensor([])
        total_y = torch.Tensor([])
        total_action_id = torch.Tensor([])

        # self.base_model.eval()
        self.modal_fuser.eval()
        self.classifier.eval()
        self.vid_encoder.eval()
        torch.set_grad_enabled(False)
        
        for batch_idx, (data, exemplar) in enumerate(tqdm(loader, leave=False)):
            lan_feat = data['language_feat'].cuda()     # B, L, D
            lan_mask = data['language_mask'].cuda()
            l_len = data['language_len'].cuda()
            vid_feat = data['video_feat'].cuda()
            vid_mask = data['video_mask'].cuda()
            v_len = data['video_len'].cuda()
            y = data['label'][:,:,0].cuda().float()
            action_ids = data['action_id'].cuda()
            
            vid_feat = self.vid_encoder(vid_feat, src_key_padding_mask=vid_mask, v_len=v_len)
            fused_feat = self.modal_fuser(vid_feat, lan_feat, v_mask=vid_mask, t_mask=lan_mask, t_len=l_len, v_len=v_len) # B, L, hidden_dim
            y_hat = self.classifier(fused_feat)[:,:,0] # B, 1
            
            loss = self.bce(y_hat, y) # B, L
                
            lan_mask_ = (~lan_mask).int().expand(loss.shape)
            loss = loss * lan_mask_
            loss = torch.sum(loss) / torch.sum(lan_mask_)
            
            total_loss += loss.item()
            l_len = l_len.reshape(-1)
            
            
            correct_instances += metrics.correct_instance(y, y_hat, l_len)
            total_instances += y.shape[0]
            
            y_hat_flatten = torch.cat([y_hat[i][:l_len[i]] for i in range(y_hat.shape[0])])
            y_hat_flatten = torch.Tensor([[int(y_hat_i>=0.5)] for y_hat_i in y_hat_flatten.reshape(-1) ])
            y_flatten = torch.cat([y[i][:l_len[i]] for i in range(y.shape[0])])
            y_flatten = torch.Tensor([[int(y_i>=0.5)] for y_i in y_flatten.reshape(-1) ])
            # print(y_hat_flatten.shape) # N, 1
            # assert 1==2
            
            total_y_hat = torch.cat([total_y_hat, y_hat_flatten.detach().cpu()], dim=0)
            total_y = torch.cat([total_y, y_flatten.detach().cpu()], dim=0) 
            total_action_id = torch.cat([total_action_id, data['action_id'].detach().cpu()], dim=0)
        
        total_acc = utils.accuracy(total_y_hat, total_y, topk=(1,))[0]
        # print(total_y)
        # print(total_y_hat)
        
        class_count_dict = {0:0, 1:0}
        acc_count_dict = {0:0, 1:0}
        utils.per_class_accuracy(total_y, total_y_hat, class_count_dict, acc_count_dict, topk=(1,))
        
        # precision and recall
        wo_error_true_positives = acc_count_dict[1]      # sum((y_true == 1) & (y_pred == 1))
        wo_error_true_negatives = acc_count_dict[0]      # sum((y_true == 0) & (y_pred == 0))
        wo_error_false_positives = class_count_dict[0] - acc_count_dict[0]   # sum((y_true == 0) & (y_pred == 1))
        wo_error_false_negatives = class_count_dict[1] - acc_count_dict[1]   # sum((y_true == 1) & (y_pred == 0))

        wo_error_precision = wo_error_true_positives / (wo_error_true_positives + wo_error_false_positives + 1e-7)
        wo_error_recall = wo_error_true_positives / (wo_error_true_positives + wo_error_false_negatives + 1e-7)
        wo_error_f1_score =  2 * (wo_error_precision * wo_error_recall) / (wo_error_precision + wo_error_recall + 1e-7 )
        
        w_error_true_positives = acc_count_dict[0]      # sum((y_true == 0) & (y_pred == 0))
        w_error_true_negatives = acc_count_dict[1]      # sum((y_true == 1) & (y_pred == 1))
        w_error_false_positives = class_count_dict[1] - acc_count_dict[1]   # sum((y_true == 1) & (y_pred == 0))
        w_error_false_negatives = class_count_dict[0] - acc_count_dict[0]   # sum((y_true == 0) & (y_pred == 1))
        w_error_precision = w_error_true_positives / (w_error_true_positives + w_error_false_positives + 1e-7)
        w_error_recall = w_error_true_positives / (w_error_true_positives + w_error_false_negatives + 1e-7)
        w_error_f1_score =  2 * (w_error_precision * w_error_recall) / (w_error_precision + w_error_recall + 1e-7 )
        
        
        print('  {} set testing: '.format(subset))
        print('  |  avg loss: '.format(epoch), total_loss / len(loader), end='    ')
        print('acc: {}'.format(total_acc))
        print('  |  wo_error_precision: {}    wo_error_recall: {}    wo_error_f1-score: {}'.format(wo_error_precision, wo_error_recall, wo_error_f1_score))
        print('  |  w_error_precision: {}    w_error_recall: {}    w_error_f1-score: {}'.format(w_error_precision, w_error_recall, w_error_f1_score))

        # print(class_count_dict)
        print('  |  correct instances: {} / {}'.format(correct_instances, total_instances))
        for key in class_count_dict.keys():
            print('  |  label {}: {} / {}   {}'.format(key, acc_count_dict[key], class_count_dict[key], acc_count_dict[key]/class_count_dict[key]))

        eval_results = {
            'tot_loss': total_loss / len(loader), 'acc' : total_acc,
            'w_error_true_positives' : w_error_true_positives, 'w_error_true_negatives' : w_error_true_negatives,
            'w_error_false_positives' : w_error_false_positives, 'w_error_false_negatives' : w_error_false_negatives,
            'w_error_precision': w_error_precision, 'w_error_recall': w_error_recall, 'w_error_f1_score': w_error_f1_score,
            
            'wo_error_true_positives' : wo_error_true_positives, 'wo_error_true_negatives' : wo_error_true_negatives,
            'wo_error_false_positives' : wo_error_false_positives, 'wo_error_false_negatives' : wo_error_false_negatives,
            'wo_error_precision': wo_error_precision, 'wo_error_recall': wo_error_recall, 'wo_error_f1_score': wo_error_f1_score,
            
            "correct_instances": correct_instances
            # 'avg_f1_class': avg_f1_class
        }
        return eval_results

    def test(self):
        # 直接测试的接口
        print('========================================')
        print('test on {}'.format(self.args.ckpt_path))
        print('========================================')
        # load ckpt
        self.load_ckpt(self.args.ckpt_path)
        # to cuda
        self.to_cuda()
        # test
        results = self.eval(True)