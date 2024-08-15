import os
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter
# Model
import models.model
# Dataset
import Human36M.dataset, COCO.dataset, PW3D.dataset, MPII3D.dataset, MPII.dataset
from multiple_datasets import MultipleDatasets
# Trainer
from core.config import cfg
from core.loss import get_loss, get_loss_dict
from funcs_utils import get_optimizer, load_checkpoint, get_scheduler, count_parameters, lr_check

def COCO2H36M(coco_joint):
    """
    B, T, J, 2
    coco : ['Nose':0, 'L_Eye':1, 'R_Eye':2, 'L_Ear':3, 'R_Ear':4, 'L_Shoulder':5, 'R_Shoulder':6, 'L_Elbow':7, 'R_Elbow':8, 'L_Wrist':9,
            'R_Wrist':10, 'L_Hip':11, 'R_Hip':12, 'L_Knee':13, 'R_Knee':14, 'L_Ankle':15, 'R_Ankle':16, 'Pelvis':17, 'Neck':18)]
    h36m : ['Pelvis':0, 'R_Hip':1, 'R_Knee':2, 'R_Ankle':3, 'L_Hip':4, 'L_Knee':5, 'L_Ankle':6, 'Torso':7, 'Neck':8, 'Nose':9, 'Head':10,
        'L_Shoulder':11, 'L_Elbow':12, 'L_Wrist':13, 'R_Shoulder':14, 'R_Elbow':15, 'R_Wrist':16]
    """
    B, T = coco_joint.shape[:2]
    h36m_joint = torch.zeros((T, 16, 2), dtype=coco_joint.dtype, device=coco_joint.device)
    
    h36m_joint[..., 0, :] = coco_joint[..., 17, :]
    h36m_joint[..., 1, :] = coco_joint[..., 12, :]
    h36m_joint[..., 2, :] = coco_joint[..., 14, :]
    h36m_joint[..., 3, :] = coco_joint[..., 16, :]
    h36m_joint[..., 4, :] = coco_joint[..., 11, :]
    h36m_joint[..., 5, :] = coco_joint[..., 13, :]
    h36m_joint[..., 6, :] = coco_joint[..., 15, :]
    h36m_joint[..., 7, :] = (coco_joint[..., 17, :] + coco_joint[..., 18, :])/2
    h36m_joint[..., 8, :] = coco_joint[..., 18, :]
    h36m_joint[..., 9, :] = coco_joint[..., 0, :]
    h36m_joint[..., 10, :] = (coco_joint[..., 1, :] + coco_joint[..., 2, :])/2
    h36m_joint[..., 11, :] = coco_joint[..., 5, :]
    h36m_joint[..., 12, :] = coco_joint[..., 7, :]
    h36m_joint[..., 13, :] = coco_joint[..., 9, :]
    h36m_joint[..., 14, :] = coco_joint[..., 6, :]
    h36m_joint[..., 15, :] = coco_joint[..., 8, :]
    h36m_joint[..., 16, :] = coco_joint[..., 10, :]

    return h36m_joint

def get_dataloader(args, dataset_names, is_train):
    dataset_split = 'TRAIN' if is_train else 'TEST'
    batch_per_dataset = cfg[dataset_split].batch_size // len(dataset_names)
    dataset_list, dataloader_list = [], []

    print(f"==> Preparing {dataset_split} Dataloader...")
    for name in dataset_names:
        dataset = eval(f'{name}.dataset')(dataset_split.lower(), args=args)
        print("# of {} {} data: {}".format(dataset_split, name, len(dataset)))
        dataloader = DataLoader(dataset,
                                batch_size=batch_per_dataset,
                                shuffle=cfg[dataset_split].shuffle,
                                num_workers=cfg.DATASET.workers,
                                pin_memory=False)
        dataset_list.append(dataset)
        dataloader_list.append(dataloader)

    if not is_train:
        return dataset_list, dataloader_list
    else:
        trainset_loader = MultipleDatasets(dataset_list, make_same_len=True)
        batch_generator = DataLoader(dataset=trainset_loader, \
                          batch_size=batch_per_dataset * len(dataset_names), \
                          shuffle=cfg[dataset_split].shuffle, \
                          num_workers=cfg.DATASET.workers, pin_memory=False)
        return dataset_list, batch_generator

def prepare_network(args, load_dir='', is_train=True):
    # Dataset loader
    dataset_names = cfg.DATASET.train_list if is_train else cfg.DATASET.test_list
    dataset_list, dataloader = get_dataloader(args, dataset_names, is_train)
    print(f"==> Dataset load done!")

    model, criterion, optimizer, lr_scheduler = None, None, None, None
    loss_history, test_error_history = [], {'surface': [], 'joint': []}

    main_dataset = dataset_list[0]
    J_regressor = eval(f'torch.Tensor(main_dataset.joint_regressor_{cfg.DATASET.input_joint_set})')
    
    # Model
    if is_train or load_dir:
        print(f"==> Preparing {cfg.MODEL.name} MODEL...")
        model = models.model.Model()
        print('# of model parameters: {}'.format(count_parameters(model)))

    # For training
    if is_train:
        criterion = get_loss_dict()
        optimizer = get_optimizer(model=model)
        lr_scheduler = get_scheduler(optimizer=optimizer)

    # Load checkpoint
    if load_dir and (not is_train or args.resume_training):
        print('==> Loading checkpoint')
        checkpoint = load_checkpoint(load_dir=load_dir, pick_best=(cfg.MODEL.name == 'PoseEst'))
        model.load_state_dict(checkpoint['model_state_dict'])

        if is_train:
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            curr_lr = 0.0

            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']

            lr_state = checkpoint['scheduler_state_dict']
            # update lr_scheduler
            lr_state['milestones'], lr_state['gamma'] = Counter(cfg.TRAIN.lr_step), cfg.TRAIN.lr_factor
            lr_scheduler.load_state_dict(lr_state)

            loss_history = checkpoint['train_log']
            test_error_history = checkpoint['test_log']
            cfg.TRAIN.begin_epoch = checkpoint['epoch'] + 1
            print('===> resume from epoch {:d}, current lr: {:.0e}, milestones: {}, lr factor: {:.0e}'
                  .format(cfg.TRAIN.begin_epoch, curr_lr, lr_state['milestones'], lr_state['gamma']))

    return dataloader, dataset_list, model, criterion, optimizer, lr_scheduler, loss_history, test_error_history


class Trainer:
    def __init__(self, args, load_dir):
        self.batch_generator, self.dataset_list, self.model, self.loss, self.optimizer, self.lr_scheduler, self.loss_history, self.error_history\
            = prepare_network(args, load_dir=load_dir, is_train=True)

        self.main_dataset = self.dataset_list[0]
        self.print_freq = cfg.TRAIN.print_freq

        self.J_regressor = eval(f'torch.Tensor(self.main_dataset.joint_regressor_{cfg.DATASET.target_joint_set}).cuda()')

        self.model = self.model.cuda()

        self.normal_weight = cfg.MODEL.normal_loss_weight
        self.edge_weight = cfg.MODEL.edge_loss_weight
        self.joint_weight = cfg.MODEL.joint_loss_weight
        self.edge_add_epoch = cfg.TRAIN.edge_loss_start
        self.shape_weight = cfg.MODEL.shape_weight
        self.pose_weight = cfg.MODEL.pose_weight
        self.trans_weight = cfg.MODEL.trans_weight

        self.seqlen = cfg.DATASET.seqlen

    def train(self, epoch):
        self.model.train()

        lr_check(self.optimizer, epoch)

        running_loss = 0.0
        batch_generator = tqdm(self.batch_generator)
        for i, (inputs, targets, meta) in enumerate(batch_generator):
            input_pose, input_feat = inputs['pose2d'], inputs['img_feature'].cuda()
            gt_lift3dpose, gt_reg3dpose = targets['lift_pose3d'], targets['reg_pose3d'].cuda()
            input_pose, gt_lift3dpose = COCO2H36M(input_pose), COCO2H36M(gt_lift3dpose)

            gt_mesh, gt_pose, gt_shape, gt_trans = \
                  targets['mesh'].cuda(), targets['pose'].cuda(), targets['shape'].cuda(), targets['trans'].cuda()
            val_lift3dpose, val_reg3dpose, val_mesh, val_pose, val_shape, val_trans =\
                  meta['lift_pose3d_valid'].cuda(), meta['reg_pose3d_valid'].cuda(), meta['mesh_valid'].cuda(), meta['pose_valid'].cuda(), meta['shape_valid'].cuda(), meta['trans_valid'].cuda() 

            # keypoint  : pred_kp3d(49x3), pred_kp2d(49x3), pred_lift3d(19x3)
            # SMPL      : pred_mesh(6890x3), pred_pose(24x3), pred_shape(10)
            lift3d_pos, pred_global, pred, mask_ids = self.model(input_feat, input_pose, is_train=True, J_regressor=self.J_regressor)

            pred_kp3d_global = torch.matmul(self.J_regressor[None, :, :], pred_global[0] * 1000)    # mm2m
            pred_kp3d = torch.matmul(self.J_regressor[None, :, :], pred[0] * 1000)                  # mm2m
            loss_kp3d = self.joint_weight * self.loss['L2'](pred_kp3d_global, gt_reg3dpose, val_reg3dpose, mask_ids) + \
                self.joint_weight * self.loss['L2'](pred_kp3d, gt_reg3dpose, val_reg3dpose)

            loss_lift3d = self.joint_weight * self.loss['L2'](lift3d_pos, gt_lift3dpose, val_lift3dpose)
            loss_mesh = self.loss['L1'](pred_global[0], gt_mesh, val_mesh, mask_ids)+\
                self.loss['L1'](pred[0], gt_mesh, val_mesh)
            loss_pose = self.pose_weight * self.loss['L1'](pred_global[1], gt_pose, val_pose, mask_ids)+\
                self.pose_weight * self.loss['L1'](pred[1], gt_pose, val_pose)
            loss_shape = self.shape_weight * self.loss['L1'](pred_global[2], gt_shape, val_shape, mask_ids)+\
                self.shape_weight * self.loss['L1'](pred[2], gt_shape, val_shape)
            loss_trans = self.trans_weight * self.loss['L1'](pred_global[3], gt_trans, val_trans, mask_ids)+\
                self.trans_weight * self.loss['L1'](pred[3], gt_trans, val_trans)
            
            loss = loss_kp3d + loss_lift3d + loss_mesh + loss_pose + loss_shape + loss_trans

            # update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log
            running_loss += float(loss.detach().item())

            if i % self.print_freq == 0:
                loss1, loss2, loss4, loss5 = loss_mesh.detach(), loss2.detach(), loss_kp3d.detach(), loss5.detach()
                loss3 = loss3.detach() if epoch > self.edge_add_epoch else 0
                loss6 = loss_lift3d.detach()
                batch_generator.set_description(f'Epoch{epoch}_({i}/{len(batch_generator)}) => '
                                                f'vertice loss: {loss1:.4f} '
                                                f'mesh->3d joint loss: {loss4:.4f} '
                                                f'lift joint loss: {loss6:.4f} ')

        self.loss_history.append(running_loss / len(batch_generator))

        print(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f}')


class Tester:
    def __init__(self, args, load_dir=''):
        self.val_loader, self.val_dataset, self.model, _, _, _, _, _ = \
            prepare_network(args, load_dir=load_dir, is_train=False)

        self.val_loader, self.val_dataset = self.val_loader[0], self.val_dataset[0]
        self.print_freq = cfg.TRAIN.print_freq

        self.J_regressor = eval(f'torch.Tensor(self.val_dataset.joint_regressor_{cfg.DATASET.target_joint_set}).cuda()')

        if self.model:
            self.model = self.model.cuda()

        # initialize error value
        self.surface_error = 9999.9
        self.joint_error = 9999.9

    def test(self, epoch, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()

        surface_error = 0.0
        joint_error = 0.0

        result = []
        eval_prefix = f'Epoch{epoch} ' if epoch else ''
        loader = tqdm(self.val_loader)
        with torch.no_grad():
            for i, (inputs, targets, meta) in enumerate(loader):
                input_pose, input_feat = inputs['pose2d'].cuda(), inputs['img_feature'].cuda()
                gt_reg3dpose = targets['reg_pose3d'].cuda()
                gt_mesh = targets['mesh'].cuda()
                input_pose = COCO2H36M(input_pose)

                lift3d_pos, pred_global, pred, mask_ids = self.model(input_feat, input_pose, is_train=True, J_regressor=self.J_regressor)
                pred_mesh, gt_mesh = pred[0] * 1000, gt_mesh * 1000

                pred_pose = torch.matmul(self.J_regressor[None, :, :], pred_mesh)

                j_error, s_error = self.val_dataset.compute_both_err(pred_mesh, gt_mesh, pred_pose, gt_reg3dpose)
                
                if i % self.print_freq == 0:
                    loader.set_description(f'{eval_prefix}({i}/{len(self.val_loader)}) => surface error: {s_error:.4f}, joint error: {j_error:.4f}')

                joint_error += j_error
                surface_error += s_error

                # Final Evaluation
                if (epoch == 0 or epoch == cfg.TRAIN.end_epoch):
                    pred_mesh, target_mesh = pred_mesh.detach().cpu().numpy(), gt_mesh.detach().cpu().numpy()
                    pred_pose, gt_pose3d = pred_pose.detach().cpu().numpy(), gt_pose3d.detach().cpu().numpy()
                    for j in range(len(input_pose)):
                        out = {}
                        out['mesh_coord'], out['mesh_coord_target'] = pred_mesh[j], target_mesh[j]
                        out['joint_coord'], out['joint_coord_target'] = pred_pose[j], gt_pose3d[j]
                        result.append(out)

            self.surface_error = surface_error / len(self.val_loader)
            self.joint_error = joint_error / len(self.val_loader)
            
            print(f'{eval_prefix}MPVPE: {self.surface_error:.2f}, MPJPE: {self.joint_error:.2f}')

            # Final Evaluation
            if (epoch == 0 or epoch == cfg.TRAIN.end_epoch):
                self.val_dataset.evaluate(result)