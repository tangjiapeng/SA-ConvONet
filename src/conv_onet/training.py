import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from src.common import (
    compute_iou, make_3d_grid, add_key,
)
from src.utils import visualize as vis
from src.training import BaseTrainer
import numpy as np

class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, device=None, input_type='pointcloud',
                 vis_dir=None, threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)
        
        batch_size = points.size(0)

        kwargs = {}
        
        # add pre-computed index
        inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
        # add pre-computed normalized coordinates
        points = add_key(points, data.get('points.normalized'), 'p', 'p_n', device=device)
        points_iou = add_key(points_iou, data.get('points_iou.normalized'), 'p', 'p_n', device=device)

        # Compute iou
        with torch.no_grad():
            p_out = self.model(points_iou, inputs, 
                               sample=self.eval_sample, **kwargs)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()

        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, voxels_occ.shape[1:])
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
        
        if 'pointcloud_crop' in data.keys():
            # add pre-computed index
            inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
            inputs['mask'] = data.get('inputs.mask').to(device)
            # add pre-computed normalized coordinates
            p = add_key(p, data.get('points.normalized'), 'p', 'p_n', device=device)

        c = self.model.encode_inputs(inputs)

        kwargs = {}
        # General points
        logits = self.model.decode(p, c, **kwargs).logits
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
        loss = loss_i.sum(-1).mean()

        return loss

    ## sign-agnostic optimization of a whole object/scene during inference 
    def sign_agnostic_optim_step(self, data, state_dict, batch_size=16, npoints1=1024, npoints2=1024, sigma=0.1, num_points_input=None):
        ''' Performs a sign-agnostic optimization step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_sign_agnostic_loss(data, batch_size, npoints1, npoints2, sigma, num_points_input)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_sign_agnostic_loss(self, data, batch_size, npoints1, npoints2, sigma, num_points_input):
        ''' Computes the sign agnostic cross entropy loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        inputs = data.get('inputs')
        batch_size, npoints1, npoints2 = int(batch_size), int(npoints1), int(npoints2)

        # load input point clouds
        if num_points_input is None:
            batch_inputs = inputs.expand(batch_size, inputs.size(1), inputs.size(2))
        else:
            input_list = []
            for i in range(batch_size):
                index = np.random.randint(inputs.size(1), size=num_points_input)
                input_list.append(inputs[:,index, :])
            batch_inputs = torch.cat(input_list, dim=0).contiguous()

        # load query points and corresponding labels.
        batch_p = []
        batch_occ = []
        for i in range(batch_size):
            inputs_noise = sigma * np.random.normal(0, 1.0, size=inputs.cpu().numpy().shape)
            inputs_noise = torch.from_numpy(inputs_noise).type(torch.FloatTensor)
            inputs_noise = inputs + inputs_noise
            index1 = np.random.randint(inputs.size(1), size=npoints1)
            index2 = np.random.randint(inputs.size(1), size=npoints2)
            p = torch.cat([inputs[:, index1, :], inputs_noise[:, index2, :]], dim=1)
            occ = torch.cat([torch.ones((1, npoints1), dtype=torch.float32)*0.5, torch.ones((1, npoints2), dtype=torch.float32)], dim=1)
            batch_p.append(p)
            batch_occ.append(occ)
        batch_p = torch.cat(batch_p, dim=0).to(device)
        batch_occ = torch.cat(batch_occ, dim=0).to(device)


        c = self.model.encode_inputs(batch_inputs.to(device))
        kwargs = {}
        # General points
        logits = self.model.decode(batch_p, c, **kwargs).logits
        logits = logits.abs() # absolute value
        loss_i = F.binary_cross_entropy_with_logits(
            logits, batch_occ, reduction='none')
        loss = loss_i.sum(-1).mean()

        return loss

    ## sign-agnostic optimization of crop scenes during inference 
    def sign_agnostic_optim_cropscene_step(self, data, state_dict): 
        ''' Performs a sign-agnostic optimization step.
        Args:
            data (dict): data dictionary
        '''
        self.model.train(mode=True, freeze_norm=True, freeze_norm_affine=True)
        self.optimizer.zero_grad()
        loss = self.compute_sign_agnostic_cropscene_loss(data)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_sign_agnostic_cropscene_loss(self, data):
        ''' Computes the sign agnostic cross entropy loss.
        Args:
            data (dict): data dictionary
        '''
        assert 'pointcloud_crop' in data.keys()

        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
        
        # occ[occ==0] = self.threshold # for surface points
        occ[occ==0] = 0.5

        # add pre-computed index
        inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
        inputs['mask'] = data.get('inputs.mask').to(device)
        # add pre-computed normalized coordinates
        p = add_key(p, data.get('points.normalized'), 'p', 'p_n', device=device)

        # import ipdb; ipdb.set_trace()
        c = self.model.encode_inputs(inputs)
        logits = self.model.decode(p, c).logits

        logits = logits.abs() # absolute value
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
        loss = loss_i.sum(-1).mean()

        return loss