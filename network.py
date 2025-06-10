import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from read_dataset import BananaDataset
from torchvision import transforms
import numpy as np

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T  # Steps
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, noisy_coords, floor_plan, depth, cam_position, cam_rotation, cam_view):
        """
        Args：
        - noisy_coords：Noised 3D coordinates (batch_size, 3)
        - floor_plan：2D layout (batch_size, 1, 256, 256)
        - cam_position：Camera position (batch_size, 3)
        - cam_rotation：Camera rotation (batch_size, 3)
        - cam_view：RGB-D image from camera (batch_size, 3, 256, 256)
        - t：Timestep
        """

        # print("DDPM Class")
        # print(noisy_coords.shape)
        # print(floor_plan.shape)
        # print(cam_position.shape)
        # print(cam_rotation.shape)
        # print(cam_view.shape)

        # Choose timestep t
        _ts = torch.randint(1, self.n_T+1, (noisy_coords.shape[0],)).to(self.device)

        # Generate Gaussian noise
        noise = torch.randn_like(noisy_coords)

        # Compute x_t in forward diffusion
        coords_t = (
            self.sqrtab[_ts, None] * noisy_coords
            + self.sqrtmab[_ts, None] * noise
        )  # x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ϵ

        # Generate context_mask
        context_mask = torch.bernoulli(torch.zeros(noisy_coords.shape[0]).to(self.device) + self.drop_prob)

        # Estimate denoised coordinate
        pred_noise = self.nn_model(coords_t, floor_plan, depth, cam_position, cam_rotation, cam_view, _ts / self.n_T, context_mask)

        # Compute MSE loss
        return self.loss_mse(noise, pred_noise)

    def sample(self, n_sample, floor_plan, depth, cam_position, cam_rotation, cam_view, device, guide_w=0.0):
        """
        - From noise x_T keep denoising to get 3D coordinate
        - floor_plan, cam_view, cam_position, cam_rotation as conditioning
        """
        coords_i = torch.randn(n_sample, 3).to(device)

        context_mask = torch.zeros(n_sample).to(device)

        floor_plan = floor_plan.repeat(n_sample, 1, 1, 1)
        cam_view = cam_view.repeat(n_sample, 1, 1, 1)
        depth = depth.repeat(n_sample, 1, 1, 1)
        cam_position = cam_position.repeat(n_sample, 1)
        cam_rotation = cam_rotation.repeat(n_sample, 1)

        # Double the batch
        floor_plan = floor_plan.repeat(2, 1, 1, 1)
        cam_view = cam_view.repeat(2, 1, 1, 1)
        depth = depth.repeat(2, 1, 1, 1)

        cam_position = cam_position.repeat(2, 1)
        cam_rotation = cam_rotation.repeat(2, 1)

        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.  # Makes second half of batch context free

        coords_i_store = []
        # Denoise
        for i in range(self.n_T, 0, -1):
            # print(f'sampling timestep {i}', end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1)

            coords_i = coords_i.repeat(2, 1)
            t_is = t_is.repeat(2, 1)

            z = torch.randn(n_sample, 3).to(device) if i > 1 else 0

            # Prediction
            # print(coords_i.shape, floor_plan.shape, cam_position.shape, cam_rotation.shape, cam_view.shape, t_is.shape, context_mask.shape)
            eps = self.nn_model(coords_i, floor_plan, depth, cam_position, cam_rotation, cam_view, t_is, context_mask)

            # Classifier-Free Guidance
            eps1 = eps[:n_sample]  # Predictions with context
            eps2 = eps[n_sample:]  # Predictions without context
            eps = (1 + guide_w) * eps1 - guide_w * eps2  # guidance

            coords_i = coords_i[:n_sample]
            coords_i = (
                self.oneover_sqrta[i] * (coords_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

            # if i == 0 or i == 1 or i == 50 or i == 100 or i == 250 or i == 251 or i == 500:
            #     print(f'sampling timestep {i}, coords_i: {coords_i}')

            if i%20==0 or i==self.n_T or i<8:
                coords_i_store.append(coords_i.detach().cpu().numpy())

        coords_i_store = np.array(coords_i_store)

        return coords_i, coords_i_store


class Estimator(nn.Module):
    def __init__(self, n_feat=128, out_dim=3):
        super(Estimator, self).__init__()
        self.n_feat = n_feat

        # noisy_coords embedding
        self.init_conv_noisy_coords1 = EmbedFC(3, n_feat // 2)
        self.init_conv_noisy_coords2 = EmbedFC(n_feat // 2, n_feat)

        # RGBD (4 channels) extractor
        self.rgbd_extractor = models.resnet18(pretrained=True)
        self.rgbd_extractor.fc = nn.Identity()  # 512-dimensional features

        # modify conv1 to accept 4 channels input
        self.rgbd_extractor.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.rgbd_extractor.conv1.weight, mode='fan_out', nonlinearity='relu')

        # floor_plan feature extractor
        self.floor_plan_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        # Feature embedding layers
        self.fembed = EmbedFC(32 * 7 * 7 + n_feat, 2 * n_feat)

        self.contextembed1 = EmbedFC(256, 2 * n_feat)
        self.contextembed2 = EmbedFC(256, n_feat)
        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, n_feat)

        self.fusion = nn.Sequential(
            nn.Linear(512 + 3 + 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # output layers
        self.out_fc1 = nn.Sequential(
            nn.Linear(n_feat * 2, n_feat),
            nn.ReLU(),
            nn.Linear(n_feat, n_feat)
        )

        self.out_fc2 = nn.Sequential(
            nn.Linear(n_feat, n_feat),
            nn.ReLU(),
            nn.Linear(n_feat, n_feat // 2)
        )

        self.out_fc3 = nn.Sequential(
            nn.Linear(n_feat // 2 + n_feat, n_feat),
            nn.ReLU(),
            nn.Linear(n_feat, n_feat)
        )

        self.out_conv = nn.Sequential(
            nn.Conv1d(n_feat, n_feat // 2, 1),
            nn.Conv1d(n_feat // 2, n_feat // 4, 1),
            nn.BatchNorm1d(n_feat // 4),
            nn.ReLU(),
            nn.Conv1d(n_feat // 4, n_feat // 8, 1),
            nn.BatchNorm1d(n_feat // 8),
            nn.ReLU(),
            nn.Conv1d(n_feat // 8, out_dim, 1)
        )

    def forward(self, noisy_coords, floor_plan, depth, cam_position, cam_rotation, cam_view, t, context_mask):
        # noisy coordinates embedding
        n1 = self.init_conv_noisy_coords1(noisy_coords)
        n2 = self.init_conv_noisy_coords2(n1)

        # print(floor_plan.shape)

        # Concatenate RGB-D (cam_view + depth) into 4-channel data
        rgbd = torch.cat((cam_view, depth), dim=1)  # (B, 4, H, W)

        context_mask_reshape = context_mask[:, None, None, None].repeat(1, rgbd.shape[1], rgbd.shape[2],
                                                                        rgbd.shape[3])
        context_mask_floor = context_mask[:, None, None, None].repeat(1, floor_plan.shape[1], floor_plan.shape[2],
                                                                        floor_plan.shape[3])

        # print(context_mask_floor.shape)

        rgbd_masked = rgbd * (-1 * (1 - context_mask_reshape))
        floor_plan_masked = floor_plan * (-1 * (1 - context_mask_floor))

        # print(floor_plan_masked.shape)

        # Extract RGBD_seg features
        rgbd_feat = self.rgbd_extractor(rgbd_masked)

        # Extract floor_plan features
        floor_plan_feat = self.floor_plan_extractor(floor_plan_masked).view(floor_plan_masked.size(0), -1)

        # Combine coordinate and floor plan features
        vec_f = torch.cat([n2, floor_plan_feat], dim=1)
        f_embed = self.fembed(vec_f)

        # Apply context masks on camera parameters
        context_mask_params = context_mask[:, None].repeat(1, cam_position.shape[1])
        cam_position_masked = cam_position * (-1 * (1 - context_mask_params))
        cam_rotation_masked = cam_rotation * (-1 * (1 - context_mask_params))

        # Fusion features (RGBD_seg + camera params)
        fusion_feat = torch.cat([rgbd_feat.view(rgbd_feat.shape[0], -1),
                                 cam_position_masked,
                                 cam_rotation_masked], dim=1)

        fusion_vector = self.fusion(fusion_feat)

        # context and time embedding
        cemb1 = self.contextembed1(fusion_vector)
        temb1 = self.timeembed1(t)
        cemb2 = self.contextembed2(fusion_vector)
        temb2 = self.timeembed2(t)

        # Feature refinement
        v1 = self.out_fc1(cemb1 * f_embed + temb1)
        v2 = self.out_fc2(cemb2 * v1 + temb2)
        v3 = self.out_fc3(torch.cat((v2, n2), dim=1)).unsqueeze(-1)

        # output coordinates
        out = self.out_conv(v3).squeeze(-1)

        return out