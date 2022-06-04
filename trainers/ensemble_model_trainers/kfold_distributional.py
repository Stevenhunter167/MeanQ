import code

import torch
import numpy as np
from trainers.ensemble_model_trainers.distributional_ensemble_q import Distributional_EnsembleBase


class Kfold_Distributional(Distributional_EnsembleBase):

    def target_estimate(self, s):
        with torch.no_grad():
            B, K, d, M = self.args.batch_size, self.args.ensemble_size, self.expl_env.action_space(), self.args.natoms
            z, delta_z = self.z, self.delta_z
            ps_ = self.q_target(s.view(B*K, 4, 84, 84))  # ps_ (B*K, K, d, M)
            # print("ps_", ps_.shape)
            qs_ = (z.expand_as(ps_) * ps_).sum(-1)  # q values (B*K, K, d)
            a_star = (qs_.sum(1, keepdim=True) - qs_).argmax(-1, keepdim=True) # best actions (B*K, 1)
            # print("a*", a_star.shape)
            psa = torch.gather(ps_, dim=2, index=a_star.unsqueeze(-1).repeat(1,1,1,M))  # (B*K, K, 1, M)
            psa = psa.squeeze()  # (B*K, K, M)
            # print("psa", psa.shape)
            # mean over distributions
            psa = psa.mean(1)
            return psa

