import code

import torch
import numpy as np

from models.ensemble import Ensemble
# from models.q_networks import Distributional_Noisy_Q_Network
# from components.memory import EnsembleReplayMemory
# from trainers.ensemble_model_trainers.distributional_ensemble_q import Distributional_EnsembleBase
from trainers.ensemble_model_trainers.mean_q_distributional import MeanQ_Noisy



class Double_MeanQ_Noisy(MeanQ_Noisy):

    def target_estimate(self, s):
            with torch.no_grad():
                B, K, d, M = self.args.batch_size, self.args.ensemble_size, self.expl_env.action_space(), self.args.natoms
                z, delta_z = self.z, self.delta_z
                ps_target = self.q_target(s.view(B*K, 4, 84, 84))  # ps_target (B*K, K, d, M)
                ps_learner = self.q_learner(s.view(B*K, 4, 84, 84))  # ps_learner (B*K, K, d, M)
                qs_learner = (z.expand_as(ps_learner) * ps_learner).sum(-1)  # qs_learner (B*K, K, d)
                a_star = qs_learner.sum(1, keepdim=True).argmax(-1, keepdim=True)  # best actions (B*K, 1)
                self.exp_path['a_star'].csv_writerow(
                    [self.t, a_star.view(-1).cpu().numpy().tolist()]
                )
                # print("a*", a_star.shape)
                psa = torch.gather(ps_target, dim=2, index=a_star.unsqueeze(-1).repeat(1,K,1,M))  # (B*K, K, 1, M)
                psa = psa.squeeze(2)  # (B*K, K, M)
                # print("psa", psa.shape)
                # mean over distributions
                psa = psa.mean(1)
                return psa
