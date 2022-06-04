import code

import torch
import numpy as np

from models.ensemble import Ensemble
from components.memory import EnsembleReplayMemory
from models.q_networks import Distributional_Noisy_Q_Network
from trainers.ensemble_model_trainers.kfold_distributional import Kfold_Distributional

class Kfold_Noisy(Kfold_Distributional):

    def init_model(self):
        print("init Distributional Noisy Ensemble Q model")
        ModelClass = Distributional_Noisy_Q_Network
        self.q_learner = Ensemble(ModelClass, self.args.ensemble_size,
                                  action_size=self.expl_env.action_space(),
                                  atoms=self.args.natoms).to(self.args.device)
        self.q_target = Ensemble(ModelClass, self.args.ensemble_size,
                                  action_size=self.expl_env.action_space(),
                                  atoms=self.args.natoms).to(self.args.device)
        self.optimizer = torch.optim.Adam(
            params=self.q_learner.parameters(),
            lr=self.args.lr)
        self.q_target.load_state_dict(self.q_learner.state_dict())
        self.replay_buffer = EnsembleReplayMemory(self.args)

        self.z = torch.linspace(self.args.vmin, self.args.vmax, self.args.natoms).to(self.args.device)
        self.delta_z = (self.args.vmax - self.args.vmin) / (self.args.natoms - 1)

    def ucb(self, obs):
        self.q_learner.reset_noise()
        return super().ucb(obs)



class Kfold_Noisy_Mean_Target(Kfold_Noisy):

    def target_estimate(self, s):
        with torch.no_grad():
            R = 3  # temp set hyp
            B, K, d, M = self.args.batch_size, self.args.ensemble_size, self.expl_env.action_space(), self.args.natoms
            z, delta_z = self.z, self.delta_z
            ps_ = []
            s_reshaped = s.view(B*K, 4, 84, 84)
            for _ in range(R):
                ps_.append(self.q_target(s_reshaped))  # ps_ (B*K, K*R, d, M)
                self.q_target.reset_noise()
            ps_ = torch.cat(ps_, dim=1)
            # print("ps_", ps_.shape)
            qs_ = (z.expand_as(ps_) * ps_).sum(-1)  # q values (B*K, K*R, d)
            a_star = (qs_.sum(1, keepdim=True) - qs_).argmax(-1, keepdim=True) # best actions (B*K, 1)
            # print("a*", a_star.shape)
            psa = torch.gather(ps_, dim=2, index=a_star.unsqueeze(-1).repeat(1,1,1,M))  # (B*K, K*R, 1, M)
            psa = psa.squeeze()  # (B*K, K, M)
            # print("psa", psa.shape)
            # mean over distributions
            psa = psa.mean(1)
            return psa