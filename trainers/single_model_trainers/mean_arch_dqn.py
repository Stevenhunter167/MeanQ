import code

import torch
import numpy as np
from models.ensemble import EnsembleMeanQTestArch
from trainers.single_model_trainers.dqn import DQN

class MeanArchDQN(DQN):

    def init_model(self):
        self.print("init Mean Arch DQN model")
        # self.q_learner = Q_Network(self.expl_env.action_space()).to(self.args.device)
        # self.q_target = Q_Network(self.expl_env.action_space()).to(self.args.device)
        self.q_learner = EnsembleMeanQTestArch(
            self.expl_env.action_space(), 
            self.args.ensemble_size).to(self.args.device)
        self.q_target = EnsembleMeanQTestArch(
            self.expl_env.action_space(), 
            self.args.ensemble_size).to(self.args.device)
        self.optimizer = torch.optim.Adam(
            params=self.q_learner.parameters(),
            lr=self.args.lr)
        self.q_target.load_state_dict(self.q_learner.state_dict())
