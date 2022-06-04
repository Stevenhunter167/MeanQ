import code

import torch
import numpy as np
from models.q_networks import Q_Network, Dueling_Q_Network
from trainers.single_model_trainers.dqn import DQN

class AvgDQN(DQN):

    def init_model(self):
        self.print("init DQN model")
        if 'dueling' in self.args and self.args.dueling:
            ModelClass = Dueling_Q_Network
        else:
            ModelClass = Q_Network
        self.q_learner = ModelClass(self.expl_env.action_space()).to(self.args.device)
        self.q_state_dict_cache = self.q_learner.state_dict()
        self.q_snapshots = [
            ModelClass(self.expl_env.action_space()).to(self.args.device)
            for _ in range(self.args.ensemble_size)]
        for k in range(self.args.ensemble_size):
            self.q_snapshots[k].load_state_dict(self.q_learner.state_dict())
        self.optimizer = torch.optim.Adam(
            params=self.q_learner.parameters(),
            lr=self.args.lr)

    def after_learn_on_batch(self):
        pass

    def learn_on_batch(self, treeidx, s, a, r, s_, d, w):
        super(AvgDQN, self).learn_on_batch(treeidx, s, a, r, s_, d, w)
        self.q_snapshots[self.t % self.args.ensemble_size].load_state_dict(self.q_state_dict_cache)
        self.q_state_dict_cache = self.q_learner.state_dict()

    @torch.no_grad()
    def V(self, s):
        qs = []
        for k in range(self.args.ensemble_size):
            qs.append(self.q_snapshots[k](s))
        v, _ = torch.stack(qs).mean(0).max(1)
        return v
