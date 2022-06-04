import code

import torch
import numpy as np
from trainers.single_model_trainers.dqn import DQN

class DQN_NoTargetNet(DQN):

    def V(self, s_):
        with torch.no_grad():
            max_a_qsa, a_indicies = self.q_learner(s_).max(dim=1)
            max_a_qsa = torch.unsqueeze(max_a_qsa, dim=1)
            return max_a_qsa
