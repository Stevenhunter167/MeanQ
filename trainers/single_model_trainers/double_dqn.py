import torch
from trainers.single_model_trainers.dqn import DQN


class DoubleDQN(DQN):

    def V(self, s_):
        with torch.no_grad():
            best_actions = torch.unsqueeze(torch.argmax(self.q_learner(s_), dim=1), dim=1)
            max_a_qsa = torch.gather(self.q_target(s_), 1, best_actions)
            return max_a_qsa
