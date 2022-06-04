import torch
from models.q_networks import Dropout_Q_Network, Noisy_Q_Network
from trainers.freq_dqn import FreqDQN


class DropoutDQN(FreqDQN):

    def init_model(self):
        self.print("init DropoutDQN model")
        ModelClass = eval(self.args.network_type)
        self.q_learner = ModelClass(self.expl_env.action_space()).to(self.args.device)
        self.q_target = ModelClass(self.expl_env.action_space()).to(self.args.device)
        self.optimizer = torch.optim.Adam(
            params=self.q_learner.parameters(),
            lr=self.args.lr)
        self.q_target.load_state_dict(self.q_learner.state_dict())

    def V(self, s_):
        with torch.no_grad():
            outs = []
            for _ in range(self.args.ensemble_size):
                self.q_target.reset_noise()
                max_a_qsa, a_indicies = self.q_target(s_).max(dim=1)
                max_a_qsa = torch.unsqueeze(max_a_qsa, dim=1)
                outs.append(max_a_qsa)
            return torch.cat(outs, dim=1).mean(dim=1, keepdim=True)

    def expl_action(self, obs):
        self.q_learner.eval()
        res = super().expl_action(obs)
        self.q_learner.train()
        return res
    
    def eval_action(self, obs):
        self.q_learner.eval()
        res = super().eval_action(obs)
        self.q_learner.train()
        return res
