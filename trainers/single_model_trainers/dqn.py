import code

import torch
import numpy as np
from models.q_networks import Q_Network, Dueling_Q_Network, UnitQ_Network
from trainers.base_trainer import BaseTrainer

class DQN(BaseTrainer):

    def init_model(self):
        self.print("init DQN model")
        if 'dueling' in self.args and self.args.dueling:
            ModelClass = Dueling_Q_Network
        else:
            ModelClass = Q_Network
        self.q_learner = ModelClass(self.expl_env.action_space()).to(self.args.device)
        self.q_target = ModelClass(self.expl_env.action_space()).to(self.args.device)
        self.optimizer = torch.optim.Adam(
            params=self.q_learner.parameters(),
            lr=self.args.lr)
        self.q_target.load_state_dict(self.q_learner.state_dict())

    def sample_batch(self, batch_size):
        return self.replay_buffer.sample(batch_size)

    def expl_action(self, obs) -> int:
        if np.random.rand() < self.epsilon():
            return np.random.randint(self.expl_env.action_space())
        with torch.no_grad():
            obs = torch.unsqueeze(obs, dim=0)
            q = torch.squeeze(self.q_learner(obs), dim=0)
            a = torch.argmax(q).item()
            return a

    def eval_action(self, obs) -> int:
        with torch.no_grad():
            obs = torch.unsqueeze(obs, dim=0)
            q = torch.squeeze(self.q_learner(obs), dim=0)
            a = torch.argmax(q).item()
            return a

    def V(self, s_):
        with torch.no_grad():
            max_a_qsa, a_indicies = self.q_target(s_).max(dim=1)
            max_a_qsa = torch.unsqueeze(max_a_qsa, dim=1)
            return max_a_qsa

    def before_learn_on_batch(self):
        # PER (prioritized experience replay) anneal
        self.replay_buffer.anneal_priority_weight()

    def learn_on_batch(self, treeidx, s, a, r, s_, d, w):

        # learn on batch
        a = torch.nn.functional.one_hot(a, num_classes=self.expl_env.action_space())
        r = torch.unsqueeze(r, dim=1)
        w = torch.unsqueeze(w, dim=1)
        self.optimizer.zero_grad()
        # compute loss
        target = r + (1 - d) * self.args.discount * self.V(s_)
        pred = torch.unsqueeze(torch.sum(self.q_learner(s) * a, dim=1), dim=1)
        loss = (pred - target) ** 2
        # PER Importance Sampling
        IS_mseloss = torch.mean(loss * w)
        IS_mseloss.backward()
        self.optimizer.step()
        # PER update
        abs_loss = torch.sqrt(loss).detach().view(-1).cpu().numpy()
        self.replay_buffer.update_priorities(treeidx, abs_loss)

        # logging
        with torch.no_grad():
            self.exp_path['batch_stats[t|num_done|pred_mean|q_loss|target_mean]'].csv_writerow(
                [self.t, d.sum().item(), pred.mean().item(), IS_mseloss.item(), target.mean().item()])

    def after_learn_on_batch(self):
        # target network update
        if self.t % self.args.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_learner.state_dict())

    def state_value_pred(self, s):
        with torch.no_grad():
            s = s.view(1, 4, 84, 84)
            return self.V(s).mean().item()


class UnitDQN(DQN):

    def init_model(self):
        self.print("init UnitDQN model")
        ModelClass = UnitQ_Network
        self.q_learner = ModelClass(self.expl_env.action_space()).to(self.args.device)
        self.q_target = ModelClass(self.expl_env.action_space()).to(self.args.device)
        self.optimizer = torch.optim.Adam(
            params=self.q_learner.parameters(),
            lr=self.args.lr)
        self.q_target.load_state_dict(self.q_learner.state_dict())

