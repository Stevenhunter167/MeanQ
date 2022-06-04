import code

import torch
import numpy as np
from models import *
from components.memory import EnsembleReplayMemory
from trainers.base_trainer import BaseTrainer


class EnsembleBase(BaseTrainer):

    def init_model(self):
        print("init Ensemble Q model")
        ModelClass = Dueling_Q_Network if self.args.dueling else Q_Network
        self.q_learner = Ensemble(ModelClass, self.args.ensemble_size,
                                  action_size=self.expl_env.action_space()).to(self.args.device)
        self.q_target = Ensemble(ModelClass, self.args.ensemble_size,
                                  action_size=self.expl_env.action_space()).to(self.args.device)
        self.optimizer = torch.optim.Adam(
            params=self.q_learner.parameters(),
            lr=self.args.lr)
        self.q_target.load_state_dict(self.q_learner.state_dict())
        self.replay_buffer = EnsembleReplayMemory(self.args)
        self.learners_t = None

    # basic ensemble routines

    def expl_action(self, obs) -> int:
        # ucb exploration
        if 'lpuct' in self.args and self.args.lpuct is not None:
            return self.ucb(obs)
        # default e-greedy exploration
        else:
            if np.random.random() < self.epsilon():
                return np.random.randint(self.expl_env.action_space())
            return self.greedy_action(obs)

    def eval_action(self, obs) -> int:
        return self.greedy_action(obs)

    def greedy_action(self, obs):
        with torch.no_grad():
            obs = obs.view(1, 4, 84, 84)
            return torch.argmax(torch.mean(self.q_learner(obs), dim=1), dim=1).item()

    def ucb(self, obs):
        """ ucb exploration action """
        with torch.no_grad():
            obs = obs.view(1, 4, 84, 84)
            q_pred = self.q_learner(obs)
            q_mean = torch.mean(q_pred, dim=1)
            q_std = torch.std(q_pred, dim=1)
            bonus_value = q_mean + self.args.lpuct * q_std
            action = torch.argmax(bonus_value, dim=1).item()
        # if self.debug:
        #     self.print(f"ucb_expl t:{self.t} action:{action}")
        #     self.print(f"ucb_expl bonus: {bonus_value}")
        return action

    def before_learn_on_batch(self):
        # PER (prioritized experience replay) anneal
        self.replay_buffer.anneal_priority_weight()

    def select_learners(self):
        self.learners_t = [i for i in range(self.args.ensemble_size)]  # set learners for this grad update
        self.exp_path['debug']['learner_select[t|*learners]'].csv_writerow([self.t] + self.learners_t)

    def sample_batch(self, batch_size):
        self.select_learners()
        return self.replay_buffer.sample(batch_size, self.learners_t)

    def forward_learners(self, obs, *args, **kwargs):
        kwargs.update(dict(learners=self.learners_t))
        # self.print(kwargs)
        return self.q_learner(obs, *args, **kwargs)

    def update_priorities(self, idxs, priorities):
        """ most of these below are copied from memory.py """
        self.replay_buffer.update_priorities(idxs, priorities, self.learners_t)

    def after_learn_on_batch(self):
        # target network update
        if self.t % self.args.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_learner.state_dict())

    def state_value_pred(self, s):
        """ state value prediction """
        with torch.no_grad():
            # s = s.view(1, 1, 4, 84, 84).repeat(1, self.args.ensemble_size, 1, 1, 1)
            s = s.view(1, 4, 84, 84)
            self.learners_t = [i for i in range(self.args.ensemble_size)]
            qsa = self.forward_learners(s)
            best_action = qsa.mean(1).view(-1).argmax()
            vs = qsa[0, :, best_action].cpu()
            self.exp_path['debug']['trace[t|*learner_vinit]'].csv_writerow([self.t] + vs.numpy().tolist())
            return vs.mean().item()

    def target_estimate(self, s):
        self.print("warning: ensemble target_estimate not implemented")
        return -1

    def learn_on_batch(self, idxs, s, a, r, s_, d, w):
        """
        s (b, k, s)
        a (b, k, a)
        r (b, k, 1)
        s_(b, k, s)
        d (b, k, 1)
        """
        r = torch.squeeze(r)                                            # (b, k)
        d = torch.squeeze(d)                                            # (b, k)
        a = torch.nn.functional.one_hot(a, num_classes=self.expl_env.action_space())
        self.optimizer.zero_grad()
        # compute next state value
        next_state_value = self.target_estimate(s_)
        targets = r + (1 - d) * (self.args.discount ** self.args.multi_step) * next_state_value   # (b, k)
        # compute state action values
        q_pred = self.forward_learners(s)                               # (b, k, a)
        qsa_pred = torch.sum(q_pred * a, dim=2)                         # (b, k)
        # compute loss, optimize
        loss = (qsa_pred - targets) ** 2
        # PER Importance Sampling (IS)
        IS_mseloss = torch.mean(loss * w)
        IS_mseloss.backward()
        self.optimizer.step()
        # PER update
        abs_loss = torch.sqrt(loss).detach().permute(1,0).cpu().numpy()
        self.update_priorities(idxs, abs_loss)
        # logging
        with torch.no_grad():
            self.exp_path['batch_stats[t|num_done|pred_mean|q_loss|target_mean]'].csv_writerow(
                [self.t, d.sum().item(), q_pred.mean().item(), IS_mseloss.item(), targets.mean().item()])
