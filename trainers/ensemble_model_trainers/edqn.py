import code

import torch
import numpy as np
from models import *
from components.memory import ReplayMemory
from trainers.ensemble_model_trainers.mean_q import MeanQ

class EDQN(MeanQ):

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
        self.replay_buffer = ReplayMemory(self.args)
        self.learners_t = None

    def ucb(self, obs):
        pass

    def sample_batch(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        idxs_k, states_k, actions_k, returns_k, next_states_k, nonterminals_k, weights_k = [], [], [], [], [], [], []
        for k in range(self.args.ensemble_size):
            idxs, states, actions, returns, next_states, nonterminals, weights = batch
            idxs_k.append(idxs)
            states_k.append(torch.unsqueeze(states, dim=1))
            actions_k.append(torch.unsqueeze(actions, dim=1))
            returns_k.append(returns.view(batch_size, 1, 1))
            next_states_k.append(torch.unsqueeze(next_states, dim=1))
            nonterminals_k.append(torch.unsqueeze(nonterminals, dim=1))
            weights_k.append(weights.view(batch_size, 1))
        states = torch.cat(states_k, dim=1)
        actions = torch.cat(actions_k, dim=1)
        returns = torch.cat(returns_k, dim=1)
        next_states = torch.cat(next_states_k, dim=1)
        dones = torch.cat(nonterminals_k, dim=1)
        weights = torch.cat(weights_k, dim=1)
        return idxs_k, states, actions, returns, next_states, dones, weights

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
        # code.interact(local=locals())
        IS_mseloss = torch.mean(loss * w)
        IS_mseloss.backward()
        self.optimizer.step()
        # PER update
        abs_loss = torch.sqrt(loss).detach().mean(1).cpu().numpy()
        self.replay_buffer.update_priorities(idxs[0], abs_loss)
        # logging
        with torch.no_grad():
            self.exp_path['batch_stats[t|num_done|pred_mean|q_loss|target_mean]'].csv_writerow(
                [self.t, d.sum().item(), q_pred.mean().item(), IS_mseloss.item(), targets.mean().item()])
