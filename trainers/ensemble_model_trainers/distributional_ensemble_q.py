import code

import torch
from torch.nn import functional as F
import numpy as np
from models import *
from components.memory import EnsembleReplayMemory
from trainers.ensemble_model_trainers.ensemble_q import EnsembleBase


class Distributional_EnsembleBase(EnsembleBase):

    def init_model(self):
        print("init Distributional Ensemble Q model")
        ModelClass = Distributional_Q_Network
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

    def greedy_action(self, obs):
        with torch.no_grad():
            obs = obs.view(1, 4, 84, 84)
            q = self.q_learner(obs)
            qs = self.dist_to_value(q).mean(1)
            if self.debug:
                print(qs)
            return qs.argmax(1).item()

    def dist_to_value(self, p):
        return (self.z.expand_as(p) * p).sum(-1)

    def ucb(self, obs):
        with torch.no_grad():
            qdist = self.q_learner(obs.view(1, 4, 84, 84))
            qs = self.dist_to_value(qdist)
            bonus = qs.mean(1) + self.args.lpuct * qs.std(1)
            return bonus.argmax(1).item()

    def state_value_pred(self, s):
        with torch.no_grad():
            p = self.q_learner(s.view(1, 4, 84, 84))
            qs = self.dist_to_value(p)  # (1, K, d)
            return qs.mean(1).view(-1).max().item()

    def target_estimate(self, s):
        self.print("warning: distributional ensemble target_estimate not implemented")
        return -1

    def learn_on_batch(self, idxs, s, a, r, s_, d, w):

        B, K, A, M = self.args.batch_size, self.args.ensemble_size, self.expl_env.action_space(), self.args.natoms
        z, delta_z = self.z, self.delta_z
        device = self.args.device

        self.optimizer.zero_grad()
        ps_ = self.target_estimate(s_)
        # Tz
        r, done = r.view(B*K, 1), d.view(B*K, 1)
        # print(r.shape, done.shape)
        Tz = r + (1 - done) * (self.args.discount ** self.args.multi_step) * z.unsqueeze(0)
        assert Tz.shape == torch.Size([B*K, M])
        Tz = Tz.clamp(min=z[0], max=z[-1])  # Clamp between supported values
        # Compute L2 projection of Tz onto fixed support z
        b = (Tz - z[0]) / delta_z  # b = (Tz - Vmin) / Δz
        l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
        # Fix disappearing probability mass when l = b = u (b is int)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (M - 1)) * (l == u)] += 1
        # Distribute probability of Tz
        ps_a = torch.zeros(B*K, M, device=device)
        offset = torch.linspace(0, ((B*K - 1) * M), B*K).unsqueeze(1).expand(B*K, M).to(device).int()
        ps_a.view(-1).index_add_(0, (l + offset).view(-1), (ps_ * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
        ps_a.view(-1).index_add_(0, (u + offset).view(-1), (ps_ * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        a = a.view(B, K, 1, 1)
        log_ps = self.forward_learners(s, True)
        # self.print(log_ps.shape, a.shape)
        log_psa = torch.gather(log_ps, dim=2, index=a.repeat(1,1,1,M)).squeeze()

        if self.debug:
            for i in range(B):
                for j in range(K):
                    a_select = a[i][j][0][0]
                    for k in range(M):
                        assert torch.allclose(log_ps[i, j, a_select, k], log_psa[i, j, k])
                        #  (log_ps[i, j, a_select, k], log_psa[i, j, k])
        # self.print(ps_a.shape, log_psa.shape)
        loss = -torch.sum(ps_a.view(B, K, M) * log_psa, dim=-1)
        IS_crossentropyloss = (w * loss).mean()
        IS_crossentropyloss.backward()
        self.optimizer.step()
        # PER update
        abs_loss = torch.abs(loss).detach().permute(1,0).cpu().numpy()
        self.update_priorities(idxs, abs_loss)
        with torch.no_grad():
            self.exp_path['batch_stats[t|num_done|pred_mean|q_loss|target_mean]'].csv_writerow(
                [self.t, d.sum().item(), self.dist_to_value(torch.exp(log_ps)).mean().item(), IS_crossentropyloss.item(), self.dist_to_value(ps_a).mean().item()])


