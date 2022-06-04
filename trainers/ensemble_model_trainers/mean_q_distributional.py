import code

import torch
import numpy as np
from trainers.ensemble_model_trainers.distributional_ensemble_q import Distributional_EnsembleBase
from models.ensemble import Ensemble
from components.memory import EnsembleReplayMemory
from models.q_networks import Distributional_Noisy_Q_Network #, DataEff


class MeanQ_Distributional(Distributional_EnsembleBase):

    def target_estimate(self, s):
        with torch.no_grad():
            B, K, d, M = self.args.batch_size, self.args.ensemble_size, self.expl_env.action_space(), self.args.natoms
            z, delta_z = self.z, self.delta_z
            ps_ = self.q_target(s.view(B*K, 4, 84, 84))  # ps_ (B*K, K, d, M)
            # print("ps_", ps_.shape)
            qs_ = (z.expand_as(ps_) * ps_).sum(-1)  # q values (B*K, K, d)
            a_star = qs_.sum(1, keepdim=True).argmax(-1, keepdim=True) # best actions (B*K, 1)
            # print("a*", a_star.shape)
            psa = torch.gather(ps_, dim=2, index=a_star.unsqueeze(-1).repeat(1,K,1,M))  # (B*K, K, 1, M)
            psa = psa.squeeze(2)  # (B*K, K, M)
            # print("psa", psa.shape)
            # mean over distributions
            psa = psa.mean(1)
            return psa


class MeanQ_Noisy(MeanQ_Distributional):

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


class MeanQ_Equiv_Dropout(MeanQ_Noisy):

    def select_learners(self):
        # self.learners_t = np.random.permutation(self.args.ensemble_size).tolist()  # set learners for this grad update
        self.learners_t = np.random.choice(
            self.args.ensemble_size,
            size=self.args.ensemble_size,
            replace=True).tolist()  # set learners for this grad update
        self.exp_path['debug']['learner_select[t|*learners]'].csv_writerow([self.t] + self.learners_t)


class MeanQ_SanityCheck(MeanQ_Noisy):

    def select_learners(self):
        self.learners_t = np.random.permutation(self.args.ensemble_size).tolist()  # set learners for this grad update
        self.exp_path['debug']['learner_select[t|*learners]'].csv_writerow([self.t] + self.learners_t)


class MeanQ_Kdrop3b(MeanQ_Noisy):

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
        # changes starts here
        # learners_t = np.random.choice(
        #     self.args.ensemble_size,
        #     size=(subset_size, self.args.ensemble_size),
        #     replace=True).tolist()
        learners_t = np.random.binomial(size=(5,5), n=1, p=1/5)
        selected_batches = (learners_t.sum(1) > 0)
        selected_batches = [i for i in range(self.args.ensemble_size) if selected_batches[i] > 0]
        allpreds = []
        for bi in selected_batches:
            s_bi = s[:,bi,:,:,:]
            lbi = [j for j in range(self.args.ensemble_size) if learners_t[bi][j] == 1]
            allpreds.append(self.q_learner(s_bi, learners=lbi).mean(1, keepdim=True))
        if len(allpreds) != 0:
            allpreds = torch.cat(allpreds, dim=1)
            log_ps = torch.log(allpreds)
            log_psa = torch.gather(log_ps, dim=2, index=a.repeat(1,1,1,M)[:, selected_batches, :, :]).squeeze()
            # self.print(ps_a.shape, log_psa.shape)
            loss = -torch.sum(ps_a.view(B, K, M)[:, selected_batches, :] * log_psa, dim=-1)  # (B, K)
            IS_crossentropyloss = (w[:, selected_batches] * loss).mean()
            IS_crossentropyloss.backward()
            self.optimizer.step()

            # PER update
            with torch.no_grad():
                log_ps_no_grad = self.forward_learners(s, True)
                log_psa_no_grad = torch.gather(log_ps_no_grad, dim=2, index=a.repeat(1, 1, 1, M)).squeeze()
                loss_no_grad = -torch.sum(ps_a.view(B, K, M) * log_psa_no_grad, dim=-1)
                abs_loss = torch.abs(loss_no_grad).detach().permute(1,0).cpu().numpy()
                self.update_priorities(idxs, abs_loss)

            with torch.no_grad():
                self.exp_path['batch_stats[t|num_done|pred_mean|q_loss|target_mean]'].csv_writerow(
                    [self.t, d.sum().item(), self.dist_to_value(torch.exp(log_ps)).mean().item(), IS_crossentropyloss.item(), self.dist_to_value(ps_a).mean().item()])