import code
import torch
from models.q_networks import Distributional_Noisy_Q_Network
from components.memory import ReplayMemory
from trainers.base_trainer import BaseTrainer
from trainers.single_model_trainers.dqn import DQN
from trainers.ensemble_model_trainers.double_mean_q import Double_MeanQ_Noisy
from models.q_networks import RainbowMeanArch

# class Rainbow(Double_MeanQ_Noisy):
#
#     def __init__(self, args):
#         assert args.ensemble_size == 1, "rainbow is a single network algorithm"
#         super().__init__(args)
#
#     def ucb(self, obs):
#         self.q_learner.reset_noise()
#         with torch.no_grad():
#             qdist = self.q_learner(obs.view(1, 4, 84, 84))  # B, 1, d, M
#             qs = self.dist_to_value(qdist)  # B, 1, d
#             action = qs.mean(1).argmax(1).item()
#             return action

# takaways
# 1. disable noise for eval loop
# 2. reset noise in target est
#


class Rainbow(DQN):

    def init_model(self):
        self.print('init rainbow')
        self.q_learner = Distributional_Noisy_Q_Network(
            self.expl_env.action_space(), self.args.natoms).to(self.args.device)
        self.q_target = Distributional_Noisy_Q_Network(
            self.expl_env.action_space(), self.args.natoms).to(self.args.device)
        self.optimizer = torch.optim.Adam(
            params=self.q_learner.parameters(),
            lr=self.args.lr)
        self.q_target.load_state_dict(self.q_learner.state_dict())
        self.replay_buffer = ReplayMemory(self.args)

        self.z = torch.linspace(self.args.vmin, self.args.vmax, self.args.natoms).to(self.args.device)
        self.delta_z = (self.args.vmax - self.args.vmin) / (self.args.natoms - 1)

    def dist_to_value(self, ps):
        return (self.z.expand_as(ps) * ps).sum(-1)

    def greedy_action(self, obs):
        with torch.no_grad():
            ps = self.q_learner(obs.view(1,4,84,84))
            qs = self.dist_to_value(ps)
            return qs.argmax(1).item()

    def expl_action(self, obs) -> int:
        self.q_learner.train()
        self.q_learner.reset_noise()
        return self.greedy_action(obs)

    def eval_action(self, obs) -> int:
        self.q_learner.eval()
        return self.greedy_action(obs)

    def target_estimate(self, s):
        with torch.no_grad():
            ps_target = self.q_target(s)
            actions = self.dist_to_value(self.q_learner(s)).argmax(1)
            self.q_target.reset_noise()
            ps_a = ps_target[range(self.args.batch_size), actions]
            return ps_a


    def learn_on_batch(self, treeidx, s, a, r, s_, d, w):
        B, M = self.args.batch_size, self.args.natoms
        # learner forward pass
        log_ps = self.q_learner(s, True)  # (B, d, M)
        log_psa = torch.gather(log_ps, dim=1, index=a.view(-1, 1, 1).repeat(1,1,M)).squeeze()
        # estimate target
        ps_a = self.target_estimate(s_)
        # Tz = R^n + (γ^n)z (accounting for terminal states)
        Tz = r.unsqueeze(1) + (1 - d) * (self.args.discount ** self.args.multi_step) * self.z.unsqueeze(0)
        Tz = Tz.clamp(min=self.args.vmin, max=self.args.vmax)  # Clamp between supported values
        # Compute L2 projection of Tz onto fixed support z
        b = (Tz - self.args.vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
        l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
        # Fix disappearing probability mass when l = b = u (b is int)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.args.natoms - 1)) * (l == u)] += 1
        # Distribute probability of Tz
        m = torch.zeros(B, M, dtype=torch.float32, device=self.args.device)
        offset = torch.linspace(0, ((B - 1) * M), B).unsqueeze(1).expand(B, M).to(self.args.device).int()
        m.view(-1).index_add_(0, (l + offset).view(-1),
                              (ps_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
        m.view(-1).index_add_(0, (u + offset).view(-1),
                              (ps_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)
        loss = -torch.sum(m * log_psa, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        self.optimizer.zero_grad()
        (w * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        self.optimizer.step()
        abs_loss = torch.abs(loss).detach().cpu().numpy()
        self.replay_buffer.update_priorities(treeidx, abs_loss)

        with torch.no_grad():
            self.exp_path['batch_stats[t|num_done|pred_mean|q_loss|target_mean]'].csv_writerow(
                [self.t, d.sum().item(), self.dist_to_value(torch.exp(log_psa)).mean().item(),
                 loss.mean().item(), self.dist_to_value(ps_a).mean().item()])

    def state_value_pred(self, s):
        with torch.no_grad():
            value, _ = self.dist_to_value(self.q_learner(s.view(1, 4, 84, 84))).max(1)
            return value.item()



class Rainbow_MeanArch_Trainer(Rainbow):

    def init_model(self):
        self.print('init rainbow, but mean over 5 models')

        self.q_learner = RainbowMeanArch(
            ensemble_size=self.args.ensemble_size,
            action_size=self.expl_env.action_space(),
            atoms=self.args.natoms).to(self.args.device)
        self.q_target = RainbowMeanArch(
            ensemble_size=self.args.ensemble_size,
            action_size=self.expl_env.action_space(),
            atoms=self.args.natoms).to(self.args.device)

        self.optimizer = torch.optim.Adam(
            params=self.q_learner.parameters(),
            lr=self.args.lr)
        self.q_target.load_state_dict(self.q_learner.state_dict())
        self.replay_buffer = ReplayMemory(self.args)

        self.z = torch.linspace(self.args.vmin, self.args.vmax, self.args.natoms).to(self.args.device)
        self.delta_z = (self.args.vmax - self.args.vmin) / (self.args.natoms - 1)

