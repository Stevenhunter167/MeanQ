import torch
from models.q_networks import Dropout_Q_Network, Noisy_Q_Network
from trainers.dqn import DQN

class FreqDQN(DQN):

    def learn(self, treeidx, s, a, r, s_, d, w):
        # learn on batch
        if self.t % self.args.grad_step_period == 0:

            for _ in range(self.args.ensemble_size):
                treeidx, s, a, r, s_, d, w = self.replay_buffer.sample(self.args.batch_size)
            
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
                (IS_mseloss / self.args.ensemble_size).backward()  # take k steps, with each step size lr / k
                self.optimizer.step()
                # PER update
                abs_loss = torch.sqrt(loss).detach().view(-1).cpu().numpy()
                self.replay_buffer.update_priorities(treeidx, abs_loss)

            # logging
            with torch.no_grad():
                self.exp_path['batch_stats[t|num_done|pred_mean|q_loss|target_mean]'].csv_writerow(
                    [self.t, d.sum().item(), pred.mean().item(), IS_mseloss.item(), target.mean().item()])
