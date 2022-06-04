import code

import torch
from trainers.ensemble_model_trainers.ensemble_q import EnsembleBase


class MeanQ(EnsembleBase):

    def target_estimate(self, s):
        """ MeanQ target estimate """
        b, k, d = s.shape[0], s.shape[1], self.expl_env.action_space()
        with torch.no_grad():
            s = torch.reshape(s, (b * k, 4, 84, 84))  # (bk, s)
            q_targets = self.q_target(s).view(b, k, self.q_learner.ensemble_size, d)  # (b, k, k, a)
            E_q_targets = torch.mean(q_targets, dim=2)  # (b, k, a)
            max_a_E_q_targets, ind = torch.max(E_q_targets, dim=2)  # (b, k)
            if "clip_return" in self.args and self.args.clip_return:
                max_a_E_q_targets = torch.clamp(max_a_E_q_targets, min=-10, max=10)
            return max_a_E_q_targets


class Kfold(MeanQ):

    def target_estimate(self, s):
        b, k, d = s.shape[0], s.shape[1], self.expl_env.action_space()
        with torch.no_grad():
            s = torch.reshape(s, (b * k, 4, 84, 84))  # (bk, s)
            # leave one out cross-validtation
            q_targets = self.q_target(s).view(b, k, k, d)  # (b, k, k, a)
            a = (q_targets.sum(dim=2, keepdim=True) - q_targets).argmax(dim=3, keepdim=True)
            vs = torch.gather(q_targets, dim=3, index=a)
            res = torch.squeeze(vs, dim=3).mean(2)
            return res
