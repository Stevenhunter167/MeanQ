import torch
import torch.nn as nn

class Ensemble(nn.Module):
    """ ensemble of neural net model """

    def __init__(self, model_class, ensemble_size, *args, **kwargs):
        super(Ensemble, self).__init__()
        self.ensemble_size = ensemble_size
        self.models = nn.ModuleList(
            [model_class(*args, **kwargs) for _ in range(self.ensemble_size)]
        )

    def forward(self, obs, *args, learners=None):
        """
        X: input shape
        O: output shape
        B: batch size
        K: ensemble size

        obs (B [, K], X) return (B, K, O)
        specify learner indicies in learners for kdrop forwarding
        """
        if learners is None:
            learners = [i for i in range(self.ensemble_size)]

        res = []
        for i, k in enumerate(learners):
            if len(obs.shape) == 4:  # sep batch
                ok = self.models[k](obs, *args)
            elif len(obs.shape) == 5:  # no sep batch
                ok = self.models[k](obs[:, i, :, :, :], *args)
            else:  # unsupported shape
                raise Exception("unsupported shape")
            res.append(ok)
        return torch.stack(res, dim=1)

    def reset_noise(self):
        for m in self.models:
            m.reset_noise()