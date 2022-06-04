# register trainer here
from trainers.base_trainer import                                   BaseTrainer

# single model trainers
from trainers.single_model_trainers.dqn import                                      DQN
from trainers.single_model_trainers.dqn import                                      UnitDQN
from trainers.single_model_trainers.dqn_no_targetnet import                         DQN_NoTargetNet
from trainers.single_model_trainers.double_dqn import                               DoubleDQN
from trainers.single_model_trainers.rainbow import                                  Rainbow
from trainers.single_model_trainers.rainbow import                                  Rainbow_MeanArch_Trainer
from trainers.single_model_trainers.avg_dqn import                                  AvgDQN
from trainers.ensemble_model_trainers.edqn import                                   EDQN

# from trainers.single_model_trainers.mean_arch_dqn import            MeanArchDQN
# from trainers.single_model_trainers.freq_dqn import                 FreqDQN
# from trainers.single_model_trainers.dropout_dqn import              DropoutDQN

# ensemble model trainers
from trainers.ensemble_model_trainers.mean_q import                                 MeanQ, Kfold
from trainers.ensemble_model_trainers.mean_q_distributional import                  MeanQ_Distributional, MeanQ_Noisy, MeanQ_Equiv_Dropout, MeanQ_SanityCheck, MeanQ_Kdrop3b
from trainers.ensemble_model_trainers.kfold_distributional import                   Kfold_Distributional
from trainers.ensemble_model_trainers.kfold_distributional_noisy import             Kfold_Noisy, Kfold_Noisy_Mean_Target

# rainbow
# from trainers.ensemble_model_trainers.double_mean_q import                           Double_MeanQ_Noisy
# from trainers.ensemble_model_trainers.k_dropout import              Kdropout
# from trainers.ensemble_model_trainers.k_drop_sanity_check import    Kdrop_SanityCheck
# from trainers.ensemble_model_trainers.no_target_net import          NoTargetNet
# from trainers.ensemble_model_trainers.kfold import                  Kfold
# from trainers.ensemble_model_trainers.mean_q_same_batch import      SameBatchMeanQ
