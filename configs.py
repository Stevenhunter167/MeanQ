# each experiment must specify 1 name, 2 game, 3 seed
# config var name should use ALL CAPS
import copy


EXP_BASE = dict(
    # Basic Experiment
    # name=None,
    # seed=None,
    # game=None,
    # debug=False,
    device='cuda',
    exp_root='exp',
    checkpoint_freq=100_000,

    history_length=4,
    reward_std=0,
    discount=0.99,

    min_num_steps_before_training=1600,
    num_steps_per_epoch=10000,
    max_episode_length=int(108e3),
    num_eval_steps_per_epoch=int(5e5),
    eval_max_episode=20,

    capacity=int(5e5),
    priority_weight=0.4,
    priority_exponent=0.5,

    epsilon_steps=200_000,
)

NR_DQN = dict(
    trainer='DQN',
    num_epochs=100,
    network_type=None,
    lr=6.25e-5,
    ensemble_size=5,
    batch_size=32,
    target_update_freq=2000,
    grad_step_period=3,
    grad_step_repeat=1,
    multi_step=1,
    dueling=True,
    **EXP_BASE
)

AVG_DQN = copy.deepcopy(NR_DQN)
AVG_DQN.update(trainer='AvgDQN')

AVG_DQN_50 = copy.deepcopy(AVG_DQN)
AVG_DQN_50.update(
    dict(num_epochs=50, grad_step_period=2)
)

EDQN = copy.deepcopy(NR_DQN)
EDQN.update(trainer='EDQN')

EDQN_50 = copy.deepcopy(EDQN)
EDQN_50.update(
    dict(num_epochs=50, grad_step_period=2)
)

NR_DQN_50 = copy.deepcopy(NR_DQN)
NR_DQN_50.update(
    dict(num_epochs=50, grad_step_period=2)
)

NR_UNIT_DQN_50 = copy.deepcopy(NR_DQN)
NR_UNIT_DQN_50.update(
    dict(trainer='UnitDQN', num_epochs=50, grad_step_period=2)
)

NR_UNIT_DQN_NOTARGETNET_50 = copy.deepcopy(NR_UNIT_DQN_50)
NR_UNIT_DQN_NOTARGETNET_50.update(
    dict(target_update_freq=1)
)

NR_DQN_NOTARGETNET_50 = copy.deepcopy(NR_DQN_50)
NR_DQN_NOTARGETNET_50.update(
    dict(target_update_freq=1)
)

NR_DOUBLE_DQN = copy.deepcopy(NR_DQN)
NR_DOUBLE_DQN.update(
    dict(trainer='DoubleDQN')
)

NR_DQN_NOTARGETNET = copy.deepcopy(NR_DQN)
NR_DQN_NOTARGETNET.update(
    dict(target_update_freq=1)
)

NR_MEANQ = copy.deepcopy(NR_DQN)
NR_MEANQ.update(
    dict(trainer='MeanQ')
)

NR_MEANQ_50 = copy.deepcopy(NR_MEANQ)
NR_MEANQ_50.update(
    dict(num_epochs=50, grad_step_period=2)
)

NR_MEANQ_NOTARGETNET = copy.deepcopy(NR_MEANQ)
NR_MEANQ_NOTARGETNET.update(
    dict(target_update_freq=1)
)

NR_MEANQ_NOTARGETNET_50 = copy.deepcopy(NR_MEANQ_50)
NR_MEANQ_NOTARGETNET_50.update(
    dict(target_update_freq=1)
)

NR_MEANQ_NOTARGET_CLIP = copy.deepcopy(NR_MEANQ)
NR_MEANQ_NOTARGET_CLIP.update(
    dict(target_update_freq=1, clip_return=True)
)

NR_KFOLD = copy.deepcopy(NR_MEANQ)
NR_KFOLD.update(
    dict(trainer='Kfold')
)

KFOLD_NOISY = dict(
    # trainer
    trainer='Kfold_Noisy',
    num_epochs=50,
    # optim
    network_type=None,
    lr=1e-4,
    ensemble_size=5,
    batch_size=32,
    target_update_freq=1,
    grad_step_period=1,
    grad_step_repeat=1,
    # rainbow tricks
    dueling=True,
    multi_step=3,
    lpuct=1,
    vmin=-10,
    vmax=10,
    natoms=51,
    **EXP_BASE
)

MEANQ_NOISY = copy.deepcopy(KFOLD_NOISY)
MEANQ_NOISY.update(
    dict(
        trainer='MeanQ_Noisy'
    )
)

MEANQ_WITH_TARGET = copy.deepcopy(MEANQ_NOISY)
MEANQ_WITH_TARGET.update(
    dict(target_update_freq=2000)
)

MEANQ_MORE_DROPOUT = copy.deepcopy(KFOLD_NOISY)
MEANQ_MORE_DROPOUT.update(
    dict(
        trainer='MeanQ_Kdrop3b'
    )
)

MEANQ_EQUIV_DROPOUT = copy.deepcopy(MEANQ_NOISY)
MEANQ_EQUIV_DROPOUT.update(
    dict(
        trianer='MeanQ_Equiv_Dropout'
    )
)

RAINBOW_NO_TARGET = copy.deepcopy(KFOLD_NOISY)
# same setting as kfold, but with ensemble size 1
RAINBOW_NO_TARGET.update(
    dict(
        trainer='Rainbow',
        ensemble_size=1
    )
)

RAINBOW = copy.deepcopy(RAINBOW_NO_TARGET)  # same setting as RAINBOW_NO_TARGET, but with delayed target
RAINBOW.update(dict(target_update_freq=2000))

RAINBOW_MEANARCH = copy.deepcopy(RAINBOW)
RAINBOW.update(dict(trainer='Rainbow_MeanArch_Trainer'))

RAINBOW_MEANARCH_FREQ = copy.deepcopy(RAINBOW_MEANARCH)
RAINBOW.update(dict(grad_step_repeat=5))

RAINBOW_NO_TARGET_FREQ = copy.deepcopy(RAINBOW_NO_TARGET)
# same setting as RAINBOW_NO_TARGET, but with more frequent grad update
RAINBOW_NO_TARGET_FREQ.update(
    dict(
        grad_step_repeat=5
    )
)

RAINBOW_FREQ = copy.deepcopy(RAINBOW_NO_TARGET_FREQ)
# same as RAINBOW_NO_TARGET_FREQ, but with delayed target
RAINBOW_FREQ.update(
    dict(
        target_update_freq=2000
    )
)


if __name__ == '__main__':
    print(RAINBOW)
