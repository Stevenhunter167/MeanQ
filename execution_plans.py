import code
import copy
import gc
import tqdm
import traceback
import torch
import pprint
from trainers import *


def basic_experiment(args):
    """ basic experiment execution plan """
    args = copy.deepcopy(args)
    pprint.pprint(vars(args))
    Trainer = eval(args.trainer)
    trainer = Trainer(args)
    try:
        for epoch in range(trainer.args.num_epochs):
            trainer.print("epoch", epoch)
            if trainer.t == 0:
                trainer.collect_init_steps()
            epoch_steps = trainer.args.num_steps_per_epoch - trainer.args.min_num_steps_before_training * (epoch == 0)
            for _ in tqdm.trange(epoch_steps):
                trainer.sample_transition()
                trainer.before_learn_on_batch()
                if trainer.t % trainer.args.grad_step_period == 0:
                    for _ in range(trainer.args.grad_step_repeat):
                        batch = trainer.sample_batch(args.batch_size)
                        trainer.learn_on_batch(*batch)
                trainer.after_learn_on_batch()
            trainer.evaluate()
            if trainer.t % trainer.args.checkpoint_freq == 0:
                trainer.save()

    except Exception as e:
        tb = traceback.format_exc()
        trainer.print(e)
        trainer.print(tb)

    finally:
        trainer.save("final")
