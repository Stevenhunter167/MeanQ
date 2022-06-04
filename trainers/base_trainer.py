import code
import os.path
from tqdm import trange

import numpy as np
import random
import torch
import torch.nn as nn

from components.env import Env
from components.memory import ReplayMemory
from components.filesys_manager import ExperimentPath


class BaseTrainer:

    """ Initialize Atari environment, define basic execution plan """

    def __init__(self, args):
        # init experiment hyper-parameters
        self.args = args
        self.debug = self.args.debug
        self.allow_printing = True
        self.exp_path = ExperimentPath(f"{self.args.exp_root}/{self.args.name}/{self.args.game}/{self.args.preset}/{self.args.seed}")
        self.print("log directory initialized")
        # env init
        self.expl_env = Env(self.args)
        self.eval_env = Env(self.args)
        self.eval_env.eval()
        # replay buffer
        self.replay_buffer = ReplayMemory(self.args)
        # global time steps
        self.t = 0
        # exploration environment
        self.s = self.expl_env.reset()
        self.done = False
        self.episode_num = 0
        self.episode_len = 0
        self.episode_return = 0
        self.episode_discounted_return = 0
        # init model
        self.init_model()
        # log
        self.exp_path['timestamp[timenow|t|msg]'].csv_writerow([self.exp_path.now(), self.t, "model initialized"])
        self.exp_path['config'].json_write(vars(self.args))
        self.exp_path['model_info'].txt_write(str(self.q_learner))
        self.save('initial')

    def print(self, *args, **kwargs):
        verbose = kwargs.get('verbose', 0)
        if 'verbose' in kwargs: del kwargs['verbose']
        if self.allow_printing and self.args.verbose >= verbose:
            print("|", self.exp_path.now(), self.exp_path.str(), "|",  *args, **kwargs)

    def init_model(self):
        """ init model """
        self.print("warning: init model not implemented")
        self.q_learner = None
        self.q_target = None
        self.optimizer = None

    def expl_action(self, obs) -> int:
        self.print("warning: expl action not implemented")
        return np.random.randint(self.expl_env.action_space())

    def eval_action(self, obs) -> int:
        self.print("warning: eval action not implemented")
        return np.random.randint(self.expl_env.action_space())

    def before_learn_on_batch(self):
        self.print("warning: before learn on batch not implemented")

    def learn_on_batch(self, treeidx, s, a, r, s_, d, w):
        """ learn on batch """
        self.print("warning: learn on batch not implemented")

    def after_learn_on_batch(self):
        self.print("warning: after learn on batch not implemented")

    def state_value_pred(self, s):
        """ log the state value of initial state """
        self.print("warning: state value pred not implemented")
        return -1

    def reset_expl_env(self):
        """ reset exploration env """
        self.s = self.expl_env.reset()
        self.done = False
        self.episode_num += 1
        self.episode_len = 0
        self.episode_return = 0
        self.episode_discounted_return = 0

    def advance_expl_env(self, next_obs, reward, done):
        """ advance exploration env for next step """
        self.s = next_obs
        self.done = done
        self.episode_return += reward
        clipped_reward = min(max(reward, -1), 1)
        self.episode_discounted_return += clipped_reward * self.args.discount ** self.episode_len
        self.episode_len += 1

    def epsilon(self):
        """ compute current epsilon (linear annealing 1M steps) """
        return 1 - self.t * ((1 - 0.1) / self.args.epsilon_steps) if self.t < self.args.epsilon_steps else 0.1

    def collect_init_steps(self, render=False):
        """ fill the replay buffer before learning """
        for _ in trange(self.args.min_num_steps_before_training):
            if self.done:
                self.reset_expl_env()
            # sample transition
            a = np.random.randint(self.expl_env.action_space())
            s_, r, d = self.expl_env.step(a)
            # store into replay
            clipped_r = min(max(r, -1), 1)
            if 'reward_std' in self.args:
                clipped_r += np.random.normal(0, self.args.reward_std)  # clip, add noise
            self.replay_buffer.append(self.s, a, clipped_r, d)
            # render
            if render:
                self.expl_env.render()
            # sample next state
            self.advance_expl_env(s_, r, d)
            self.t += 1
        self.reset_expl_env()
        self.print("finished collect init steps")

    def sample_transition(self, render=False):
        """ train one step """
        if self.done:
            self.reset_expl_env()
            # log initial state value
            # Vs = self.state_value_pred(self.s)
            # self.expl_stats_writer.write('t_vs_expl_v_initial', self.t, Vs)

        # sample transition
        a = self.expl_action(self.s)
        s_, r, d = self.expl_env.step(a)
        # store into replay
        clipped_r = min(max(r, -1), 1)
        if 'reward_std' in self.args:
           clipped_r += np.random.normal(0, self.args.reward_std)  # clip, add noise
        self.replay_buffer.append(self.s, a, clipped_r, d)
        # render
        if render:
            self.expl_env.render()
        # sample next state
        self.advance_expl_env(s_, r, d)

        # log status
        if self.done:
            self.exp_path['expl_episode_stats[t|return|discounted_return]'].csv_writerow(
                [self.t, self.episode_return, self.episode_discounted_return])
            self.exp_path['debug[t|buffer_size|epsilon]'].csv_writerow(
                [self.t, self.replay_buffer.data_buffer.index, self.epsilon()]
            )

        self.t += 1

    def evaluate(self, render=False, random=False):
        """ evaluate current policy """
        self.exp_path['timestamp[timenow|t|msg]'].csv_writerow([self.exp_path.now(), self.t, "evaluation started"])
        self.print(f"evaluation started t={self.t}")
        # init eval
        s = self.eval_env.reset()
        done = False
        episode_len = 0
        episode_return = 0
        episode_discounted_return = 0
        v_initials = []
        v_initials.append(self.state_value_pred(s))
        returns = []
        discounted_returns = []
        episode_lens = []
        # start eval
        for _ in trange(self.args.num_eval_steps_per_epoch):
            if done:
                # record data
                returns.append(episode_return)
                discounted_returns.append(episode_discounted_return)
                episode_lens.append(episode_len)
                # reset env
                s = self.eval_env.reset()
                v_initials.append(self.state_value_pred(s))
                episode_len = 0
                episode_return = 0
                episode_discounted_return = 0
            if len(returns) >= self.args.eval_max_episode:
                # check if enough episodes simulated
                break
            # sample transition
            a = self.eval_action(s)
            if random:
                a = np.random.randint(self.eval_env.action_space())
            s_, r, d = self.eval_env.step(a)
            # advance env
            s = s_
            done = d
            episode_return += r
            clipped_r = min(max(r, -1), 1)
            episode_discounted_return += clipped_r * self.args.discount ** episode_len
            episode_len += 1
            # render
            if render:
                print(f"s={s.shape} a={a} r={r} r_clip={clipped_r}")
                self.eval_env.render()
        # finished eval
        self.print(f"returns={returns}\nmean={np.mean(returns) if len(returns) > 0 else 0}")
        if len(returns) == 0:
            # self.eval_stats_writer.write('t_vs_eval_episode_return', self.t, episode_return)
            # self.eval_stats_writer.write('t_vs_eval_episode_discounted_return_clipped', self.t, episode_discounted_return)
            # self.eval_stats_writer.write('t_vs_eval_episode_len', self.t, episode_len)
            self.exp_path['eval_episode_stats[t|v_init|return|discounted_return|length]'].csv_writerow(
                [self.t, v_initials[0], episode_return, episode_discounted_return, episode_len])
            self.exp_path['eval_mean_stats[t|num_episode|v_init|return|discounted_return|length]'].csv_writerow(
                [self.t, 1, v_initials[0], episode_return, episode_discounted_return, episode_len])

        else:
            # self.eval_stats_writer.write('t_vs_eval_episode_return_mean', self.t, np.mean(returns))
            # self.eval_stats_writer.write('t_vs_eval_episode_discounted_return_clipped_mean', self.t, np.mean(discounted_returns))
            # self.eval_stats_writer.write('t_vs_eval_episode_len_mean', self.t, np.mean(episode_lens))
            # self.eval_stats_writer.write('t_vs_eval_v_initial_mean', self.t, np.mean(v_initials))
            # for i in v_initials:
            #     self.eval_stats_writer.write('t_vs_eval_v_initial', self.t, i)
            # for i in returns:
            #     self.eval_stats_writer.write('t_vs_eval_episode_return', self.t, i)
            # for i in discounted_returns:
            #     self.eval_stats_writer.write('t_vs_eval_episode_discounted_return_clipped', self.t, i)
            # for i in episode_lens:
            #     self.eval_stats_writer.write('t_vs_eval_episode_len', self.t, i)
            for V_init, R, discounted_R, length in zip(v_initials, returns, discounted_returns, episode_lens):
                self.exp_path['eval_episode_stats[t|v_init|return|discounted_return|length]'].csv_writerow(
                    [self.t, V_init, R, discounted_R, length])
            self.exp_path['eval_mean_stats[t|num_episode|v_init|return|discounted_return|length]'].csv_writerow(
                [self.t, len(returns), np.mean(v_initials), np.mean(returns), np.mean(discounted_returns), np.mean(episode_lens)])

        # self.eval_stats_writer.write('t_vs_eval_num_episode', self.t, len(returns))
        # self.time_writer.write('timestamp', now(), f'evaluation finished t={self.t}')
        self.exp_path['timestamp[timenow|t|msg]'].csv_writerow([self.exp_path.now(), self.t, "evaluation finished"])

    def save(self, name='trainer'):
        """ save this trainer """
        # buffer = self.replay_buffer
        # expl_env = self.expl_env
        # eval_env = self.eval_env
        # # skip the things not saving
        # self.replay_buffer = None
        # self.expl_env = None
        # self.eval_env = None
        # self.exp_path["checkpoint"][f"{name}_{self.t}.pth"].save_model(self)
        # self.exp_path['timestamp[timenow|t|msg]'].csv_writerow([self.exp_path.now(), self.t, "model saved"])
        # self.replay_buffer = buffer
        # self.expl_env = expl_env
        # self.eval_env = eval_env
        self.exp_path["checkpoint"][f"{name}_{self.t}.pth"].save_model(self.q_learner.state_dict())
        self.exp_path['timestamp[timenow|t|msg]'].csv_writerow([self.exp_path.now(), self.t, "model saved"])
        

    # @classmethod
    # def load(cls, path):
    #     # with open(path, 'rb') as f:
    #     #     return torch.load(f)
    #     return ExperimentPath(path).load()

