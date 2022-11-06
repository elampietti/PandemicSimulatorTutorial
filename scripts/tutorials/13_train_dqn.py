# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

# training script heavily based off tianshou lunarlander dqn example
import argparse
import os
import pprint
import datetime
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net

import pandemic_simulator as ps
from pandemic_simulator.environment.interfaces import InfectionSummary
from pandemic_simulator.environment.done import ORDone, DoneFunctionFactory, DoneFunctionType 
from tianshou_utils import get_args, make_agent
 

def train_dqn(args=get_args()):
    # init env
    ps.init_globals(seed=args.seed)
    sim_config = ps.sh.small_town_config
    done_threshold = sim_config.max_hospital_capacity * 3
    done_fn = ORDone(
            done_fns=[
                DoneFunctionFactory.default(
                    DoneFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                    summary_type=InfectionSummary.CRITICAL,
                    threshold=done_threshold,
                ),
                DoneFunctionFactory.default(DoneFunctionType.NO_PANDEMIC, num_days=40),
            ]
        )

    env = ps.env.PandemicGymEnv3Act.from_config(sim_config=sim_config, 
                 pandemic_regulations=ps.sh.austin_regulations,
                 done_fn=done_fn)

    ps.env.PandemicGymEnv3Act.setup_viz(config=sim_config, gymviz=ps.viz.GymViz, simviz=ps.viz.SimViz)

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    
    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    policy, optim = make_agent(args)
    # initialize collector
    train_collector = Collector(
        policy,
        env,
        ReplayBuffer(args.buffer_size),
        exploration_noise=True
    )
    # we do not use a test collector because we cannot initialize more than 1 env at a time
    test_collector = None

    # setup logging
    date_time = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    log_path = os.path.join(args.logdir, args.task, f"dqn_{args.expt_name}_{date_time}")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, 
        train_interval=args.log_interval, # this interval is over the number of training timesteps
        update_interval=5) # this interval is over the number of updates

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        '''
        If desired, can specify a reward threshold to stop at with the following line:        
        return mean_rewards >= reward_threshold
        '''
        return False

    def train_fn(epoch, env_step):  # exp decay
        eps = max(args.eps_train_init * (1 - 1.5e-4)**env_step, args.eps_train_final)
        policy.set_eps(eps)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # prefill buffer with some random data
    train_collector.collect(n_step=args.init_random_steps, random=True)



    # main training loop
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.n_epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=1, # no test env
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        stop_fn=stop_fn,
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
        logger=logger
    )

    # save last policy
    torch.save(policy.state_dict(), os.path.join(log_path, "last_policy.pth"))

    return result


if __name__ == '__main__':
    args = get_args()

    start = time.time()
    result = train_dqn(args)
    print("Final result: ")
    pprint.pprint(result)


    
    elapsed = time.time() - start
    print("Train time: ", str(datetime.timedelta(seconds=elapsed)))