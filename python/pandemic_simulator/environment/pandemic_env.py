# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from typing import List, Optional, Dict, Tuple, Mapping, Type, Sequence

import gym
import numpy as np

from .done import DoneFunction, ORDone, DoneFunctionType, DoneFunctionFactory
from .interfaces import LocationID, PandemicObservation, NonEssentialBusinessLocationState, PandemicRegulation, \
    InfectionSummary
from .pandemic_sim import PandemicSim
from .reward import RewardFunction, SumReward, RewardFunctionFactory, RewardFunctionType
from .simulator_config import PandemicSimConfig
from .simulator_opts import PandemicSimOpts
__all__ = ['PandemicGymEnv', 'PandemicGymEnv3Act']


class PandemicGymEnv(gym.Env):
    """A gym environment interface wrapper for the Pandemic Simulator."""

    _pandemic_sim: PandemicSim
    _stage_to_regulation: Mapping[int, PandemicRegulation]
    _obs_history_size: int
    _sim_steps_per_regulation: int
    _non_essential_business_loc_ids: Optional[List[LocationID]]
    _reward_fn: Optional[RewardFunction]
    _done_fn: Optional[DoneFunction]

    _last_observation: PandemicObservation
    _last_reward: float

    def __init__(self,
                 pandemic_sim: PandemicSim,
                 pandemic_regulations: Sequence[PandemicRegulation],
                 reward_fn: Optional[RewardFunction] = None,
                 done_fn: Optional[DoneFunction] = None,
                 obs_history_size: int = 1,
                 sim_steps_per_regulation: int = 24,
                 non_essential_business_location_ids: Optional[List[LocationID]] = None,
                 ):
        """
        :param pandemic_sim: Pandemic simulator instance
        :param pandemic_regulations: A sequence of pandemic regulations
        :param reward_fn: reward function
        :param done_fn: done function
        :param obs_history_size: number of latest sim step states to include in the observation
        :param sim_steps_per_regulation: number of sim_steps to run for each regulation
        :param non_essential_business_location_ids: an ordered list of non-essential business location ids
        """
        self._pandemic_sim = pandemic_sim
        self._stage_to_regulation = {reg.stage: reg for reg in pandemic_regulations}
        self._obs_history_size = obs_history_size
        self._sim_steps_per_regulation = sim_steps_per_regulation

        if non_essential_business_location_ids is not None:
            for loc_id in non_essential_business_location_ids:
                assert isinstance(self._pandemic_sim.state.id_to_location_state[loc_id],
                                  NonEssentialBusinessLocationState)
        self._non_essential_business_loc_ids = non_essential_business_location_ids

        self._reward_fn = reward_fn
        self._done_fn = done_fn
        self.action_space = gym.spaces.Discrete(len(self._stage_to_regulation))

    @classmethod
    def from_config(cls: Type['PandemicGymEnv'],
                    sim_config: PandemicSimConfig,
                    pandemic_regulations: Sequence[PandemicRegulation],
                    sim_opts: PandemicSimOpts = PandemicSimOpts(),
                    reward_fn: Optional[RewardFunction] = None,
                    done_fn: Optional[DoneFunction] = None,
                    obs_history_size: int = 1,
                    non_essential_business_location_ids: Optional[List[LocationID]] = None,
                    ) -> 'PandemicGymEnv':
        """
        Creates an instance using config

        :param sim_config: Simulator config
        :param pandemic_regulations: A sequence of pandemic regulations
        :param sim_opts: Simulator opts
        :param reward_fn: reward function
        :param done_fn: done function
        :param obs_history_size: number of latest sim step states to include in the observation
        :param non_essential_business_location_ids: an ordered list of non-essential business location ids
        """
        sim = PandemicSim.from_config(sim_config, sim_opts)

        if sim_config.max_hospital_capacity == -1:
            raise Exception("Nothing much to optimise if max hospital capacity is -1.")

        reward_fn = reward_fn or SumReward(
            reward_fns=[
                RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                                              summary_type=InfectionSummary.CRITICAL,
                                              threshold=sim_config.max_hospital_capacity),
                RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                                              summary_type=InfectionSummary.CRITICAL,
                                              threshold=3 * sim_config.max_hospital_capacity),
                RewardFunctionFactory.default(RewardFunctionType.LOWER_STAGE,
                                              num_stages=len(pandemic_regulations)),
                RewardFunctionFactory.default(RewardFunctionType.SMOOTH_STAGE_CHANGES,
                                              num_stages=len(pandemic_regulations))
            ],
            weights=[.4, 1, .1, 0.02]
        )

        return PandemicGymEnv(pandemic_sim=sim,
                              pandemic_regulations=pandemic_regulations,
                              sim_steps_per_regulation=sim_opts.sim_steps_per_regulation,
                              reward_fn=reward_fn,
                              done_fn=done_fn,
                              obs_history_size=obs_history_size,
                              non_essential_business_location_ids=non_essential_business_location_ids)

    @property
    def pandemic_sim(self) -> PandemicSim:
        return self._pandemic_sim

    @property
    def observation(self) -> PandemicObservation:
        return self._last_observation

    @property
    def last_reward(self) -> float:
        return self._last_reward

    def step(self, action: int) -> Tuple[PandemicObservation, float, bool, Dict]:
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # execute the action if different from the current stage
        if action != self._last_observation.stage[-1, 0, 0]:  # stage has a TNC layout
            regulation = self._stage_to_regulation[action]
            self._pandemic_sim.impose_regulation(regulation=regulation)

        # update the sim until next regulation interval trigger and construct obs from state hist
        obs = PandemicObservation.create_empty(
            history_size=self._obs_history_size,
            num_non_essential_business=len(self._non_essential_business_loc_ids)
            if self._non_essential_business_loc_ids is not None else None)

        hist_index = 0
        for i in range(self._sim_steps_per_regulation):
            # step sim
            self._pandemic_sim.step()

            # store only the last self._history_size state values
            if i >= (self._sim_steps_per_regulation - self._obs_history_size):
                obs.update_obs_with_sim_state(self._pandemic_sim.state, hist_index,
                                              self._non_essential_business_loc_ids)
                hist_index += 1

        prev_obs = self._last_observation
        self._last_reward = self._reward_fn.calculate_reward(prev_obs, action, obs) if self._reward_fn else 0.
        done = self._done_fn.calculate_done(obs, action) if self._done_fn else False
        self._last_observation = obs
        print("first obs: ", type(self._last_observation))
        print("first obs value: ",self._last_observation)
        return self._last_observation, self._last_reward, done, {}

    def reset(self) -> PandemicObservation:
        self._pandemic_sim.reset()
        self._last_observation = PandemicObservation.create_empty(
            history_size=self._obs_history_size,
            num_non_essential_business=len(self._non_essential_business_loc_ids)
            if self._non_essential_business_loc_ids is not None else None)
        self._last_reward = 0.0
        if self._done_fn is not None:
            self._done_fn.reset()
        return self._last_observation

    def render(self, mode: str = 'human') -> bool:
        pass

def norm_obs(x, obs_min, obs_max):
    return (x - obs_min) / (obs_max - obs_min)

class PandemicGymEnv3Act(gym.ActionWrapper):

    def __init__(self, env: PandemicGymEnv):
        super().__init__(env)
         
        self.env = env
        self.max_days = 120
        self.obs_norm_bds = {
            "global_infection_summary": [0, 1000], # obs min, obs max
            # "global_testing_summary": [0, 1000],
            "stage": [0, 4],
            "infection_above_threshold": [0, 1],
            "time_day": [0, self.max_days],
        }
        self.observation_space = gym.spaces.MultiDiscrete(np.ones(shape=(1, 1, 5+1+1+1)), 
                                                          dtype=np.float32)
        # note that the action space for the learner is {0, 1, 2}; different than the action space of the PandemicGymEnv
        self.action_space = gym.spaces.Discrete(5)

    @classmethod
    def setup_viz(self,config, gymviz, simviz):
        # setup viz
        self.config = config
        self.gymviz = gymviz
        self.simviz= simviz
        self.viz = gymviz.from_config(sim_config=config)
        self.sim_viz = simviz.from_config(sim_config=config)

    @classmethod
    def from_config(self,
                    sim_config: PandemicSimConfig,
                    pandemic_regulations: Sequence[PandemicRegulation],
                    sim_opts: PandemicSimOpts = PandemicSimOpts(),
                    reward_fn: Optional[RewardFunction] = None,
                    done_fn: Optional[DoneFunction] = None,
                    ) -> 'PandemicGymEnv3Act':

        env = PandemicGymEnv.from_config(sim_config = sim_config,
        pandemic_regulations=pandemic_regulations,
        sim_opts = sim_opts,
        reward_fn=reward_fn,
        done_fn=done_fn,
        )

        return PandemicGymEnv3Act(env=env)

    def flatten_obs(self, obs:PandemicObservation) -> np.ndarray:
        return np.concatenate([norm_obs(obs.global_infection_summary, *self.obs_norm_bds["global_infection_summary"]),
                               # norm_obs(obs.global_testing_summary, *self.obs_norm_bds["global_testing_summary"]),
                               norm_obs(obs.stage, *self.obs_norm_bds["stage"]),
                               norm_obs(obs.infection_above_threshold, *self.obs_norm_bds["infection_above_threshold"]),
                               norm_obs(obs.time_day, *self.obs_norm_bds["time_day"])
                               # unlocked_non_essential_business_locations is always none so it is excluded
            ], axis=-1)

    def step(self, action, flatten_obs=True):
        action = int(action)
        obs, reward, done, info = self.env.step(self.remap_action(action))
        # TODO: look into why the shape of the obs has 3 dimensions
        info["infection_above_threshold"] = obs.infection_above_threshold[0, 0, 0]
        #print("second obs: ",type(obs))
        #print("second obs value: ",obs)
        self.viz.record((obs, reward))
        self.sim_viz.record_state(state = self.env._pandemic_sim.state)
        if flatten_obs:
            obs = self.flatten_obs(obs)
        # also return done if we reach the maximal number of days
        self.current_days += 1
        done = done or (self.current_days >= self.max_days)

       
        print("Day is: ",self.current_days)
        if (done):
            self.viz.plot()
            self.sim_viz.plot()
            self.viz = self.gymviz.from_config(sim_config=self.config)
            self.sim_viz = self.simviz.from_config(sim_config=self.config)
            

        return obs, reward, done, info

    def remap_action(self, action):
        '''Remap action back to stages
        '''
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        #action -= 1 # map actions from range {0, 1, 2, 3, 4} into {-1, 0, 1}
        #remapped_action = int(min(4, max(0, self.env._last_observation.stage[-1, 0, 0] + action)))
        print("Stage is: ", action)
        return action
    
    def reset(self, flatten_obs=True):
        self.current_days = 0
        obs = self.env.reset()
        if flatten_obs:
            obs = self.flatten_obs(obs)
        return obs
        