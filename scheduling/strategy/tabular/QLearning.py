from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, TYPE_CHECKING, Final

import numpy as np
import torch

from misc.StaticRng import StaticRng
from scheduling.InferenceRequest import InferenceRequest, InferenceStats
from scheduling.Scheduler import SchedulingResult

if TYPE_CHECKING:
    from scheduling.DefaultScheduler import  DefaultScheduler


class QLearningWrapper(ABC):
    @abstractmethod
    def take_preparation_action(self, scheduler : DefaultScheduler,
                                pending_requests : List[InferenceRequest],
                                prepared_requests : List[InferenceRequest],
                                any_stream_free : bool) -> None:
        pass

    @abstractmethod
    def take_execution_action(self, scheduler : DefaultScheduler,
                              pending_requests : List[InferenceRequest],
                              prepared_requests : List[InferenceRequest],
                              ongoing_requests : List[InferenceRequest | None],
                              cuda_streams : List[torch.cuda.Stream],) -> None:
        pass

    @abstractmethod
    def on_completed_request(self, stats : InferenceStats) -> None:
        pass

    @abstractmethod
    def on_energy_violation(self, active_results : List[SchedulingResult]):
        pass

    @abstractmethod
    def get_state_id(self, pending_requests : List[InferenceRequest],
                     prepared_requests : List[InferenceRequest],
                     any_stream_free : bool):
        pass

    @abstractmethod
    def on_external_state_change(self, old_state : int, new_state : int):
        pass

    @abstractmethod
    def on_begin_next_episode(self):
        pass

    @abstractmethod
    def save(self, path: os.PathLike):
        pass

    @property
    @abstractmethod
    def total_episode_reward(self) -> float:
        pass


class QLearning:
    # if values get this low, then what even happened
    ACTION_MASK_VALUE : Final[int] = -2000000000
    def __init__(self, initial_q_values : np.ndarray,
                 learning_rate : float = 0.1,
                 discount_factor : float = 0.99,
                 start_epsilon : float = 1.0,
                 min_epsilon : float = 0.05,
                 epsilon_decay : float = 0.99999, # roughly 300.000 steps until decayed to min_epsilon
                 ):
        # NOTE: current implementation assumes ACTION_MASK_VALUE is low enough that a legitimate action never gets a q-value that low
        self._q : np.ndarray = initial_q_values
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._epsilon = start_epsilon
        self._min_epsilon = min_epsilon
        self._epsilon_decay = epsilon_decay
        self._rng : Final[StaticRng] = StaticRng()

    def get_action(self, state : int, deterministic = False) -> int:
        actions = self._q[state]
        if deterministic:
            return int(np.argmax(actions))
        p = self._rng.random()
        if p < self._epsilon:
            valid_actions = np.flatnonzero(actions > QLearning.ACTION_MASK_VALUE) # equivalent to np.nonzero(actions >= QLearning.ACTION_MASK_VALUE)[0] if actions is one-dimensional
            #action_index = self._rng.choice(valid_actions)
            action_index = valid_actions[int(self._rng.random() * len(valid_actions))]
        else:
            best_actions = np.flatnonzero(actions == np.max(actions))
            #action_index = self._rng.choice(best_actions)
            action_index = best_actions[int(self._rng.random() * len(best_actions))]

        self._epsilon = max(self._epsilon * self._epsilon_decay, self._min_epsilon)
        return action_index

    def update_q_values(self, state : int, action : int, reward : float, next_state : int) -> None:
        # No action always keeps q-value 0, since it leads to the same state and has no reward, while invalid
        # actions have negative q values
        delta = reward + self._discount_factor * np.max(self._q[next_state]).item() - self._q[state][action]
        self._q[state][action] += self._learning_rate * delta

    def save_q_values(self, path : Path):
        np.save(path, self._q)

