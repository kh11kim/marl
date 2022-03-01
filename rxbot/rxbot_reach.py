import os
import gym
import numpy as np
from gym.envs.registration import register
from gym import spaces
from .rxbot import RxbotAbstractEnv

class RxbotReachEnv(RxbotAbstractEnv, gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False, dim=2, reward_type="task", random_init=True, task_ll=[0,-1,0], task_ul=[1,1,1], joint_range=2*np.pi):
        super().__init__(render=render, dim=dim, task_ll=task_ll, task_ul=task_ul, joint_range=joint_range)
        self.reward_type = reward_type
        self.random_init = random_init
        self.eps = 0.05
        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(self.robot.joint_idxs),), dtype=np.float32)
        self.observation_space = spaces.Box(
            np.hstack([self.robot.joint_ll, self.task_ll, self.task_ll]),
            np.hstack([self.robot.joint_ul, self.task_ul, self.task_ul]),
            shape=(self.robot.n_joints + 6,),
            dtype=np.float32
        )

    def _get_observation(self):
        """ observation : joint, ee_curr, ee_goal
        """
        joints = self.robot.get_joints()
        ee_pos = self.robot.get_ee_pos()
        goal_pos = self.goal.copy()
        return np.hstack([joints, ee_pos, goal_pos])

    def _is_success(self, ee_pos, goal_pos):
        return np.linalg.norm(ee_pos - goal_pos) < self.eps

    def get_random_joint_in_task_space(self):
        for i in range(100):
            joints = self.robot.get_random_joints(set=True)
            pos = self.robot.get_ee_pos()
            if np.all(self.task_ll < pos) & np.all(pos < self.task_ul):
                return joints, pos
        raise ValueError("EE position by a random configuration seems not in task-space.")

    def reset(self):
        with self.sim.no_rendering():
            self.goal_joints, self.goal = self.get_random_joint_in_task_space()
            if self.random_init:
                self.start_joints, self.start = self.get_random_joint_in_task_space()
            else:
                self.start_joints = np.zeros(self.dim)
                self.start = self.robot.get_ee_pos()
            self.robot.set_joints(self.start_joints)
        
        if self.is_render == True:
            self.sim.view_pos("goal", self.goal)
            self.sim.view_pos("curr", self.start)
        return self._get_observation()

    def select_action(self, action):
        action = action.copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        joint_target = self.robot.get_joints() + action * self.robot.max_joint_change
        joint_target = np.clip(joint_target, self.robot.joint_ll, self.robot.joint_ul) # joint limit
        self.robot.set_joints(joint_target)

    def step(self, action:np.ndarray):
        self.select_action(action)
        obs_ = self._get_observation()
        done = False
        info = dict(
            is_success=self._is_success(obs_[-6:-3], obs_[-3:]),
            joints=obs_[:self.robot.n_joints].copy(),
            actions=action.copy(),
        )
        reward = self.compute_reward(obs_[-6:-3], obs_[-3:], info)
        if self.is_render == True:
            self.sim.view_pos("curr", obs_[-6:-3])
        return obs_, reward, done, info

    def compute_reward(self, ee_curr, ee_goal, info):
        r = -np.linalg.norm(ee_curr - ee_goal, axis=-1)
        # joints = info["joints"]
        # goal_joints = info["goal_joints"]
        # actions = info["actions"]
        return r
        
        # if self.reward_type == "task":
        #     r -= np.linalg.norm(desired_goal - achieved_goal, axis=-1)
        #     #test
        #     mask_goal = np.linalg.norm(desired_goal - achieved_goal, axis=-1) < self.eps
        #     r -= np.linalg.norm(actions, axis=-1) / 10
        # elif self.reward_type == "joint":
        #     r -= np.linalg.norm(desired_goal - achieved_goal, axis=-1)
        #     mask1 = np.linalg.norm(desired_goal - achieved_goal, axis=-1) >= self.eps
        #     mask2 = np.linalg.norm(goal_joints - joints, axis=-1) > np.pi
        #     r -= mask1 * mask2 * np.linalg.norm(goal_joints - joints, axis=-1) / 10
            
        # else:
        #     raise NotImplementedError
        # return r
        

register(
    id='RxbotReach-v0',
    entry_point='utils.rxbot.rxbot_reach:RxbotReachEnv',
    max_episode_steps=50,
)