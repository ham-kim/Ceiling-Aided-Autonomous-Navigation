import gym
import numpy as np
import time
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardLoggingCallback, self).__init__(verbose)
        self.start_time = time.time()
        self.episode_reward = 0  # 에피소드 누적 보상 초기화

    def _on_step(self) -> bool:
        # 경과 시간 기록
        elapsed_time = time.time() - self.start_time
        self.logger.record("custom/elapsed_time", elapsed_time)
        
        # 스텝 보상 기록 및 누적
        reward = self.locals.get("rewards")[0]
        self.logger.record("custom/step_reward", reward)
        self.episode_reward += reward

        # 에피소드 종료 감지: 'dones' 플래그를 체크하여 기록
        dones = self.locals.get("dones")
        if dones and dones[0]:
            self.logger.record("custom/episode_reward", self.episode_reward)
            self.episode_reward = 0  # 에피소드 종료 후 누적 보상 초기화

        # 추가 정보 기록
        info = self.locals.get("infos")[0]
        if info is not None:
            self.logger.record("custom/distance_reward", info.get("distance_reward", 0))
            self.logger.record("custom/static_penalty", info.get("static_penalty", 0))
            self.logger.record("custom/dynamic_penalty", info.get("dynamic_penalty", 0))
        
        return True


# PPO 기반 Grid World 환경에서의 Mobile Robot 강화학습 환경 클래스
class MobileRobotEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(MobileRobotEnv, self).__init__()
        self.render_mode = render_mode

        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0, 0, 0]),
                                                high=np.array([19, 15, 19, 15, 19, 15]),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Discrete(8)  # 0: ↑, 1: ↓, 2: ←, 3: →

        self.obstacles = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])

        self.grid_size = self.obstacles.shape
        self.length, self.width = self.grid_size

        self.robot_pos = None
        self.goal_pos = None
        self.prev_pos = None
        self.new_pos = None
        self.dynamic_obstacles = []

        self.reset()

    def reset(self):
        self.goal_pos = np.array([18, 1])
        self.dynamic_obstacles = np.array([[16, 8]])  
        self.robot_pos = np.array([1, 14])
        self.waypoints = [self.robot_pos.copy()]
        return self._get_obs()

    def step(self, action):
        self.prev_pos = self.robot_pos.copy()
        self.new_pos = self.robot_pos.copy()
        self._move_dynamic_obstacles()

        # 행동에 따라 새로운 위치 계산 (행: y, 열: x)
        if action == 0 and self.robot_pos[1] < self.length:  # 위로 이동
            self.new_pos[1] += 1
        elif action == 1 and self.robot_pos[1] > 0:  # 아래로 이동
            self.new_pos[1] -= 1
        elif action == 2 and self.robot_pos[0] > 0:  # 왼쪽으로 이동
            self.new_pos[0] -= 1
        elif action == 3 and self.robot_pos[0] < self.width:  # 오른쪽으로 이동
            self.new_pos[0] += 1
        elif action == 4 and self.robot_pos[0] < self.width and self.robot_pos[1] < self.length:
            self.new_pos[0] += 1
            self.new_pos[1] += 1
        elif action == 5 and self.robot_pos[1] > 0 and self.robot_pos[0] > 0:
            self.new_pos[0] -= 1
            self.new_pos[1] -= 1
        elif action == 6 and self.robot_pos[0] > 0 and self.robot_pos[1] < self.length:
            self.new_pos[0] -= 1
            self.new_pos[1] += 1 
        elif action == 7 and self.robot_pos[1] > 0 and self.robot_pos[0] < self.width:
            self.new_pos[1] -= 1
            self.new_pos[0] += 1
        
        # # step() 함수 내에서 로봇 이동 조건 수정
        # if self.obstacles[self.new_pos[1], self.new_pos[0]] != 1 and not any(np.array_equal(self.new_pos, d) for d in self.dynamic_obstacles):
        #     self.robot_pos = self.new_pos
           
        # #new_pos로 이동 후 보상(충돌이 아예 이루어지지 않음) 
        # reward, dist_r, static_penalty, dynamic_penalty = self._calculate_reward()
        
        if not any(np.array_equal(self.new_pos, d) for d in self.dynamic_obstacles):
            self.robot_pos = self.new_pos

        if not np.array_equal(self.prev_pos, self.robot_pos):
            self.waypoints.append(self.robot_pos.copy())
            
        #new_pos 기준 보상
        reward, dist_r, static_penalty, dynamic_penalty = self._calculate_reward()

        done = False
        success = False
        collision = False

        if static_penalty < 0 or dynamic_penalty < 0:
            done = True
            collision = True
        elif np.allclose(self.robot_pos, self.goal_pos, atol=1e-3):
            done = True
            success = True

        if self.render_mode == 'human':
            self.render()

        return self._get_obs(), reward, done, {
            "distance_reward": dist_r,
            "static_penalty": static_penalty,
            "dynamic_penalty": dynamic_penalty,
            "success": success,
            "collision": collision
        }

    def _move_dynamic_obstacles(self):
        for i in range(len(self.dynamic_obstacles)):
            move = random.choice([(0, 1), (0, -1), (-1, 0), (1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)])
            dyn_pos = self.dynamic_obstacles[i] + move
            # x 좌표는 width, y 좌표는 length와 비교
            if 0 < dyn_pos[0] < self.width and 0 < dyn_pos[1] < self.length:
                if self.obstacles[dyn_pos[1], dyn_pos[0]] != 1 and \
                   not np.array_equal(dyn_pos, self.goal_pos) and \
                   not np.array_equal(dyn_pos, self.robot_pos):
                    self.dynamic_obstacles[i] = dyn_pos

    def _get_obs(self):
        # 로봇 위치, 목표 위치, 동적 장애물 위치(평탄화) 포함
        return np.concatenate((self.robot_pos, self.goal_pos, self.dynamic_obstacles.flatten()))


    def _calculate_reward(self):
        static_penalty = 0
        dynamic_penalty = 0
        
        #이동할 new_pos에 대한 reward. 충돌 시 -reward를 받는다.
        if self.obstacles[self.new_pos[1], self.new_pos[0]] == 1:
            static_penalty = -10
            
           
        for dyn_obs in self.dynamic_obstacles:
            if np.array_equal(self.new_pos, dyn_obs):
                dynamic_penalty = -50

        if np.array_equal(self.new_pos, self.goal_pos):
            return 100, 0, static_penalty, dynamic_penalty

        prev_dist = np.linalg.norm(self.prev_pos - self.goal_pos)
        current_dist = np.linalg.norm(self.robot_pos - self.goal_pos)
        diff =  prev_dist - current_dist   
        dist = 1 / np.linalg.norm(self.new_pos - self.goal_pos) + diff
        time_penalty = -0.01  # 예: 매 스텝마다 -0.01 보상(패널티) 추가
        return dist + static_penalty + dynamic_penalty + time_penalty, dist, static_penalty, dynamic_penalty
    
        # #로봇 행동이 종료 후 로봇 위치를 기준으로 계산함. 충돌이 고려되지 않음.
        # if self.obstacles[self.robot_pos[1], self.robot_pos[0]] == 1:
        #     static_penalty = -10
            
            
        # for dyn_obs in self.dynamic_obstacles:
        #     if np.array_equal(self.robot_pos, dyn_obs):
        #         dynamic_penalty = -50

        # if np.array_equal(self.robot_pos, self.goal_pos):
        #     return 200, 0, static_penalty, dynamic_penalty

        # dist = np.linalg.norm(self.robot_pos - self.goal_pos)
        # time_penalty = -0.01  # 예: 매 스텝마다 -0.01 보상(패널티) 추가
        # return -dist + static_penalty + dynamic_penalty + time_penalty, -dist, static_penalty, dynamic_penalty



    def render(self, mode='human'):
        env_map = np.full((self.length, self.width), '.', dtype=str)
        env_map[self.obstacles == 1] = 'X'
        if not np.array_equal(self.goal_pos, self.robot_pos):
            env_map[self.goal_pos[1], self.goal_pos[0]] = 'G'
        env_map[self.robot_pos[1], self.robot_pos[0]] = 'R'
        for dyn_obs in self.dynamic_obstacles:
            if not np.array_equal(dyn_obs, self.robot_pos):
                env_map[dyn_obs[1], dyn_obs[0]] = 'D'
        print("\n".join(" ".join(row) for row in env_map), end="\n\n")