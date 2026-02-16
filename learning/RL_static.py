from RL_envs import MobileRobotEnv, TensorboardLoggingCallback
import torch
import gym
import numpy as np
import time
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


if torch.cuda.is_available():
    print("CUDA 사용 가능 - GPU를 사용합니다.")
else:
    print("CUDA 사용 불가능 - CPU를 사용합니다.")

print(torch.version.cuda) 
    
    
# 환경 초기화 및 학습 (TensorBoard 로그 디렉토리를 지정)
env = DummyVecEnv([lambda: MobileRobotEnv(render_mode='human')])
tensorboard_log_dir = "./tensorboard_logs/"
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device='cpu', #gpu 사용
    tensorboard_log=tensorboard_log_dir,
    # ===== 여기서부터 하이퍼파라미터를 수정 =====
    learning_rate=3e-4,     # 학습률
    n_steps=2048,           # 한 번에 모으는 샘플 수 (rollout buffer size)
    batch_size=128,          # 미니배치 크기
    gamma=0.99,             # 할인율
    gae_lambda=0.95,        # GAE 람다
    clip_range=0.2,         # PPO 클리핑 범위
    ent_coef=0.0,           # 엔트로피 계수 (탐색 유도)
    # ===== 추가로 policy_kwargs로 신경망 구조 설정 가능 =====
    policy_kwargs={
        "net_arch": [128, 128],  # 은닉층 2개, 각 128 유닛
        # "activation_fn": torch.nn.ReLU,  # 기본값 ReLU
    }
)

# TensorBoard 콜백 생성
tb_callback = TensorboardLoggingCallback()

model.learn(total_timesteps=300000, callback=tb_callback)

# 학습 성능 기록 (에피소드별 최종 보상 등)
start_time = time.time()
episode_rewards, distance_rewards = [], []
static_penalties, dynamic_penalties = [], []
collision_counts_static, collision_counts_dynamic = [], []
waypoints_list, episode_results = [], []

for episode in range(1000):
    print(f"\n[Episode {episode + 1}]")
    obs = env.reset()
    done = False
    total_r, dist_r, static_p, dynamic_p = 0, 0, 0, 0
    collision_s, collision_d = 0, 0
    waypoints = []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done_flag, info = env.step(action)
        done = done_flag[0]
        info = info[0]

        total_r += reward[0]
        dist_r += info['distance_reward']
        static_p += info['static_penalty']
        dynamic_p += info['dynamic_penalty']

        pos = env.envs[0].robot_pos
        waypoints.append(np.array(pos))

        if info['static_penalty'] < 0:
            collision_s += 1
        if info['dynamic_penalty'] < 0:
            collision_d += 1

        if done:
            episode_results.append("성공" if info["success"] else "충돌함!" if info["collision"] else "실패")
            waypoints_list.append([] if info["collision"] else waypoints)
            break

    episode_rewards.append(total_r)
    distance_rewards.append(dist_r)
    static_penalties.append(static_p)
    dynamic_penalties.append(dynamic_p)
    collision_counts_static.append(collision_s)
    collision_counts_dynamic.append(collision_d)

print("\n[학습 결과 요약]")
print(f"총 소요 시간: {time.time() - start_time:.2f}초")
print(f"에피소드별 보상: {episode_rewards}")
print(f"거리 보상: {distance_rewards}")
print(f"정적 장애물 패널티: {static_penalties}")
print(f"동적 장애물 패널티: {dynamic_penalties}")
print(f"정적 충돌 수: {collision_counts_static}")
print(f"동적 충돌 수: {collision_counts_dynamic}")

for i, (wps, result) in enumerate(zip(waypoints_list, episode_results)):
    if result == "성공":
        print(f"[Episode {i+1}] 결과: {result}")
        print(", ".join([f"np.array([{wp[0]}, {wp[1]}])" for wp in wps]))

print(episode_results.count("성공")/1000)

# 모델 저장 (RL_grid_test.zip 형태로 저장)
model.save("RL_gird_test")
