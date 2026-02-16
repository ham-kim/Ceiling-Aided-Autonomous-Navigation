from RL_envs import MobileRobotEnv, TensorboardLoggingCallback
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# 동적 장애물이 포함된 환경으로 변경
env = DummyVecEnv([lambda: MobileRobotEnv(render_mode='human')])

# 정적 장애물 환경에서 학습한 모델 불러오기
model = PPO.load("RL_gird_test.zip")

# 환경을 새 동적 장애물 환경으로 설정
model.set_env(env)

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

# for i, (wps, result) in enumerate(zip(waypoints_list, episode_results)):
#     print(f"[Episode {i+1}] 결과: {result}")
#     if wps:
#         print(", ".join([f"np.array([{wp[0]}, {wp[1]}])" for wp in wps]))
print(episode_results.count("성공")/1000)

# 재학습된 모델 저장
model.save("RL_dynamic_finetuned_model")