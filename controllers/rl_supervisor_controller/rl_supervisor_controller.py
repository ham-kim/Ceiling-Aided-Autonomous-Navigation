from controller import Supervisor
import numpy as np
import torch
from stable_baselines3 import PPO
import math
import time
import socket


class RLSupervisor(Supervisor):
    def __init__(self):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())

        # 로봇 노드 정의
        self.robot_node = self.getFromDef("ROBOT_NAME")
        if self.robot_node is None:
            print("❌ 로봇 노드를 찾을 수 없습니다. DEF 이름을 확인하세요.")
            exit(1)

        self.robot_translation = self.robot_node.getField("translation")
        self.robot_rotation = self.robot_node.getField("rotation")


        # 목표 위치
        self.goal_pos = np.array([18.0, 1.0])
        self.step_size = 0.2  # Grid 단위

        # PPO 모델 로드
        self.rl_model = PPO.load("RL_grid_test.zip")


        # 소켓 초기화 (수신은 run() 안에서)
        self.conn = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('127.0.0.1', 9999))
        self.sock.listen(1)
        print("🟢 YOLO 데이터 수신 대기 중...")

    def run(self):
        goal_reached = False

        while self.step(self.timestep) != -1:
            # 연결이 없으면 계속 대기
            if self.conn is None:
                try:
                    self.sock.settimeout(0.1)
                    self.conn, addr = self.sock.accept()
                    self.conn.settimeout(0.01)
                    print(f"🔗 YOLO 연결됨: {addr}")
                except socket.timeout:
                    continue
                continue  # 연결 직후 스킵

            try:
                data = self.conn.recv(1024).decode()
                if not data:
                    continue

                print(f"📩 받은 데이터: {data}")
                rx, ry, px, py = map(float, data.strip().split(","))

                robot_pos = np.array([rx, ry])
                person_pos = np.array([px, py])

                # PPO 입력 관측값 구성
                obs = np.concatenate((robot_pos, person_pos, self.goal_pos)).astype(np.float32).reshape(1, -1)
                print(f"📥 obs: {obs}")

                # 행동 예측 및 이동
                action, _ = self.rl_model.predict(obs)
                move = self._convert_action_to_vector(action[0])
                print(f"🤖 action: {action[0]}, move: {move}")

                new_pos = robot_pos + move * self.step_size
                print(f"📦 로봇 위치 업데이트: ({new_pos[0]:.2f}, {new_pos[1]:.2f}, 0.2)")

                self.robot_translation.setSFVec3f([new_pos[0], new_pos[1], 0.2])
                heading = math.atan2(move[1], move[0])
                self.robot_rotation.setSFRotation([0, 0, 1, heading])

                # 목표 도달 체크
                if not goal_reached and np.linalg.norm(robot_pos - self.goal_pos) < 0.5:
                    print(f"🎯 목표 도달! 위치: ({rx:.2f}, {ry:.2f})")
                    goal_reached = True

                time.sleep(0.05)  # 이동 속도 조절
            except socket.timeout:
                continue
            except Exception as e:
                print(f"⚠️ 에러 발생: {e}")
                break

    def _convert_action_to_vector(self, action):
        directions = {
            0: np.array([0, 1]),    # ↑
            1: np.array([0, -1]),   # ↓
            2: np.array([-1, 0]),   # ←
            3: np.array([1, 0])     # →
        }
        return directions.get(action, np.array([0, 0]))


if __name__ == "__main__":
    supervisor = RLSupervisor()
    supervisor.run()

