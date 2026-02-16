"""Pedestrian class container."""
from controller import Supervisor
from random import randint
import time
import optparse
import math
from random import choice


import sys
print("Webots 사용 중인 Python 경로:", sys.executable)
print(sys.executable)



class Pedestrian(Supervisor):
    """Control a Pedestrian PROTO."""

    def __init__(self):
        """Constructor: initialize constants."""
        self.BODY_PARTS_NUMBER = 13
        self.WALK_SEQUENCES_NUMBER = 8
        self.ROOT_HEIGHT = 1.27
        self.CYCLE_TO_DISTANCE_RATIO = 0.05
        self.speed = 1
        self.current_height_offset = 0
        self.joints_position_field = []
        self.joint_names = [
            "leftArmAngle", "leftLowerArmAngle", "leftHandAngle",
            "rightArmAngle", "rightLowerArmAngle", "rightHandAngle",
            "leftLegAngle", "leftLowerLegAngle", "leftFootAngle",
            "rightLegAngle", "rightLowerLegAngle", "rightFootAngle",
            "headAngle"
        ]
        self.height_offsets = [ -0.02, 0.04, 0.08, -0.03, -0.02, 0.04, 0.08, -0.03 ]
        self.angles = [
            [-0.52, -0.15, 0.58, 0.7, 0.52, 0.17, -0.36, -0.74],
            [0.0, -0.16, -0.7, -0.38, -0.47, -0.3, -0.58, -0.21],
            [0.12, 0.0, 0.12, 0.2, 0.0, -0.17, -0.25, 0.0],
            [0.52, 0.17, -0.36, -0.74, -0.52, -0.15, 0.58, 0.7],
            [-0.47, -0.3, -0.58, -0.21, 0.0, -0.16, -0.7, -0.38],
            [0.0, -0.17, -0.25, 0.0, 0.12, 0.0, 0.12, 0.2],
            [-0.55, -0.85, -1.14, -0.7, -0.56, 0.12, 0.24, 0.4],
            [1.4, 1.58, 1.71, 0.49, 0.84, 0.0, 0.14, 0.26],
            [0.07, 0.07, -0.07, -0.36, 0.0, 0.0, 0.32, -0.07],
            [-0.56, 0.12, 0.24, 0.4, -0.55, -0.85, -1.14, -0.7],
            [0.84, 0.0, 0.14, 0.26, 1.4, 1.58, 1.71, 0.49],
            [0.0, 0.0, 0.42, -0.07, 0.07, 0.07, -0.07, -0.36],
            [0.18, 0.09, 0.0, 0.09, 0.18, 0.09, 0.0, 0.09]
        ]
        Supervisor.__init__(self)
        # Enable keyboard (필요시 사용)
        
        self.point_list = ["13 8", "13 8"]  # 시작 목표 좌표들
        self.current_time = 0
        self.angle = 0
        self.action = 0
        self.no_of_steps = 0
        self.current_step = 0
        self.heading = 0  # 초기 진행 방향 (라디안 단위, 0이면 x축 양의 방향)
        
    def Start_up(self):
        self.time_step = int(self.getBasicTimeStep())
        self.number_of_waypoints = len(self.point_list)
        self.waypoints = []
        for i in range(0, self.number_of_waypoints):
            self.waypoints.append([])
            self.waypoints[i].append(float(self.point_list[i].split()[0]))
            self.waypoints[i].append(float(self.point_list[i].split()[1]))
        self.root_node_ref = self.getSelf()
        self.root_translation_field = self.root_node_ref.getField("translation")
        self.root_rotation_field = self.root_node_ref.getField("rotation")
        
        for i in range(0, self.BODY_PARTS_NUMBER):
            self.joints_position_field.append(self.root_node_ref.getField(self.joint_names[i]))

        # compute waypoints distance
        self.waypoints_distance = []
        for i in range(0, self.number_of_waypoints):
            x = self.waypoints[i][0] - self.waypoints[(i + 1) % self.number_of_waypoints][0]
            z = self.waypoints[i][1] - self.waypoints[(i + 1) % self.number_of_waypoints][1]
            if i == 0:
                self.waypoints_distance.append(math.sqrt(x * x + z * z))
            else:
                self.waypoints_distance.append(self.waypoints_distance[i - 1] + math.sqrt(x * x + z * z))
        self.current_time = self.getTime()  # 기존 self.time을 self.current_time으로 변경

    def run(self):
        self.Start_up()
        while not self.step(self.time_step) == -1:
            current_time = self.current_time  # 기존 self.time을 self.current_time으로 변경
            self.keyboardvalue()
            current_sequence = int(((current_time * self.speed) / self.CYCLE_TO_DISTANCE_RATIO) % self.WALK_SEQUENCES_NUMBER)
            ratio = (current_time * self.speed) / self.CYCLE_TO_DISTANCE_RATIO - \
                int(((current_time * self.speed) / self.CYCLE_TO_DISTANCE_RATIO))

            for i in range(0, self.BODY_PARTS_NUMBER):
                current_angle = self.angles[i][current_sequence] * (1 - ratio) + \
                    self.angles[i][(current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio
                self.joints_position_field[i].setSFFloat(current_angle)

            # adjust height
            self.current_height_offset = self.height_offsets[current_sequence] * (1 - ratio) + \
                self.height_offsets[(current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio

            # move everything
            distance = current_time * self.speed  # 기존 self.time을 current_time으로 변경
            relative_distance = distance - int(distance / self.waypoints_distance[self.number_of_waypoints - 1]) * \
                self.waypoints_distance[self.number_of_waypoints - 1]

            for i in range(0, self.number_of_waypoints):
                if self.waypoints_distance[i] > relative_distance:
                    break

            distance_ratio = 0
            if i == 0:
                distance_ratio = relative_distance / self.waypoints_distance[0]
            else:
                distance_ratio = (relative_distance - self.waypoints_distance[i - 1]) / \
                    (self.waypoints_distance[i] - self.waypoints_distance[i - 1])
            # x = distance_ratio * self.waypoints[(i + 1) % self.number_of_waypoints][0] + \
            #     (1 - distance_ratio) * self.waypoints[i][0]
            # z = distance_ratio * self.waypoints[(i + 1) % self.number_of_waypoints][1] + \
            #     (1 - distance_ratio) * self.waypoints[i][1]
            x = distance_ratio * self.waypoints[(i + 1) % self.number_of_waypoints][0] + (1 - distance_ratio) * self.waypoints[i][0]
            y = distance_ratio * self.waypoints[(i + 1) % self.number_of_waypoints][1] + (1 - distance_ratio) * self.waypoints[i][1]

            # root_translation = [x, self.ROOT_HEIGHT + self.current_height_offset, z]
            root_translation = [x, y, 1.35]

            angle = math.atan2(self.waypoints[(i + 1) % self.number_of_waypoints][0] - self.waypoints[i][0],
                               self.waypoints[(i + 1) % self.number_of_waypoints][1] - self.waypoints[i][1])
            
            rotation = [0, 0, 1, self.angle]


            self.root_translation_field.setSFVec3f(root_translation)
            self.root_rotation_field.setSFRotation(rotation)

    def Convert(self):
        temp = self.point_list[-1]
        X, Y = temp.split(' ')
        X = float(X)
        Y = float(Y)
        return X, Y
        
    def keyboardvalue(self):
        # 30% 확률로 90도 단위 방향 전환
        if randint(0, 9) < 3:
            self.heading = choice([0, math.pi/2, math.pi, 3*math.pi/2])  # 상, 우, 하, 좌

        X, Y = self.Convert()
        step_length = 0.2  # 그리드 1칸씩 이동
        newX = X + step_length * math.cos(self.heading)
        newY = Y + step_length * math.sin(self.heading)

        # === 평행이동된 월드에 맞춘 좌표 범위 체크 ===
        x_min, x_max = 0.5, 19.5  # 원래 -9.5 ~ 9.5
        y_min, y_max = 0.5, 15.1  # 원래 -11.8 ~ 3.8

        if newX < x_min or newX > x_max or newY < y_min or newY > y_max:
            self.heading = (self.heading + math.pi) % (2 * math.pi)
            newX = X + step_length * math.cos(self.heading)
            newY = Y + step_length * math.sin(self.heading)

        # === 책상 위치도 평행이동한 좌표로 조정 ===
        margin = 0.2
        desk_x_min = 7.75 - margin  # -2.25 + 10
        desk_x_max = 12.25 + margin
        desk_y_min = 7.25 - margin  # -5.05 + 12.3
        desk_y_max = 8.75 + margin

        if desk_x_min <= newX <= desk_x_max and desk_y_min <= newY <= desk_y_max:
            self.heading = (self.heading + math.pi / 2) % (2 * math.pi)
            newX = X + step_length * math.cos(self.heading)
            newY = Y + step_length * math.sin(self.heading)

        self.point_list[-2] = self.point_list[-1]
        self.point_list[-1] = f"{newX} {newY}"

        self.Start_up()
        self.angle = self.heading
        time.sleep(0.3)


        
    # def keyboardvalue(self):
    #     # 일정 확률(예: 30%)로 진행 방향에 무작위 회전 추가 (-0.1 ~ 0.1 라디안)
    #     if randint(0, 9) < 3:
    #         delta = (randint(-50, 50)) / 100.0  # -0.1 ~ 0.1 라디안 변화
    #         self.heading += delta

    #     # 현재 목표 좌표 (point_list의 마지막 좌표) 가져오기
    #     X, Y = self.Convert()
    #     step_length = 0.2
    #     newX = X + step_length * math.cos(self.heading)
    #     newY = Y + step_length * math.sin(self.heading)

    #     # x 좌표 범위: [-10, 10]
    #     if newX < -10 or newX > 10:
    #         # 수평 방향 성분 반전: heading = π - heading
    #         self.heading = math.pi - self.heading
    #         newX = X + step_length * math.cos(self.heading)
    #         newY = Y + step_length * math.sin(self.heading)  # 반사 후 y 좌표도 재계산

    #     # y 좌표 범위: [-12.3, 4.3]
    #     if newY < -12.3 or newY > 4.3:
    #         # 수직 방향 성분 반전: heading = -heading
    #         self.heading = -self.heading
    #         newX = X + step_length * math.cos(self.heading)
    #         newY = Y + step_length * math.sin(self.heading)

    #     # 목표 좌표 업데이트: 이전 목표 좌표를 보관하고 새 좌표를 설정
    #     self.point_list[-2] = self.point_list[-1]
    #     self.point_list[-1] = f"{newX} {newY}"

    #     # 새로운 목표 좌표로 경로 재계산
    #     self.Start_up()  
    #     # 진행 방향과 동일하게 회전각도 설정
    #     self.angle = self.heading  
    #     time.sleep(0.1)


controller = Pedestrian()
controller.run()

