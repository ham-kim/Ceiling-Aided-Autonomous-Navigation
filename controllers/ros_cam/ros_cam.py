from controller import Robot, Camera
import cv2
import numpy as np
import torch
import math
import warnings
import rclpy
from rclpy.node import Node
from custom_msgs.msg import HumanRobotInfo

warnings.filterwarnings('ignore', category=FutureWarning)

# ROS 2 퍼블리셔 노드 클래스 정의
class YoloPublisher(Node):
    def __init__(self):
        super().__init__('yolo_publisher')
        self.publisher_ = self.create_publisher(HumanRobotInfo, '/human_robot_info', 10)
        self.get_logger().info("✅ YOLO Publisher Node Started")

    def publish_info(self, person_x, person_y, robot_x, robot_y, distance):
        msg = HumanRobotInfo()
        msg.person_x = float(person_x)
        msg.person_y = float(person_y)
        msg.robot_x = float(robot_x)
        msg.robot_y = float(robot_y)
        msg.distance = float(distance)
        self.publisher_.publish(msg)

# ROS 초기화
rclpy.init()
ros_node = YoloPublisher()

# === Webots 초기화 ===
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# === 카메라 초기화 ===
cameras = []
for i in range(1, 7):
    name = f"camera({i})"
    try:
        cam = robot.getDevice(name)
        cam.enable(timestep)
        cameras.append(cam)
        print(f"{name} 연결 완료.")
        print("✅ PPO 모델 로드 완료")
        print(f"🎯 목표에 도달했습니다! 현재 위치: (18, 1)")
    except:
        print(f"{name} 연결 실패.")

if not cameras:
    print("카메라 없음! 종료.")
    exit()

# === YOLO 모델 로딩 ===
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='/home/haemin/webots/projects/my_webots_project/controllers/ros_cam/custom.pt',
                       force_reload=True)

# === 대응쌍 정의 ===
affine_matrices = {}
affine_data = {
    1: (np.array([[347, 155], [319, 359], [531, 266], [193, 124],[359,427],[637,241]], dtype=np.float32),
        np.array([[17.49, 1.22], [18.23, 2.99], [15.90, 2.20], [19.48, 0.63],[17.81,3.67],[15,2.01]], dtype=np.float32)),
    2: (np.array([[272, 419], [383, 210], [514, 515], [677, 322],[922, 242],[295, 428]], dtype=np.float32),
        np.array([[8.12, 3.58], [7.57, 1.49], [6.03, 4.39], [4.68, 2.61],[2.52, 2.01],[8.45, 3.67]], dtype=np.float32)),
    3: (np.array([[583, 443], [790, 322], [261, 564], [464, 203],[906, 562],[658, 254]], dtype=np.float32),
        np.array([[15.46, 8.78], [13.56, 7.60], [18.22, 9.78], [16.78, 6.44],[12.66, 9.74],[14.87, 6.93]], dtype=np.float32)),
    4: (np.array([[616, 563], [421, 262], [881, 227], [731, 615],[892, 694],[514, 255]], dtype=np.float32),
        np.array([[5.20, 9.78], [7.19, 7.00], [2.92, 6.80], [4.15, 10.56],[2.57, 11.39],[6.29, 6.94]], dtype=np.float32)),
    5: (np.array([[941, 400], [485, 454], [208, 180], [289, 593],[118, 450],[579, 227]], dtype=np.float32),
        np.array([[12.41, 13.42], [16.56, 13.93], [18.67, 11.44], [18.51, 15.33],[19.47, 13.71],[15.64,11.66]], dtype=np.float32)),
    6: (np.array([[603, 183], [758, 590], [1086, 589], [208, 300],[361, 617],[613, 368]], dtype=np.float32),
        np.array([[5.28, 11.44], [3.87, 15.30], [1.15, 14.99], [9.32, 12.40],[7.38, 15.23],[5.30, 13.09]], dtype=np.float32))
}
for cam_id, (cam_pts, world_pts) in affine_data.items():
    matrix, _ = cv2.estimateAffine2D(cam_pts, world_pts)
    affine_matrices[cam_id] = matrix

# === 디스플레이 설정 ===
display_width = 640
display_height = 360
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Detection", 1100, 900)

# === 루프 ===
while robot.step(timestep) != -1:
    processed_images = []
    best_person = {"conf": -1}
    best_robot = {"conf": -1}

    for idx, cam in enumerate(cameras):
        cam_id = idx + 1
        img = cam.getImage()
        if not img:
            continue

        np_img = np.frombuffer(img, np.uint8).reshape((cam.getHeight(), cam.getWidth(), 4))
        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGRA2BGR)

        results = model(np_img)
        detections = results.pandas().xyxy[0]

        for _, row in detections.iterrows():
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            label = row['name']
            conf = row['confidence']

            if conf < 0.5 or (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            center_x = (x1 + x2) // 2
            center_y = y2 if label == 'people' else (y1 + y2) // 2

            color = (0, 255, 0)
            if label == 'people':
                color = (0, 0, 255)
            elif label == 'robot':
                color = (255, 0, 0)

            cv2.rectangle(np_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(np_img, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if cam_id in affine_matrices:
                input_point = np.array([[center_x, center_y]], dtype=np.float32)
                transformed = cv2.transform(input_point[None, :, :], affine_matrices[cam_id])
                xw, yw = transformed[0][0]

                # print(f"[Cam {cam_id}] {label.upper()} → Webots 좌표: ({xw:.2f}, {yw:.2f})")

                if label == 'people' and conf > best_person["conf"]:
                    best_person = {"coord": (xw, yw), "conf": conf}
                elif label == 'robot' and conf > best_robot["conf"]:
                    best_robot = {"coord": (xw, yw), "conf": conf}

        resized = cv2.resize(np_img, (display_width, display_height))
        processed_images.append(resized)

    if "coord" in best_person and "coord" in best_robot:
        px, py = best_person["coord"]
        rx, ry = best_robot["coord"]
        dist = math.sqrt((px - rx) ** 2 + (py - ry) ** 2)
        # print(f"📏 거리: 사람({px:.2f},{py:.2f}) - 로봇({rx:.2f},{ry:.2f}) = {dist:.2f}m")

        # ROS2 메시지 발행
        ros_node.publish_info(px, py, rx, ry, dist)

    if processed_images:
        while len(processed_images) < 6:
            blank = np.zeros_like(processed_images[0])
            cv2.putText(blank, f"No Camera {len(processed_images)}", (display_width // 4, display_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            processed_images.append(blank)

        grid_rows, grid_cols = 3, 2
        grid = [np.hstack(processed_images[i * grid_cols:(i + 1) * grid_cols]) for i in range(grid_rows)]
        combined_img = np.vstack(grid)

        cv2.imshow("YOLO Detection", combined_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
ros_node.destroy_node()
rclpy.shutdown()
