# Ceiling-Aided Autonomous Navigation (CAAN)

Ceiling-Aided Autonomous Navigation (CAAN) is a vision-based localization system that improves indoor robot navigation by utilizing ceiling markers.

This project was developed for a robotics competition and focuses on enhancing localization robustness in indoor environments where traditional SLAM may suffer from drift or occlusion.

---

## 📌 Overview

CAAN introduces a ceiling-based visual localization approach to support autonomous mobile robots.  
Instead of relying only on LiDAR or ground-level features, the system detects ceiling markers to:

- Improve pose estimation accuracy
- Reduce localization drift
- Maintain navigation stability in dynamic indoor environments

---

## 🧠 System Architecture

- **Robot Platform**: Mobile robot (TurtleBot / Custom platform)
- **Framework**: ROS2
- **Localization**: Ceiling marker detection + pose estimation
- **Navigation**: Nav2 stack
- **Vision Processing**: Marker detection & coordinate transformation

---

## ⚙️ Features

- Ceiling marker-based global position correction
- Integration with ROS2 Nav2 stack
- Improved robustness in feature-poor environments
- Modular architecture for easy integration

---

## Launch Navigation
- ros2 launch <your_package> <your_launch_file>.launch.py

## 📸 Application

- Indoor autonomous robots

- Warehouse automation

- Competition robotics

- SLAM enhancement systems


