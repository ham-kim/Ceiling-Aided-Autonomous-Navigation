cmake_minimum_required(VERSION 3.8)
project(custom_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# === 필수 패키지 찾기 ===
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# === 메시지 인터페이스 생성 ===
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/HumanRobotInfo.msg"
  DEPENDENCIES std_msgs
)

# === 런타임 의존성 ===
ament_export_dependencies(rosidl_default_runtime)

# === 테스트 설정 ===
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
