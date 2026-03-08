import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import queue

# ================= 配置区域 =================
CARLA_VERSION = '0.9.13' # 根据你的版本修改，或者让下面的代码自动寻找
OUTPUT_DIR = "/Users/xuchunwei/Downloads/dataset/world/dataset_v1"
IMG_SIZE = 128           # 128x128 适合快速训练验证
TOTAL_FRAMES = 10000     # 先跑 1万帧 (约15分钟) 验证流程，没问题再挂机跑 10万帧
NOISE_PROB = 0.1         # 10% 的概率加入噪声
NOISE_INTENSITY = 0.2    # 噪声强度 (方向盘抖动幅度)
# ===========================================

# 尝试自动查找 CARLA egg 文件
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def main():
    # 1. 创建输出目录
    if not os.path.exists(os.path.join(OUTPUT_DIR, 'images')):
        os.makedirs(os.path.join(OUTPUT_DIR, 'images'))
    
    # 2. 连接客户端
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    world = client.get_world()
    
    # 3. 设置同步模式 (关键!)
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True 
    settings.fixed_delta_seconds = 0.1 # 10FPS，对于 World Model 足够，且能采集更多变化
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    # 稍微“狂野”一点的驾驶风格，避免过度保守
    traffic_manager.global_percentage_speed_difference(30.0) 

    actor_list = []
    sensor_queue = queue.Queue()

    try:
        # 4. 生成主角车
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('model3')[0] # Tesla Model 3
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        
        # 如果生成失败，换个点
        while vehicle is None:
            vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        
        actor_list.append(vehicle)
        vehicle.set_autopilot(True) # 开启基础自动驾驶

        # 5. 安装 RGB 摄像头
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IMG_SIZE))
        camera_bp.set_attribute('image_size_y', str(IMG_SIZE))
        camera_bp.set_attribute('fov', '90')
        
        # 安装在车头位置 (第一人称视角)
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)

        # 监听摄像头数据
        camera.listen(lambda image: sensor_queue.put(image))

        print(f"Start recording to {OUTPUT_DIR}...")
        
        # 打开 CSV 文件准备写入
        with open(os.path.join(OUTPUT_DIR, 'log.csv'), 'w') as f:
            f.write("frame_id,steering,throttle,brake\n") # Header

            for frame_id in range(TOTAL_FRAMES):
                # Tick 世界
                world.tick()
                
                # 获取图像 (带超时保护)
                try:
                    image = sensor_queue.get(timeout=2.0)
                except queue.Empty:
                    print("Sensor missed a tick!")
                    continue

                # 获取当前的控制状态 (由 Autopilot 产生)
                control = vehicle.get_control()
                
                # === 注入噪声 (Critical for World Model) ===
                # 如果只学完美的 Autopilot，模型一旦预测偏了就回不来了。
                # 我们偶尔人为抖动一下方向盘，模拟“修正”。
                # 注意：这里我们只记录修改后的值，并在下一帧生效，
                # 但由于是 Autopilot，它会在下一帧瞬间修正回来，这正好产生了"偏离->修正"的数据对。
                
                modified_steer = control.steer
                if random.random() < NOISE_PROB:
                    noise = random.uniform(-NOISE_INTENSITY, NOISE_INTENSITY)
                    modified_steer = np.clip(modified_steer + noise, -1.0, 1.0)
                    # 强制应用噪声控制 (覆盖 Autopilot 这一帧的意图)
                    # 注意：这可能会和 Autopilot 打架，但为了数据多样性是值得的
                    override_control = carla.VehicleControl(
                        throttle=control.throttle,
                        steer=modified_steer,
                        brake=control.brake
                    )
                    vehicle.apply_control(override_control)
                
                # === 保存数据 ===
                # 1. 保存图片
                image_path = os.path.join(OUTPUT_DIR, 'images', f'{frame_id:06d}.jpg')
                # CARLA image raw data to numpy
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image.height, image.width, 4)) # RGBA
                array = array[:, :, :3] # 去掉 Alpha 通道
                cv2.imwrite(image_path, array)

                # 2. 保存 Action (注意：我们记录的是这帧图像对应的状态，或者是导致下一帧的动作)
                # 在 World Model 中，通常是 Image_t + Action_t -> Image_t+1
                # 所以我们记录当前的 modified_steer
                f.write(f"{frame_id},{modified_steer:.4f},{control.throttle:.4f},{control.brake:.4f}\n")

                if frame_id % 100 == 0:
                    print(f"Recorded {frame_id}/{TOTAL_FRAMES} frames. Steer: {modified_steer:.2f}")

    finally:
        print("Cleaning up...")
        # 恢复设置
        world.apply_settings(original_settings)
        # 销毁物体
        for actor in actor_list:
            actor.destroy()
        print("Done.")

if __name__ == '__main__':
    main()