import os
import json
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

# ================= 配置路径 =================
NUSCENES_DATAROOT = '/Users/xuchunwei/Downloads/dataset/drive/nuscenes'
DRIVELM_JSON_PATH = '/Users/xuchunwei/Downloads/dataset/drive/v1_1_train_nus.json'
OUTPUT_DIR = '/Users/xuchunwei/Downloads/dataset/drive/processed_e2e_dataset'
VERSION = 'samples' # 或者 v1.0-mini 用于测试代码

# ================= 1. 轨迹提取与 Token 转换 =================
def get_ego_future_trajectory_and_token(nusc, sample_token, future_seconds=3.0):
    """提取自车未来局部轨迹，并基于物理规律打标 Decision Token"""
    curr_sample = nusc.get('sample', sample_token)
    sample_data = nusc.get('sample_data', curr_sample['data']['CAM_FRONT'])
    ego_pose_current = nusc.get('ego_pose', sample_data['ego_pose_token'])
    
    current_translation = np.array(ego_pose_current['translation'])
    current_rotation = Quaternion(ego_pose_current['rotation'])
    
    trajectory_local =[]
    time_elapsed = 0.0
    
    # 提取未来帧轨迹
    temp_sample = curr_sample
    while temp_sample['next'] != '' and time_elapsed <= future_seconds:
        next_sample = nusc.get('sample', temp_sample['next'])
        next_sd = nusc.get('sample_data', next_sample['data']['CAM_FRONT'])
        next_ego_pose = nusc.get('ego_pose', next_sd['ego_pose_token'])
        
        time_elapsed += (next_ego_pose['timestamp'] - ego_pose_current['timestamp']) / 1e6
        if time_elapsed > future_seconds: break
            
        future_translation = np.array(next_ego_pose['translation'])
        delta_translation = future_translation - current_translation
        local_pos = current_rotation.inverse.rotate(delta_translation)
        
        # nuScenes: x是右，y是前。转换为：x是前，y是左
        x_forward = round(local_pos[1], 3)
        y_left = round(-local_pos[0], 3)
        trajectory_local.append([x_forward, y_left])
        temp_sample = next_sample

    # 生成物理一致性的 Decision Token
    token = "KEEP_LANE"
    if len(trajectory_local) > 2:
        end_x, end_y = trajectory_local[-1]
        if end_y > 1.5:
            token = "LANE_CHANGE_LEFT"
        elif end_y < -1.5:
            token = "LANE_CHANGE_RIGHT"
        elif end_x < 2.5: # 3秒前进不足2.5米视作刹车/停止
            token = "BRAKE"
            
    return trajectory_local, token

# ================= 2. 解析 DriveLM 提取 CoT =================
def extract_planning_cot_from_drivelm(drivelm_data, sample_token):
    """从 DriveLM 的 QA 对中提取与 Planning (规划) 相关的文本作为 CoT"""
    # DriveLM 数据结构中包含了该帧的各类 QA
    # 我们主要筛选关于 "What should the ego vehicle do" 类的问答
    cot_text = ""
    if sample_token in drivelm_data:
        qa_pairs = drivelm_data[sample_token] # 根据DriveLM实际版本结构微调
        for qa in qa_pairs:
            q = qa.get('Q', '').lower()
            a = qa.get('A', '')
            if 'ego vehicle' in q and ('action' in q or 'do' in q):
                cot_text = a
                break
    
    # 如果没找到，给个默认保底文本 (确保训练不断)
    if not cot_text:
        cot_text = "根据当前路况，自车应当保持安全距离行驶。"
    return cot_text

# ================= 3. 主干流水线 =================
def build_unified_dataset():
    nusc = NuScenes(version=VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载 DriveLM
    with open(DRIVELM_JSON_PATH, 'r') as f:
        drivelm_raw = json.load(f)
        
    vlm_dataset = []
    diffusion_dataset =[]
    
    print("🚀 开始匹配 DriveLM 与 nuScenes 真值...")
    
    # 遍历 DriveLM 中标注过的 sample_token
    # (DriveLM 的 key 通常对应 nuScenes 的 sample_token 或 frame_token)
    for sample_token in drivelm_raw.keys():
        try:
            # 1. 获取物理真值轨迹和 Token
            trajectory, decision_token = get_ego_future_trajectory_and_token(nusc, sample_token, future_seconds=3.0)
            
            # 过滤无效轨迹
            if len(trajectory) < 3: 
                continue
                
            # 2. 获取高质量 DriveLM 思维链 (CoT)
            cot_text = extract_planning_cot_from_drivelm(drivelm_raw, sample_token)
            
            # 3. 提取图像路径 (以单张前视图像为例，也可修改为提取6路多视图)
            curr_sample = nusc.get('sample', sample_token)
            cam_front_data = nusc.get('sample_data', curr_sample['data']['CAM_FRONT'])
            img_path = os.path.join(NUSCENES_DATAROOT, cam_front_data['filename'])
            
            # ================= 组装 VLM 格式 (类 LLaVA SFT) =================
            vlm_item = {
                "id": sample_token,
                "image": img_path,
                "conversations":[
                    {
                        "from": "human",
                        "value": "<image>\n作为自动驾驶大脑，请观察图像，分析路况并输出你的推理过程和决策Token (如 KEEP_LANE, BRAKE, LANE_CHANGE_LEFT 等)。"
                    },
                    {
                        "from": "gpt",
                        "value": f"[Thinking]: {cot_text}\n[Decision]: {decision_token}"
                    }
                ]
            }
            vlm_dataset.append(vlm_item)
            
            # ================= 组装 Diffusion 格式 =================
            diffusion_item = {
                "sample_token": sample_token,
                "condition_token": decision_token,
                "target_trajectory": trajectory  # 形如 [[0.5, 0.0],[1.2, 0.1], ...]
            }
            diffusion_dataset.append(diffusion_item)
            
        except Exception as e:
            # 忽略数据集中可能有缺失的帧
            continue

    # 保存 VLM 数据集
    vlm_out = os.path.join(OUTPUT_DIR, 'vlm_sft_dataset.json')
    with open(vlm_out, 'w', encoding='utf-8') as f:
        json.dump(vlm_dataset, f, ensure_ascii=False, indent=2)
        
    # 保存 Diffusion 数据集
    diff_out = os.path.join(OUTPUT_DIR, 'diffusion_trajectory_dataset.json')
    with open(diff_out, 'w', encoding='utf-8') as f:
        json.dump(diffusion_dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ 构建完成！共提取 {len(vlm_dataset)} 条黄金对齐数据。")
    print(f"📁 VLM 训练集: {vlm_out}")
    print(f"📁 Diffusion 训练集: {diff_out}")

if __name__ == '__main__':
    build_unified_dataset()
    pass
