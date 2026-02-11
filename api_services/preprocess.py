"""
全景图像预处理API服务
基于crop_visualizer.py
"""

import os
import sys
import logging
import argparse
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import json
import cv2

# 添加PreciseCam路径
current_dir = os.path.dirname(os.path.abspath(__file__))
precisecam_path = os.path.join(current_dir, '..', '..', 'PreciseCam')
sys.path.append(precisecam_path)

try:
    from perspective_fields import pano_utils as pu
    logger = logging.getLogger(__name__)
    logger.info("✅ PreciseCam模块导入成功")
    logger.info(f"   PreciseCam路径: {precisecam_path}")
        
except ImportError as e:
    logging.error(f"无法导入PreciseCam模块: {e}")
    logging.error(f"当前路径: {os.getcwd()}")
    logging.error(f"PreciseCam路径: {precisecam_path}")
    logging.error(f"Python路径: {sys.path}")
    sys.exit(1)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class PanoramaPreprocessor:
    """全景图像预处理器 - 基于crop_visualizer.py"""
    
    def __init__(self, output_dir="output/preprocess"):
        # 确保使用绝对路径
        if not os.path.isabs(output_dir):
            # 获取当前脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 回到full_process目录
            full_process_dir = os.path.dirname(current_dir)
            output_dir = os.path.join(full_process_dir, output_dir)
        
        self.output_dir = output_dir
        logger.info(f"Preprocess API输出目录: {self.output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    def crop_panorama_with_params(self, pil_img, yaw, pitch, roll, vfov, xi, resize_factor=1.0):
        """裁剪全景图像 - 基于crop_visualizer.py"""
        if pil_img is None:
            return None, "无图片"
        try:
            panorama = np.array(pil_img)
            logger.info(f"图像尺寸: {panorama.shape}")
            if len(panorama.shape) == 3 and panorama.shape[2] == 3:
                panorama = cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR)
            if resize_factor != 1.0:
                h, w = panorama.shape[:2]
                new_h, new_w = int(h * resize_factor), int(w * resize_factor)
                new_h, new_w = (512,512)
                panorama = cv2.resize(panorama, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logger.info(f"图像缩放: {h}x{w} -> {new_h}x{new_w}")
            h, w = panorama.shape[:2]
            logger.info(f"裁剪参数: yaw={yaw}, pitch={pitch}, roll={roll}, vfov={vfov}, xi={xi}")
            x = -np.sin(np.radians(vfov / 2))
            z = np.sqrt(1 - x**2)
            f_px_effective = -0.5 * (w / 2) * (xi + z) / x
            logger.info(f"计算参数: f_px_effective={f_px_effective}")
            crop, *_ = pu.crop_distortion(
                panorama, f=f_px_effective, xi=xi, H=h, W=w, az=yaw, el=-pitch, roll=roll)
            logger.info(f"裁剪成功，结果尺寸: {crop.shape}")
            return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), "裁剪成功(缩放:{})".format(resize_factor)
        except Exception as e:
            logger.error(f"裁剪失败: {str(e)}")
            logger.error(f"参数: yaw={yaw}, pitch={pitch}, roll={roll}, vfov={vfov}, xi={xi}")
            return None, f"裁剪失败: {str(e)}"

    def randomize_params(self, yaw, pitch, roll, vfov, xi):
        """随机化参数 - 在基准参数±40度范围内连续随机"""
        # 在基准参数±40度范围内连续随机
        new_pitch_raw = pitch + np.random.uniform(-40, 40)
        new_vfov_raw = vfov + np.random.uniform(-40, 40)
        
        # 应用clip限制
        new_pitch = np.clip(new_pitch_raw, -90, 90)
        new_vfov = np.clip(new_vfov_raw, 30, 140)
        
        # roll和xi在全范围内连续随机
        new_roll = np.random.uniform(-90, 90)
        new_xi = np.random.uniform(0.0, 1.0)

        return {
            "yaw": float(yaw),  # yaw保持不变，不随机变动
            "pitch": float(new_pitch),
            "roll": float(new_roll),
            "vfov": float(new_vfov),
            "xi": float(new_xi),
        }

    def process_for_group(self, image_path: str, group_id: int, is_first: bool):
        """
        为图片组处理全景图像
        
        参数:
            image_path: 全景图像路径
            group_id: 图片组ID（1, 2, 3...）
            is_first: 是否为第一张图片
        """
        logger.info(f"处理图片组 {group_id} 的{'第一张' if is_first else '第二张'}全景图: {image_path}")
        
        # 创建组号目录及其子目录 - 直接创建在output_dir的同级
        # self.output_dir应该是类似 test_output/ 这样的目录
        # 我们想在 test_output/ 下创建 group_0001/
        group_dir_name = f"group_{group_id:04d}"
        # 如果output_dir是 "output/preprocess"，我们要回到上一层，创建 group_0001
        # 否则直接在output_dir下创建group_0001
        if 'preprocess' in self.output_dir:
            # 在output_dir的父目录下创建group
            parent_dir = os.path.dirname(self.output_dir)
            group_root_dir = os.path.join(parent_dir, group_dir_name)
        else:
            # 直接在output_dir下创建group
            group_root_dir = os.path.join(self.output_dir, group_dir_name)
        group_root_dir = os.path.abspath(group_root_dir)
        
        # 创建组目录结构
        group_pairs_dir = os.path.join(group_root_dir, "pairs")
        group_params_dir = os.path.join(group_root_dir, "params")
        
        # 创建pairs和params目录（预处理时需要）
        os.makedirs(group_pairs_dir, exist_ok=True)
        os.makedirs(group_params_dir, exist_ok=True)
        
        # 读取图像
        pil_img = Image.open(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 处理8个yaw区间（每45度一个区间，总共生成8个pair）
        yaw_intervals = [
            (-180, -135),   # 区间1
            (-135, -90),    # 区间2
            (-90, -45),     # 区间3
            (-45, 0),       # 区间4
            (0, 45),        # 区间5
            (45, 90),       # 区间6
            (90, 135),      # 区间7
            (135, 180)      # 区间8
        ]
        
        results = []
        
        # 如果是第一张图片，生成并保存共享的右图参数和第一张图的左图参数A
        if is_first:
            # 生成1组共享的右图参数（两张图片共用） - 连续型
            shared_right_params = []  # 保存8个yaw区间的右图参数
            panorama1_left_params = []  # 保存第一张图的左图参数A
            
            for interval_idx, (yaw_min, yaw_max) in enumerate(yaw_intervals):
                # 连续随机生成yaw值
                yaw_value = np.random.uniform(yaw_min, yaw_max)
                
                # 连续随机生成右图参数
                right_params = {
                    "yaw": float(yaw_value),
                    "pitch": float(np.random.uniform(-90, 90)),
                    "roll": float(np.random.uniform(-90, 90)),
                    "vfov": float(np.random.uniform(30, 140)),
                    "xi": float(np.random.uniform(0.0, 1.0))
                }
                shared_right_params.append(right_params)
                
                # 为第一张图生成左图参数A（基于右图参数）
                left_params_A = self.randomize_params(
                    right_params["yaw"], right_params["pitch"], right_params["roll"],
                    right_params["vfov"], right_params["xi"]
                )
                panorama1_left_params.append(left_params_A)
            
            # 保存共享的右图参数到文件
            shared_params_file = os.path.join(group_params_dir, "shared_right_params.json")
            with open(shared_params_file, 'w', encoding='utf-8') as f:
                json.dump(shared_right_params, f, indent=4, ensure_ascii=False)
            logger.info(f"保存共享右图参数到: {shared_params_file}")
            
            # 保存第一张图的左图参数A到文件
            panorama1_left_params_file = os.path.join(group_params_dir, "panorama1_left_params.json")
            with open(panorama1_left_params_file, 'w', encoding='utf-8') as f:
                json.dump(panorama1_left_params, f, indent=4, ensure_ascii=False)
            logger.info(f"保存第一张图左图参数A到: {panorama1_left_params_file}")
        else:
            # 加载第一张图片保存的右图参数和左图参数A
            shared_params_file = os.path.join(group_params_dir, "shared_right_params.json")
            panorama1_left_params_file = os.path.join(group_params_dir, "panorama1_left_params.json")
            
            if not os.path.exists(shared_params_file):
                raise FileNotFoundError(f"找不到共享参数文件: {shared_params_file}")
            if not os.path.exists(panorama1_left_params_file):
                raise FileNotFoundError(f"找不到第一张图左图参数文件: {panorama1_left_params_file}")
            
            with open(shared_params_file, 'r', encoding='utf-8') as f:
                shared_right_params = json.load(f)
            with open(panorama1_left_params_file, 'r', encoding='utf-8') as f:
                panorama1_left_params = json.load(f)
            logger.info(f"加载共享右图参数: {shared_params_file}")
            logger.info(f"加载第一张图左图参数A: {panorama1_left_params_file}")
        
        # 为每个yaw区间生成图像对
        for interval_idx, (yaw_min, yaw_max) in enumerate(yaw_intervals):
            # 使用共享的右图参数（第一张）或加载的参数（第二张）
            rand_params = shared_right_params[interval_idx].copy()
            
            # 生成左图参数
            if is_first:
                # 第一张图：使用已生成的左图参数A
                main_params = panorama1_left_params[interval_idx].copy()
            else:
                # 第二张图：基于参数A生成参数B（±20度/±0.2）
                left_params_A = panorama1_left_params[interval_idx]
                
                # 在A的基础上生成B
                pitch_B_raw = left_params_A["pitch"] + np.random.uniform(-20, 20)
                roll_B_raw = left_params_A["roll"] + np.random.uniform(-20, 20)
                vfov_B_raw = left_params_A["vfov"] + np.random.uniform(-20, 20)
                xi_B_raw = left_params_A["xi"] + np.random.uniform(-0.2, 0.2)
                
                # 应用clip限制
                main_params = {
                    "yaw": float(rand_params["yaw"]),  # yaw与右图保持一致
                    "pitch": float(np.clip(pitch_B_raw, -90, 90)),
                    "roll": float(np.clip(roll_B_raw, -90, 90)),
                    "vfov": float(np.clip(vfov_B_raw, 30, 140)),
                    "xi": float(np.clip(xi_B_raw, 0.0, 1.0))
                }
            
            # 裁剪左图（使用独立参数）
            main_crop, main_msg = self.crop_panorama_with_params(
                pil_img, main_params["yaw"], main_params["pitch"], main_params["roll"], 
                main_params["vfov"], main_params["xi"], resize_factor=0.5
            )
            if main_crop is None:
                logger.warning(f"区间{interval_idx+1}左图裁剪失败: {main_msg}")
                continue
            
            # 裁剪右图（使用共享参数）
            rand_crop, rand_msg = self.crop_panorama_with_params(
                pil_img, rand_params["yaw"], rand_params["pitch"], rand_params["roll"], 
                rand_params["vfov"], rand_params["xi"], resize_factor=0.5
            )
            if rand_crop is None:
                logger.warning(f"区间{interval_idx+1}右图裁剪失败: {rand_msg}")
                continue
            
            # 创建图像对（水平拼接）
            pair_width = main_crop.width + rand_crop.width
            pair_height = max(main_crop.height, rand_crop.height)
            pair_img = Image.new('RGB', (pair_width, pair_height))
            pair_img.paste(main_crop, (0, 0))
            pair_img.paste(rand_crop, (main_crop.width, 0))
            
            # 保存图像对到组号目录
            interval_name = f"{base_name}_yaw{interval_idx+1:02d}"
            pair_filename = f"{interval_name}_pair.jpg"
            pair_path = os.path.join(group_pairs_dir, pair_filename)
            logger.info(f"保存图像对: {pair_path}")
            pair_img.save(pair_path)
            
            # 保存裁剪参数JSON文件到params目录
            main_params_path = os.path.join(group_params_dir, f"{interval_name}_left_params.json")
            rand_params_path = os.path.join(group_params_dir, f"{interval_name}_right_params.json")
            
            with open(main_params_path, "w", encoding='utf-8') as f:
                json.dump(main_params, f, indent=4, ensure_ascii=False)
            with open(rand_params_path, "w", encoding='utf-8') as f:
                json.dump(rand_params, f, indent=4, ensure_ascii=False)
            
            result = {
                "pair_image": pair_path,
                "main_params": main_params,
                "rand_params": rand_params,
                "main_params_file": main_params_path,
                "rand_params_file": rand_params_path,
                "interval": interval_idx + 1,
                "yaw_interval": (yaw_min, yaw_max),
                "group_id": group_id
            }
            results.append(result)
        
        logger.info(f"图片组 {group_id} 处理完成: {image_path}, 生成了{len(results)}个图像对")
        return results
    
    def generate_interpolated_images_for_split(self, split_results: list, panorama1_path: str, panorama2_path: str, group_id: int):
        """为切分结果生成插值图像"""
        logger.info(f"开始生成插值图像，共有 {len(split_results)} 组数据")
        
        interpolated_results = []
        
        for split_result in split_results:
            try:
                interval = split_result.get('interval')
                main_params = split_result.get('main_params', {})
                rand_params = split_result.get('rand_params', {})
                
                # 确定这是panorama1还是panorama2的数据
                pair_image = split_result.get('pair_image', '')
                
                # 判断是panorama1还是panorama2
                # 从pair_image文件名中提取全景图信息：如 utopian_city_360.0120_yaw01_pair.jpg
                pair_basename = os.path.splitext(os.path.basename(pair_image))[0]
                panorama1_basename = os.path.splitext(os.path.basename(panorama1_path))[0]
                panorama2_basename = os.path.splitext(os.path.basename(panorama2_path))[0]
                
                # 检查pair_basename是否以panorama1或panorama2的basename开头
                if pair_basename.startswith(panorama1_basename):
                    panorama_path = panorama1_path
                    is_panorama1 = True
                elif pair_basename.startswith(panorama2_basename):
                    panorama_path = panorama2_path
                    is_panorama1 = False
                else:
                    # 如果无法判断，尝试从left_image路径判断
                    left_image = split_result.get('left_image', '')
                    if panorama1_basename in left_image:
                        panorama_path = panorama1_path
                        is_panorama1 = True
                    else:
                        panorama_path = panorama2_path
                        is_panorama1 = False
                
                # 读取全景图
                panorama_img = Image.open(panorama_path)
                panorama = np.array(panorama_img)
                if len(panorama.shape) == 3 and panorama.shape[2] == 3:
                    panorama = cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR)
                
                # 生成9组插值参数（0.9-0.1, 0.8-0.2, ..., 0.1-0.9）
                # 在main_params（左图）和rand_params（右图）之间插值
                # ToDo
                interpolation_weights = [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5), 
                                        (0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9)]
                
                # 只生成一组插值图像（从左到右的插值序列）
                interpolated_images = []
                
                for weight_idx, (w1, w2) in enumerate(interpolation_weights):
                    # 计算插值参数：在main_params和rand_params之间插值
                    interp_params = {
                        'yaw': w1 * main_params['yaw'] + w2 * rand_params['yaw'],
                        'pitch': w1 * main_params['pitch'] + w2 * rand_params['pitch'],
                        'roll': w1 * main_params['roll'] + w2 * rand_params['roll'],
                        'vfov': w1 * main_params['vfov'] + w2 * rand_params['vfov'],
                        'xi': w1 * main_params['xi'] + w2 * rand_params['xi']
                    }
                    
                    # 使用插值参数裁剪全景图
                    # 首先缩放全景图到512x512（与原始裁剪保持一致）
                    h, w = panorama.shape[:2]
                    new_h, new_w = (512, 512)
                    resized_panorama = cv2.resize(panorama, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    x = -np.sin(np.radians(interp_params['vfov'] / 2))
                    z = np.sqrt(1 - x**2)
                    f_px_effective = -0.5 * (new_w / 2) * (interp_params['xi'] + z) / x
                    
                    # 裁剪图像
                    crop, *_ = pu.crop_distortion(
                        resized_panorama, f=f_px_effective, xi=interp_params['xi'], 
                        H=new_h, W=new_w, az=interp_params['yaw'], 
                        el=-interp_params['pitch'], roll=interp_params['roll']
                    )
                    
                    # 保存插值图像
                    base_name = os.path.splitext(os.path.basename(split_result.get('left_image', '')))[0]
                    interp_filename = f"{base_name}_interp_{weight_idx+1:02d}.jpg"
                    
                    # 提取组目录
                    import re
                    left_img_path = split_result.get('left_image', '')
                    match = re.search(r'/group_\d+/', left_img_path)
                    if match:
                        # 提取匹配到的目录路径 (如 /group_0001/)
                        group_dir_match = match.group(0)
                        # 找到这个目录之前的部分
                        group_root_dir = left_img_path[:left_img_path.find(group_dir_match) + len(group_dir_match) - 1]
                        group_root_dir = os.path.abspath(group_root_dir)
                        interpolated_dir = os.path.join(group_root_dir, "interpolated")
                        os.makedirs(interpolated_dir, exist_ok=True)
                        interp_path = os.path.join(interpolated_dir, interp_filename)
                        
                        # 转换并保存
                        crop_rgb = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        crop_rgb.save(interp_path)
                        
                        interpolated_images.append({
                            'path': interp_path,
                            'weight_idx': weight_idx + 1,
                            'params': interp_params
                        })
                        
                        logger.info(f"生成插值图像: {interp_filename}")
                
                # 存储插值图像结果
                interpolated_results.append({
                    'panorama': 'panorama1' if is_panorama1 else 'panorama2',
                    'interval': interval,
                    'interpolated_images': interpolated_images,
                    'main_params': main_params,
                    'rand_params': rand_params
                })
                
            except Exception as e:
                logger.error(f"生成插值图像失败: {e}")
                continue
        
        logger.info(f"插值图像生成完成，共 {len(interpolated_results)} 组结果")
        return interpolated_results

# 全局预处理器实例
preprocessor = None

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({"status": "healthy", "service": "preprocess"})

@app.route('/preprocess_for_group', methods=['POST'])
def preprocess_for_group():
    """为图片组预处理全景图像（支持右图参数共享）"""
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        group_id = data.get('group_id')
        is_first = data.get('is_first', True)
        
        if not image_path:
            return jsonify({"error": "缺少image_path参数"}), 400
        if group_id is None:
            return jsonify({"error": "缺少group_id参数"}), 400
        
        results = preprocessor.process_for_group(image_path, group_id, is_first)
        return jsonify({"success": True, "results": results})
        
    except Exception as e:
        logger.error(f"图片组预处理失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_interpolated_images', methods=['POST'])
def generate_interpolated_images():
    """为切分结果生成插值图像"""
    try:
        data = request.get_json()
        split_results = data.get('split_results', [])
        panorama1_path = data.get('panorama1_path', '')
        panorama2_path = data.get('panorama2_path', '')
        group_id = data.get('group_id', 0)
        
        if not split_results:
            return jsonify({"error": "缺少split_results参数"}), 400
        if not panorama1_path or not panorama2_path:
            return jsonify({"error": "缺少panorama路径参数"}), 400
        if not group_id:
            return jsonify({"error": "缺少group_id参数"}), 400
        
        results = preprocessor.generate_interpolated_images_for_split(
            split_results, panorama1_path, panorama2_path, group_id
        )
        return jsonify({"success": True, "results": results})
        
    except Exception as e:
        logger.error(f"生成插值图像失败: {e}")
        return jsonify({"error": str(e)}), 500

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='全景图像预处理API服务')
    parser.add_argument('--port', type=int, default=5000, help='服务端口')
    parser.add_argument('--output_dir', type=str, default='output/preprocess', help='输出目录')
    args = parser.parse_args()
    
    # 初始化预处理器
    global preprocessor
    preprocessor = PanoramaPreprocessor(args.output_dir)
    
    logger.info(f"全景图像预处理API服务启动 - 端口: {args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=False)

if __name__ == '__main__':
    main() 