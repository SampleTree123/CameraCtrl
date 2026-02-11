"""
全景图像处理器核心类
协调所有API服务完成完整的处理流程
"""

import os
import sys
import logging
import time
import json
import shutil
from typing import List, Dict
from PIL import Image

# 添加项目路径（必须在 from utils/config 之前）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(current_dir, '..')
sys.path.insert(0, project_dir)

from utils.api_utils import APIClient


logger = logging.getLogger(__name__)

class PanoramaProcessor:
    """全景图像处理器 - 协调所有API服务"""
    
    def __init__(self, output_root_dir: str = "output", api_version: str = "original", output_file: str = "results.json", num_interpolations: int = 9):
        """
        初始化处理器
        
        Args:
            output_root_dir: 输出根目录
            api_version: API版本选择，'original' 或 'shared_left'
            output_file: 输出文件名（默认 results.json）
            num_interpolations: 插值图像数量（默认9），控制在两张图之间生成多少个插值帧
        """
        self.output_root_dir = output_root_dir
        self.api_version = api_version
        self.output_file = output_file
        self.num_interpolations = num_interpolations
        
        # 使用统一配置
        from config.api_config import get_api_ports, get_gpu_id, API_BASE_URL
        
        self.API_PORTS = get_api_ports(api_version)
        self.API_BASE_URL = API_BASE_URL
        gpu_id = get_gpu_id(api_version)
        
        logger.info(f"🔄 使用 {api_version} 版本API (端口 {self.API_PORTS['preprocess']}-{self.API_PORTS['quality']}, GPU {gpu_id})")
        
        self.api_client = APIClient(API_BASE_URL)
        
        os.makedirs(self.output_root_dir, exist_ok=True)
        logger.info(f"创建输出根目录: {self.output_root_dir}")
    
    # ==================== 工具方法 ====================
    
    def _extract_group_dir(self, file_path: str) -> str:
        """从文件路径中提取 group_xxxx 目录的绝对路径"""
        import re
        normalized = file_path.replace('\\', '/')
        match = re.search(r'/group_\d+/', normalized)
        if match:
            pos = normalized.find(match.group(0))
            group_root_dir = file_path[:pos + len(match.group(0)) - 1]
            return os.path.abspath(group_root_dir)
        raise ValueError(f"无法从路径提取组目录: {file_path}")
    
    def _call_api(self, port: int, endpoint: str, data: dict, description: str = "API调用") -> dict:
        """调用API并返回响应，失败时直接抛出异常"""
        response = self.api_client.call_api(port=port, endpoint=endpoint, data=data)
        if not response.get('success'):
            raise RuntimeError(f"{description}失败: {response.get('error')}")
        return response
    
    def _batch_super_resolve(self, input_paths: List[str]) -> list:
        """通用批量超分辨率 - 调用 OSEDiff 批量API
        
        Args:
            input_paths: 待超分的图像路径列表
        Returns:
            API返回的逐图结果列表，每项含 success, output_path 等字段
        """
        group_root_dir = self._extract_group_dir(input_paths[0])
        response = self._call_api(
            self.API_PORTS['osediff'], 'super_resolution_batch',
            {'input_paths': input_paths, 'output_dir': group_root_dir, 'align_method': 'adain'},
            "批量超分"
        )
        return response.get('results', [])
    
    # ==================== 流程子方法（按调用顺序）====================
    
    # 步骤3: 质量评估
    def evaluate_quality(self, pairs_data: List[Dict]) -> List[Dict]:
        """评估图像对质量
        
        只传递必要字段：pair_image, interval, main_params, rand_params, yaw_interval
        """
        logger.info(f"开始质量评估 {len(pairs_data)} 个图像对")
        
        # 只提取必要字段，避免传输冗余数据
        quality_pairs_data = []
        for pair_data in pairs_data:
            quality_pairs_data.append({
                'pair_image': pair_data.get('super_resolved') or pair_data.get('pair_image'),
                'interval': pair_data.get('interval'),
                'main_params': pair_data.get('main_params'),
                'rand_params': pair_data.get('rand_params'),
                'yaw_interval': pair_data.get('yaw_interval')
            })
        
        response = self._call_api(
            self.API_PORTS['quality'], 'evaluate_pairs',
            {'pairs_data': quality_pairs_data},
            "质量评估"
        )
        results = response.get('results', [])
        logger.info(f"质量评估完成，评估了 {len(results)} 个图像对")
        return results
    
    # 步骤4: 质量过滤
    def filter_high_quality_results_by_group(self, quality_results: List[Dict], threshold: float = 0.7) -> List[Dict]:
        """按组过滤高质量结果（在同一yaw区间内取最低分）"""
        # 先按interval分组
        interval_dict = {}
        for result in quality_results:
            interval = result.get('interval', 0)
            if interval not in interval_dict:
                interval_dict[interval] = []
            interval_dict[interval].append(result)
        
        # 对每个interval内的pair取最低分
        filtered_results = []
        for interval, results in interval_dict.items():
            scores = [float(r['final_score']) for r in results 
                      if r.get('final_score') is not None]
            
            if scores:
                min_score = min(scores)
                
                if min_score >= threshold:
                    logger.info(f"yaw区间 {interval} 通过质量筛选 (最低分: {min_score:.3f})")
                    filtered_results.extend(results)
                else:
                    logger.info(f"yaw区间 {interval} 未通过质量筛选 (最低分: {min_score:.3f})")
            else:
                logger.warning(f"yaw区间 {interval} 没有有效的分数，跳过")
        
        logger.info(f"质量过滤完成，从 {len(quality_results)} 个结果中保留了 {len(filtered_results)} 个")
        return filtered_results
    
    # 步骤4.5: 描述生成
    def generate_panorama2_descriptions(self, filtered_results: List[Dict], panorama2_path: str, group_id: int) -> Dict:
        """为 panorama2 的右图生成文本描述（非关键步骤，失败不中断流程）"""
        logger.info(f"开始为 panorama2 的右图生成描述")
        
        panorama2_basename = os.path.splitext(os.path.basename(panorama2_path))[0]
        panorama2_right_images = []
        group_root_dir = None
        
        try:
            for result in filtered_results:
                pair_image_path = result.get('image_path') or result.get('pair_image')
                if not pair_image_path or panorama2_basename not in os.path.basename(pair_image_path):
                    continue
                
                # 提取右图
                pair_img = Image.open(pair_image_path)
                width, height = pair_img.size
                right_img = pair_img.crop((width // 2, 0, width, height))
                
                if not group_root_dir:
                    group_root_dir = self._extract_group_dir(pair_image_path)
                
                desc_dir = os.path.join(group_root_dir, 'descriptions')
                os.makedirs(desc_dir, exist_ok=True)
                
                base_name = os.path.splitext(os.path.basename(pair_image_path))[0]
                right_img_path = os.path.join(desc_dir, f"{base_name}_right.jpg")
                right_img.save(right_img_path, quality=95)
                
                panorama2_right_images.append({
                    'image_path': right_img_path,
                    'original_pair': pair_image_path,
                    'interval': result.get('interval'),
                    'yaw_interval': result.get('yaw_interval')
                })
            
            if not panorama2_right_images:
                return {'success': True, 'descriptions': []}
            
            # 批量生成描述
            image_paths = [img['image_path'] for img in panorama2_right_images]
            response = self._call_api(
                self.API_PORTS['quality'], 'generate_descriptions',
                {'image_paths': image_paths},
                "描述生成"
            )
            
            descriptions = response.get('results', [])
            for img_info, desc_result in zip(panorama2_right_images, descriptions):
                img_info['description'] = desc_result.get('description', '')
            
            # 保存描述到 JSON
            if group_root_dir:
                desc_json_path = os.path.join(group_root_dir, 'descriptions', 'panorama2_right_descriptions.json')
                with open(desc_json_path, 'w', encoding='utf-8') as f:
                    json.dump(panorama2_right_images, f, indent=2, ensure_ascii=False)
            
            logger.info(f"描述生成完成，共 {len(descriptions)} 条")
            return {'success': True, 'descriptions': panorama2_right_images}
            
        except Exception as e:
            logger.warning(f"描述生成失败（非关键）: {e}")
            return {'success': False, 'error': str(e)}
    
    # 步骤5: 切分 + 插值生成 + 插值超分
    def interpolate_and_super_resolve(self, filtered_results: List[Dict], group_id: int,
                                      panorama_path1: str, panorama_path2: str) -> tuple:
        """切分图像对、生成插值图像并进行超分处理
        
        Returns:
            (split_results, interpolated_results): 切分结果和超分后的插值结果
        """
        # 1. 切分图像对为左右两部分
        split_results = []
        for result in filtered_results:
            pair_image_path = result.get('pair_image', '')
            if not pair_image_path or not os.path.exists(pair_image_path):
                continue
            
            group_root_dir = self._extract_group_dir(pair_image_path)
            interpolated_dir = os.path.join(group_root_dir, "interpolated")
            os.makedirs(interpolated_dir, exist_ok=True)
            
            img = Image.open(pair_image_path)
            mid = img.width // 2
            left_img = img.crop((0, 0, mid, img.height))
            right_img = img.crop((mid, 0, img.width, img.height))
            
            base_name = os.path.splitext(os.path.basename(pair_image_path))[0]
            left_path = os.path.join(interpolated_dir, f"{base_name}_left.jpg")
            right_path = os.path.join(interpolated_dir, f"{base_name}_right.jpg")
            left_img.save(left_path)
            right_img.save(right_path)
            
            split_results.append({
                'pair_image': pair_image_path, 'left_image': left_path, 'right_image': right_path,
                'main_params': result.get('main_params'), 'rand_params': result.get('rand_params'),
                'interval': result.get('interval'), 'yaw_interval': result.get('yaw_interval'),
                'group_id': group_id
            })
        
        logger.info(f"切分完成，生成了 {len(split_results)} 个结果")
        
        # 2. 生成插值图像
        interp_resp = self._call_api(
            self.API_PORTS['preprocess'], 'generate_interpolated_images',
            {'split_results': split_results, 'panorama1_path': panorama_path1,
             'panorama2_path': panorama_path2, 'group_id': group_id,
             'num_interpolations': self.num_interpolations},
            "插值图像生成"
        )
        interpolated_results = interp_resp.get('results', [])
        
        # 3. 收集插值图像路径并批量超分
        input_paths = []
        interp_img_refs = []
        for interp_group in interpolated_results:
            for interp_img in interp_group.get('interpolated_images', []):
                input_path = interp_img.get('path')
                if input_path and os.path.exists(input_path):
                    input_paths.append(input_path)
                    interp_img_refs.append(interp_img)
        
        if input_paths:
            sr_results = self._batch_super_resolve(input_paths)
            
            # 移动到 interpolated_sr 目录并回填路径
            group_root_dir = self._extract_group_dir(input_paths[0])
            sr_output_dir = os.path.join(group_root_dir, "interpolated_sr")
            os.makedirs(sr_output_dir, exist_ok=True)
            
            sr_count = 0
            for interp_img, sr_result in zip(interp_img_refs, sr_results):
                if sr_result.get('success'):
                    output_path = sr_result['output_path']
                    target_path = os.path.join(sr_output_dir, os.path.basename(output_path))
                    shutil.move(output_path, target_path)
                    interp_img['super_resolved'] = target_path
                    sr_count += 1
            logger.info(f"插值图像超分处理完成，成功 {sr_count}/{len(input_paths)}")
        
        return split_results, interpolated_results
    
    # 步骤6: 创建最终数据
    def create_final_data_with_interpolation(self, split_results: List[Dict], interpolated_results: List[Dict], 
                                            group_id: int, panorama1_path: str, panorama2_path: str, 
                                            desc_result: Dict = None) -> List[Dict]:
        """创建包含插值图像的最终数据"""
        logger.info(f"创建包含插值的最终数据")
        
        # 构建描述字典，方便按 interval 查找
        descriptions_by_interval = {}
        if desc_result and desc_result.get('success') and desc_result.get('descriptions'):
            for desc in desc_result.get('descriptions', []):
                interval = desc.get('interval')
                if interval:
                    descriptions_by_interval[interval] = desc.get('description', '')
        
        final_data = []
        
        # 按interval组织数据
        interval_to_panoramas = {}  # {interval: {'panorama1': {...}, 'panorama2': {...}}}
        
        for split_result in split_results:
            interval = split_result.get('interval')
            if interval not in interval_to_panoramas:
                interval_to_panoramas[interval] = {}
            
            # 判断是panorama1还是panorama2
            pair_image = split_result.get('pair_image', '')
            pair_basename = os.path.splitext(os.path.basename(pair_image))[0]
            panorama1_basename = os.path.splitext(os.path.basename(panorama1_path))[0]
            panorama2_basename = os.path.splitext(os.path.basename(panorama2_path))[0]
            
            if pair_basename.startswith(panorama1_basename):
                interval_to_panoramas[interval]['panorama1'] = split_result
            elif pair_basename.startswith(panorama2_basename):
                interval_to_panoramas[interval]['panorama2'] = split_result
            else:
                # 如果无法判断，尝试从pair_image路径判断
                if panorama1_basename in pair_image:
                    interval_to_panoramas[interval]['panorama1'] = split_result
                else:
                    interval_to_panoramas[interval]['panorama2'] = split_result
        
        # 为每个interval创建最终数据条目
        for interval, panoramas in sorted(interval_to_panoramas.items()):
            if 'panorama1' not in panoramas or 'panorama2' not in panoramas:
                logger.warning(f"Interval {interval} 缺少完整的panorama数据，跳过")
                continue
            
            p1_split = panoramas['panorama1']
            p2_split = panoramas['panorama2']
            
            # 组织插值图像数据和参数
            p1_interp_data = []
            p2_interp_data = []
            p1_params_sequence = []
            p2_params_sequence = []
            
            for interp_group in interpolated_results:
                panorama = interp_group.get('panorama')
                interp_images = interp_group.get('interpolated_images', [])
                
                if panorama == 'panorama1' and interp_group.get('interval') == interval:
                    # 按照从左到右的顺序组织：A1 (left) -> interp_01 -> ... -> interp_09 -> A2 (right)
                    images = [p1_split.get('left_image')]  # 起始左图
                    params = [p1_split.get('main_params')]  # 起始左图参数
                    
                    for interp_img in sorted(interp_images, key=lambda x: x.get('weight_idx', 0)):
                        images.append(interp_img.get('super_resolved', interp_img.get('path')))
                        params.append(interp_img.get('params'))  # 添加插值参数
                    
                    images.append(p1_split.get('right_image'))  # 结束右图
                    params.append(p1_split.get('rand_params'))  # 结束右图参数
                    
                    p1_interp_data = images
                    p1_params_sequence = params
                    
                elif panorama == 'panorama2' and interp_group.get('interval') == interval:
                    images = [p2_split.get('left_image')]
                    params = [p2_split.get('main_params')]
                    
                    for interp_img in sorted(interp_images, key=lambda x: x.get('weight_idx', 0)):
                        images.append(interp_img.get('super_resolved', interp_img.get('path')))
                        params.append(interp_img.get('params'))
                    
                    images.append(p2_split.get('right_image'))
                    params.append(p2_split.get('rand_params'))
                    
                    p2_interp_data = images
                    p2_params_sequence = params
            
            # 创建最终数据条目
            final_entry = {
                'group_id': group_id,
                'yaw_interval': {
                    'interval_id': interval,
                    'yaw_min': p1_split.get('yaw_interval', (0, 0))[0],
                    'yaw_max': p1_split.get('yaw_interval', (0, 0))[1]
                },
                'panorama1': {
                    'original_path': panorama1_path,
                    'interpolated_sequence': p1_interp_data,  # 11张图片序列
                    'params_sequence': p1_params_sequence     # 11组参数序列
                },
                'panorama2': {
                    'original_path': panorama2_path,
                    'interpolated_sequence': p2_interp_data,  # 11张图片序列
                    'params_sequence': p2_params_sequence,    # 11组参数序列
                    'right_image_description': descriptions_by_interval.get(interval, '')  # 添加右图描述
                }
            }
            
            final_data.append(final_entry)
            logger.info(f"创建interval {interval} 的最终数据条目，包含 {len(p1_interp_data)} + {len(p2_interp_data)} 张图像")
        
        logger.info(f"最终数据创建完成，共 {len(final_data)} 个条目")
        return final_data
    
    # ==================== 主流程 ====================
    
    def process_image_group(self, panorama_path1: str, panorama_path2: str, group_id: int) -> Dict:
        """处理图片组（两张全景图）
        
        流程:
            1. 预处理 → 2. 超分 → 3. 质量评估 → 4. 质量过滤
            → 4.5. 描述生成 → 5. 切分+插值+插值超分 → 6. 创建最终数据
        """
        logger.info(f"开始处理图片组 #{group_id}: {os.path.basename(panorama_path1)} + {os.path.basename(panorama_path2)}")
        start_time = time.time()
        
        try:
            # 1. 预处理两张全景图
            path1 = os.path.abspath(panorama_path1) if not os.path.isabs(panorama_path1) else panorama_path1
            path2 = os.path.abspath(panorama_path2) if not os.path.isabs(panorama_path2) else panorama_path2
            
            resp1 = self._call_api(
                self.API_PORTS['preprocess'], 'preprocess_for_group',
                {'image_path': path1, 'group_id': group_id, 'is_first': True}, "预处理第一张"
            )
            resp2 = self._call_api(
                self.API_PORTS['preprocess'], 'preprocess_for_group',
                {'image_path': path2, 'group_id': group_id, 'is_first': False}, "预处理第二张"
            )
            all_preprocess_results = resp1.get('results', []) + resp2.get('results', [])
            
            # 2. 超分辨率处理 - 展开嵌套结构并批量超分
            flat_pairs = []
            for r in all_preprocess_results:
                if isinstance(r, list):
                    flat_pairs.extend(d for d in r if 'pair_image' in d)
                elif isinstance(r, dict) and 'pair_image' in r:
                    flat_pairs.append(r)
            
            valid_pairs = [d for d in flat_pairs if os.path.exists(d['pair_image'])]
            batch_sr = self._batch_super_resolve([d['pair_image'] for d in valid_pairs])
            
            sr_results = []
            for pair_data, sr_result in zip(valid_pairs, batch_sr):
                if sr_result.get('success'):
                    entry = pair_data.copy()
                    entry['super_resolved'] = sr_result['output_path']
                    entry['align_method'] = 'adain'
                    sr_results.append(entry)
            logger.info(f"超分辨率处理完成，成功 {len(sr_results)}/{len(valid_pairs)}")
            
            # 3. 质量评估
            quality_results = self.evaluate_quality(sr_results)
            
            # 4. 质量过滤（按组取最低分）
            filtered_results = self.filter_high_quality_results_by_group(quality_results, threshold=0.7)
            if not filtered_results:
                logger.warning(f"图片组 #{group_id} 没有高质量图像")
                return {
                    "group_id": group_id, "panorama1": panorama_path1, "panorama2": panorama_path2,
                    "error": "没有高质量图像", "processing_time": time.time() - start_time, "success": False
                }
            
            # 4.5. 为 panorama2 的右图生成描述（非关键步骤）
            desc_result = self.generate_panorama2_descriptions(filtered_results, panorama_path2, group_id)
            
            # 5. 切分、插值生成、插值超分（一步完成）
            split_results, interpolated_results = self.interpolate_and_super_resolve(
                filtered_results, group_id, panorama_path1, panorama_path2
            )
            
            # 6. 创建最终数据（按yaw区间组织）
            final_data = self.create_final_data_with_interpolation(
                split_results, interpolated_results,
                group_id, panorama_path1, panorama_path2, desc_result
            )
            
            processing_time = time.time() - start_time
            
            result = {
                "group_id": group_id,
                "panorama1": panorama_path1,
                "panorama2": panorama_path2,
                "final_data": final_data,
                "processing_time": processing_time,
                "success": True
            }
            
            logger.info(f"图片组 #{group_id} 处理完成 (耗时: {processing_time:.2f}秒)")
            return result
            
        except Exception as e:
            logger.error(f"处理图片组 #{group_id} 失败: {e}")
            return {
                "group_id": group_id,
                "panorama1": panorama_path1,
                "panorama2": panorama_path2,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "success": False
            }
    
    # ==================== 外部调用方法 ====================
    
    def save_single_group_result(self, result: Dict):
        """保存单个图片组的处理结果（实时保存）"""
        try:
            # 构建完整的输出文件路径
            output_file_path = os.path.join(self.output_root_dir, self.output_file)
            
            # 读取现有的results
            existing_results = []
            if os.path.exists(output_file_path):
                with open(output_file_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
            
            # 添加本次的结果（如果是成功的话）
            if result.get('success', False) and 'final_data' in result:
                new_entries = result.get('final_data', [])
                existing_results.extend(new_entries)
                
                # 保存更新的results.json
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_results, f, ensure_ascii=False, indent=2)
                
                logger.info(f"实时更新 {self.output_file}，当前共有 {len(existing_results)} 个数据条目")
            
            # 生成或更新 group_info.json（无论成功失败都生成）
            group_id = result.get('group_id')
            panorama1 = result.get('panorama1', '')
            panorama2 = result.get('panorama2', '')
            processing_time = result.get('processing_time', 0)
            success = result.get('success', False)
            
            # 确定group目录位置
            group_dir_name = f"group_{group_id:04d}"
            if 'preprocess' in self.output_root_dir:
                parent_dir = os.path.dirname(self.output_root_dir)
                group_dir = os.path.join(parent_dir, group_dir_name)
            else:
                group_dir = os.path.join(self.output_root_dir, group_dir_name)
            
            # 构建group_info
            if success and 'final_data' in result:
                final_data = result.get('final_data', [])
                yaw_intervals = []
                for entry in final_data:
                    interval_info = entry.get('yaw_interval', {})
                    yaw_intervals.append({
                        'interval_id': interval_info.get('interval_id'),
                        'yaw_min': interval_info.get('yaw_min'),
                        'yaw_max': interval_info.get('yaw_max')
                    })
                
                group_info = {
                    'group_id': group_id,
                    'panorama1': os.path.basename(panorama1),
                    'panorama2': os.path.basename(panorama2),
                    'panorama1_path': panorama1,
                    'panorama2_path': panorama2,
                    'num_quadruples': len(final_data),
                    'yaw_intervals': yaw_intervals,
                    'processing_time': processing_time,
                    'success': True
                }
            else:
                # 失败或没有高质量数据的情况
                error_msg = result.get('error', '无数据或处理失败')
                group_info = {
                    'group_id': group_id,
                    'panorama1': os.path.basename(panorama1) if panorama1 else '',
                    'panorama2': os.path.basename(panorama2) if panorama2 else '',
                    'panorama1_path': panorama1 if panorama1 else '',
                    'panorama2_path': panorama2 if panorama2 else '',
                    'num_quadruples': 0,
                    'yaw_intervals': [],
                    'processing_time': processing_time,
                    'success': False,
                    'error': error_msg
                }
            
            # 保存 group_info.json
            group_info_file = os.path.join(group_dir, 'group_info.json')
            with open(group_info_file, 'w', encoding='utf-8') as f:
                json.dump(group_info, f, ensure_ascii=False, indent=2)
            logger.info(f"组信息已实时保存: {group_info_file}")
            
        except Exception as e:
            logger.error(f"保存单个组结果失败: {e}")
    
    def count_current_intervals(self) -> int:
        """统计当前 results.json 中的 yaw_interval 数量"""
        output_file_path = os.path.join(self.output_root_dir, self.output_file)
        
        if not os.path.exists(output_file_path):
            return 0
        
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # results.json 是扁平化的 yaw_interval 列表
            # 每个条目就是一个 yaw_interval，直接返回列表长度
            if isinstance(results, list):
                return len(results)
            else:
                return 0
        except Exception as e:
            logger.warning(f"统计 yaw_interval 数量时出错: {e}")
            return 0
