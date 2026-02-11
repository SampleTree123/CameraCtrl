"""
全景图像处理主程序
"""

import os
import sys
import argparse
import logging
import glob
from typing import List
from core.panorama_processor import PanoramaProcessor
from utils.group_generator import ImageGroupGenerator

# 设置GPU环境变量，只使用6号和7号GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入路径配置
from config.api_config import BASE_DIR

def resolve_path(path):
    """将路径解析为绝对路径
    
    规则：
    - 如果已经是绝对路径，直接返回
    - 如果是相对路径，相对于 full_process_m3 目录解析
    """
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(current_dir, path))


# 设置日志
def setup_logging(output_dir):
    """设置日志"""
    # 转换为绝对路径
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'panorama_processing.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = logging.getLogger(__name__)

def find_panorama_images(input_dir: str, extensions: List[str] = None) -> List[str]:
    """查找全景图像文件"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    image_paths = []
    for ext in extensions:
        pattern = os.path.join(input_dir, f"*{ext}")
        image_paths.extend(glob.glob(pattern))
        pattern = os.path.join(input_dir, f"*{ext.upper()}")
        image_paths.extend(glob.glob(pattern))
    
    # 去重并排序
    image_paths = sorted(list(set(image_paths)))
    
    logger.info(f"在 {input_dir} 中找到 {len(image_paths)} 张图像")
    return image_paths

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='全景图像处理管道')
    parser.add_argument('--input_dirs', nargs='+', required=True, help='多个输入文件夹（用于混合配对）')
    parser.add_argument('--output_dir', type=str, default='output', help='输出根目录')
    
    # sample_n 和 target_n 互斥
    count_group = parser.add_mutually_exclusive_group(required=True)
    count_group.add_argument('--sample_n', type=int, help='固定采样数量（生成指定数量的图片对）')
    count_group.add_argument('--target_n', type=int, help='目标输出数量（持续处理直到达到目标 yaw_interval 数量）')
    
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    parser.add_argument('--api_version', type=str, choices=['original', 'shared_left'], 
                       default='original', help='API版本选择: original(端口5000-5002), shared_left(端口5010-5012)')
    parser.add_argument('--num_interpolations', type=int, default=9, 
                       help='插值图像数量（默认9），控制在两张图之间生成多少个插值帧')
    
    args = parser.parse_args()
    
    # 固定配置
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # 转换输出目录为绝对路径
    args.output_dir = resolve_path(args.output_dir)
    
    # 转换输入目录为绝对路径
    args.input_dirs = [resolve_path(d) for d in args.input_dirs]
    
    # 设置日志
    logger = setup_logging(args.output_dir)
    
    # 创建处理器（输出文件固定为 results.json）
    processor = PanoramaProcessor(
        output_root_dir=args.output_dir, 
        api_version=args.api_version,
        output_file='results.json',
        num_interpolations=args.num_interpolations
    )
    
    logger.info(f"插值配置: 每个图像对将生成 {args.num_interpolations} 个插值帧")
    
    # 收集所有图片路径
    all_image_paths = []
    logger.info(f"\n{'='*60}")
    logger.info(f"📂 输入文件夹统计")
    logger.info(f"{'='*60}")
    logger.info(f"文件夹数量: {len(args.input_dirs)}")
    logger.info(f"")

    # 遍历输入文件夹并收集图片
    for idx, input_dir in enumerate(args.input_dirs, 1):
        if not os.path.exists(input_dir):
            logger.warning(f"  [{idx}] ❌ {input_dir} (不存在，跳过)")
            continue
        
        dir_images = find_panorama_images(input_dir, extensions)
        all_image_paths.extend(dir_images)
        logger.info(f"  [{idx}] ✓ {input_dir}")
        logger.info(f"       图片数: {len(dir_images)} 张")
    
    logger.info(f"")
    logger.info(f"{'─'*60}")
    logger.info(f"📊 总图片数量: {len(all_image_paths)} 张")
    logger.info(f"{'='*60}\n")
    
    if not all_image_paths:
        logger.error("❌ 没有找到任何图片")
        return
    
    # 图片配对（使用 all_pairs 方法）
    logger.info(f"{'='*60}")
    logger.info(f"🔗 图片配对")
    logger.info(f"{'='*60}")
    logger.info(f"配对方法: all_pairs (所有图片两两配对)")
    logger.info(f"总图片数: {len(all_image_paths)} 张")
    
    all_groups = ImageGroupGenerator.generate_all_pairs(all_image_paths)
    logger.info(f"理论配对数: {len(all_image_paths)} × {len(all_image_paths)-1} ÷ 2 = {len(all_image_paths)*(len(all_image_paths)-1)//2}")
    
    if not all_groups:
        logger.error("❌ 没有生成图片组")
        return
    
    logger.info(f"")
    logger.info(f"✅ 可用图片组数: {len(all_groups)} 个")
    logger.info(f"{'='*60}\n")
    
    # 根据 sample_n 或 target_n 模式处理
    if args.sample_n:
        # 固定采样模式
        logger.info(f"{'='*60}")
        logger.info(f"⚙️  处理模式: 固定采样")
        logger.info(f"{'='*60}")
        logger.info(f"总图片数: {len(all_image_paths)} 张")
        logger.info(f"理论配对数: {len(all_groups)} 个（不重复）")
        logger.info(f"采样配对数: {args.sample_n} 个")
        logger.info(f"{'='*60}\n")
        
        process_with_sample_n(all_groups, args.sample_n, args.random_seed, processor)
    else:
        # 目标数量模式（target_n）
        logger.info(f"{'='*60}")
        logger.info(f"⚙️  处理模式: 目标数量（随机采样，允许重复）")
        logger.info(f"{'='*60}")
        logger.info(f"总图片数: {len(all_image_paths)} 张")
        logger.info(f"采样策略: 随机选择两个图片配对（允许重复配对）")
        logger.info(f"目标输出: {args.target_n} 个 yaw_interval")
        logger.info(f"{'='*60}\n")
        
        process_with_target_n(all_image_paths, args.target_n, args.random_seed, processor)
    
    logger.info(f"\n🎉 所有处理完成！")

def process_with_sample_n(all_groups, sample_n, random_seed, processor):
    """固定采样模式：采样指定数量的图片对并处理"""
    import random
    random.seed(random_seed)
    
    logger = logging.getLogger(__name__)
    
    # 采样
    if sample_n < len(all_groups):
        groups = random.sample(all_groups, sample_n)
        logger.info(f"从 {len(all_groups)} 个图片组中随机采样了 {sample_n} 个")
    else:
        groups = all_groups
        logger.info(f"使用全部 {len(all_groups)} 个图片组")
    
    # 处理每个图片组
    for group_id, (panorama1, panorama2) in enumerate(groups, 1):
        logger.info(f"处理图片组 {group_id}/{len(groups)}")
        result = processor.process_image_group(panorama1, panorama2, group_id)
        processor.save_single_group_result(result)
        logger.info(f"已实时保存图片组 {group_id} 的结果")

def process_with_target_n(image_list, target_n, random_seed, processor):
    """目标数量模式：随机采样直到达到目标 yaw_interval 数量（允许重复配对）"""
    import random
    random.seed(random_seed)
    
    logger = logging.getLogger(__name__)
    
    logger.info(f"开始处理，目标: {target_n} 个 yaw_interval")
    logger.info(f"采样池图片数: {len(image_list)} 张")
    logger.info(f"采样策略: 每次随机选择两个不同的图片进行配对\n")
    
    processed_count = 0
    current_intervals = 0
    processed_pairs = []  # 记录已处理的配对（用于统计）
    
    # 无限循环，直到达到目标
    while current_intervals < target_n:
        processed_count += 1
        
        # 随机选择两个不同的图片
        panorama1, panorama2 = random.sample(image_list, 2)
        
        # 记录配对（用于显示统计）
        pair_key = tuple(sorted([panorama1, panorama2]))
        is_repeat = pair_key in processed_pairs
        processed_pairs.append(pair_key)
        
        logger.info(f"\n{'─'*60}")
        logger.info(f"处理图片组 {processed_count}")
        logger.info(f"当前进度: {current_intervals}/{target_n} ({current_intervals*100//target_n if target_n > 0 else 0}%)")
        if is_repeat:
            logger.info(f"配对状态: 🔄 重复配对")
        else:
            logger.info(f"配对状态: ✨ 新配对")
        logger.info(f"{'─'*60}")
        
        # 处理当前图片组
        result = processor.process_image_group(panorama1, panorama2, processed_count)
        
        # 实时保存
        processor.save_single_group_result(result)
        
        # 统计当前数量
        current_intervals = processor.count_current_intervals()
        
        # 计算本组贡献的 interval 数
        group_intervals = 0
        if 'final_data' in result and result['final_data']:
            group_intervals = len(result['final_data'])
        
        logger.info(f"✓ 图片组 {processed_count} 完成，贡献了 {group_intervals} 个 yaw_interval")
        logger.info(f"  累计: {current_intervals}/{target_n}")
        
        # 检查是否已达到目标
        if current_intervals >= target_n:
            logger.info(f"\n✅ 已达到目标数量 {target_n}，停止处理")
            break
    
    # 统计重复率
    unique_pairs = len(set(processed_pairs))
    repeat_rate = (processed_count - unique_pairs) / processed_count * 100 if processed_count > 0 else 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"处理统计:")
    logger.info(f"  - 处理的图片组数: {processed_count}")
    logger.info(f"  - 唯一配对数: {unique_pairs}")
    logger.info(f"  - 重复配对数: {processed_count - unique_pairs}")
    logger.info(f"  - 重复率: {repeat_rate:.1f}%")
    logger.info(f"  - 最终 yaw_interval 数: {current_intervals}")
    logger.info(f"  - 目标数量: {target_n}")
    logger.info(f"  - 达成率: {current_intervals*100//target_n if target_n > 0 else 0}%")
    logger.info(f"{'='*60}\n")
    
    return processed_pairs

if __name__ == "__main__":
    main()