#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据 pairs/params 目录中的参数文件，为 results.json 中缺失的参数路径字段补全：
- main_params_file -> 对应 *_left_params.json
- rand_params_file -> 对应 *_right_params.json

使用方式：
python utils/fill_params_paths.py \
  --results /home/chengyou/xks/data_prepare/full_process/360_output_2/results.json \
  --params_dir /home/chengyou/xks/data_prepare/full_process/360_output_2/pairs/params

脚本会创建 results.json.bak 备份，并原地覆盖 results.json。
"""

import os
import json
import argparse
from typing import Dict, Tuple


def infer_interval_name(entry: Dict) -> str:
    """从条目中推断 interval_name（不含 _pair/_left/_right 后缀）。

    规则：
    - 优先使用 left_image: 形如 {interval_name}_left.jpg -> 提取 interval_name
    - 否则使用 right_image: 形如 {interval_name}_right.jpg -> 提取 interval_name
    - 否则尝试 original_pair: 形如 {interval_name}_pair.jpg -> 提取 interval_name
    """
    left_path = entry.get('left_image', '')
    if left_path:
        fname = os.path.basename(left_path)
        if fname.endswith('_left.jpg'):
            return fname[:-len('_left.jpg')]
        stem, _ = os.path.splitext(fname)
        return stem.rstrip('_left')

    right_path = entry.get('right_image', '')
    if right_path:
        fname = os.path.basename(right_path)
        if fname.endswith('_right.jpg'):
            return fname[:-len('_right.jpg')]
        stem, _ = os.path.splitext(fname)
        return stem.rstrip('_right')

    original_pair = entry.get('original_pair', '') or entry.get('pair_image', '')
    if original_pair:
        fname = os.path.basename(original_pair)
        if fname.endswith('_pair.jpg'):
            return fname[:-len('_pair.jpg')]
        stem, _ = os.path.splitext(fname)
        return stem.rstrip('_pair')

    return ''


def expected_param_files(params_dir: str, interval_name: str) -> Tuple[str, str]:
    """构造期望的参数文件完整路径（基于 interval_name）。
    - left:  {interval_name}_left_params.json
    - right: {interval_name}_right_params.json
    """
    # params 文件以 interval_name（不含 _pair）为前缀
    stem = interval_name[:-5] if interval_name.endswith('_pair') else interval_name
    left = os.path.join(params_dir, f"{stem}_left_params.json")
    right = os.path.join(params_dir, f"{stem}_right_params.json")
    return left, right


def main():
    parser = argparse.ArgumentParser(description='为results.json补全参数文件路径')
    parser.add_argument('--results', required=True, help='results.json 路径')
    parser.add_argument('--params_dir', required=True, help='参数文件目录（包含 *_left_params.json 与 *_right_params.json）')
    args = parser.parse_args()

    results_path = os.path.abspath(args.results)
    params_dir = os.path.abspath(args.params_dir)

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"results.json 不存在: {results_path}")
    if not os.path.isdir(params_dir):
        raise NotADirectoryError(f"参数目录不存在: {params_dir}")

    # 读取结果
    with open(results_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"读取 JSON 失败: {results_path} - {e}")

    if not isinstance(data, list):
        raise ValueError('results.json 结构应为由数据条目组成的列表')

    updated = 0
    skipped_not_found = 0
    already_ok = 0

    for entry in data:
        # 检查是否已存在
        has_left = bool(entry.get('main_params_file'))
        has_right = bool(entry.get('rand_params_file'))
        if has_left and has_right:
            already_ok += 1
            continue

        interval_name = infer_interval_name(entry)
        if not interval_name:
            skipped_not_found += 1
            continue

        left_path, right_path = expected_param_files(params_dir, interval_name)

        left_exists = os.path.exists(left_path)
        right_exists = os.path.exists(right_path)

        if left_exists:
            entry['main_params_file'] = left_path
        if right_exists:
            entry['rand_params_file'] = right_path

        if left_exists or right_exists:
            updated += 1
        else:
            skipped_not_found += 1

    # 备份并写回
    backup_path = results_path + '.bak'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w', encoding='utf-8') as bf:
            json.dump(data, bf, ensure_ascii=False, indent=2)

    # 写回最新数据
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 补全完成: 已更新 {updated} 条, 已完整 {already_ok} 条, 未找到/跳过 {skipped_not_found} 条")
    print(f"results: {results_path}")
    print(f"params_dir: {params_dir}")


if __name__ == '__main__':
    main()


