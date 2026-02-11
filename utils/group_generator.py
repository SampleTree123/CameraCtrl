"""
图片组生成器，供 main.py 生成全景图配对。
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class ImageGroupGenerator:
    """图片组生成器"""

    @staticmethod
    def generate_all_pairs(image_list: list) -> List[Tuple[str, str]]:
        """从图片路径列表生成所有两两配对（不包括自己与自己配对）。
        
        Args:
            image_list: 图片路径列表
            
        Returns:
            所有可能的图片配对列表 [(img1, img2), ...]
        """
        if len(image_list) < 2:
            logger.warning("图片列表少于2张，无法生成图片组")
            return []

        groups = []
        for i in range(len(image_list)):
            for j in range(i + 1, len(image_list)):
                groups.append((image_list[i], image_list[j]))

        logger.info(f"共生成 {len(groups)} 个图片组（不含自身配对）")
        return groups
