"""
API 客户端，供 panorama_processor 调用 preprocess / osediff / quality 服务。
"""

import logging
import requests
from typing import Dict

logger = logging.getLogger(__name__)


class APIClient:
    """API 客户端"""

    def __init__(self, base_url: str = "http://localhost"):
        self.base_url = base_url
        self.session = requests.Session()

    def call_api(self, port: int, endpoint: str, data: Dict = None, files: Dict = None) -> Dict:
        """调用 API 接口"""
        url = f"{self.base_url}:{port}/{endpoint}"
        try:
            timeout = 300
            if files:
                response = self.session.post(url, files=files, timeout=timeout)
            else:
                response = self.session.post(url, json=data, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API 调用失败 ({url}): {e}")
            return {"error": str(e)}