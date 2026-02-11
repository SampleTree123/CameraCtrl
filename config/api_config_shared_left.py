"""
API配置文件 - GPU 1 共享左图版本
端口: 5010-5012
"""

# API服务端口配置
API_PORTS = {
    'preprocess': 5010,
    'osediff': 5011,
    'quality': 5012
}

# API服务基础URL
API_BASE_URL = "http://localhost"
