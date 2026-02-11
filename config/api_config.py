"""
API配置文件 - GPU 0 服务配置
端口: 5000-5002
"""

# API服务端口配置
API_PORTS = {
    'preprocess': 5000,
    'osediff': 5001,
    'quality': 5002
}

# API服务基础URL
API_BASE_URL = "http://localhost" 