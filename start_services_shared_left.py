"""
启动所有API服务的脚本（共享左图版本）
"""

import os
import sys
import subprocess
import time
import logging
import argparse
import requests
from typing import Dict, List

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 服务配置（共享左图版本：端口5010-5012，GPU 1）
SERVICES = {
    'preprocess': {
        'script': 'api_services/preprocess.py',  # 改为使用原始版本
        'port': 5010,
        'env': 'xks_precisecam',
        'args': ['--port', '5010']
    },
    'osediff': {
        'script': 'api_services/osediff_api.py',
        'port': 5011,
        'env': 'xks_OSEDiff',
        'env_vars': {'CUDA_VISIBLE_DEVICES': '1', 'HF_ENDPOINT': 'https://hf-mirror.com'},  # 使用GPU 1
        'args': [
            '--port', '5011',
            '--osediff_path', '/root/siton-tmp/sx/xks/51_code/data_prepare/OSEDiff/preset/models/osediff.pkl',
            '--pretrained_model_name_or_path', '/root/siton-tmp/sx/xks/models/AI-ModelScope/stable-diffusion-2-1-base',
            '--ram_path', '/root/siton-tmp/sx/xks/51_code/data_prepare/OSEDiff/preset/models/ram_swin_large_14m.pth',
            '--ram_ft_path', '/root/siton-tmp/sx/xks/51_code/data_prepare/OSEDiff/preset/models/DAPE.pth',
            '--device', 'cuda',
            '--upscale', '2',
            '--process_size', '512',
            '--mixed_precision', 'fp16'
        ]
    },
    'quality': {
        'script': 'api_services/quality_api.py',
        'port': 5012,
        'env': 'xks_qwen',
        'env_vars': {'CUDA_VISIBLE_DEVICES': '1'},  # 使用GPU 1
        'args': [
            '--port', '5012',
            '--model_dir', '/root/siton-tmp/sx/xks/models/Qwen2.5-VL-7B-Instruct',
            '--prompt_path', '/root/siton-tmp/sx/xks/51_code/data_prepare/qwen_filter/image_pair_quality_prompt.txt'
        ]
    }
}

class ServiceManager:
    """服务管理器"""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.processes: Dict[str, subprocess.Popen] = {}
        self.log_files: Dict[str, str] = {}
    
    def start_service(self, service_name: str, service_config: Dict, output_dir: str) -> bool:
        """启动单个服务"""
        try:
            script_path = os.path.join(self.base_dir, service_config['script'])
            port = service_config['port']
            env = service_config['env']
            args = service_config['args'].copy()  # 复制一份，避免修改原始配置
            
            # 添加输出目录参数
            args.extend(['--output_dir', output_dir])
            
            # 获取环境变量
            env_vars = service_config.get('env_vars', {})
            
            # 创建日志文件（添加shared_left后缀）
            log_file = os.path.join(self.base_dir, f"{service_name}_shared_left_log.txt")
            self.log_files[service_name] = log_file
            
            # 构建命令
            if env == 'base':
                cmd = ['python', script_path] + args
            else:
                cmd = ['conda', 'run', '-n', env, 'python', script_path] + args
            
            logger.info(f"启动 {service_name} 服务 (端口: {port})")
            logger.info(f"命令: {' '.join(cmd)}")
            if env_vars:
                logger.info(f"环境变量: {env_vars}")
            
            # 启动进程
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=self.base_dir,
                    env={**os.environ, **env_vars}  # 合并环境变量
                )
            
            self.processes[service_name] = process
            logger.info(f"✅ {service_name} 服务启动成功 (PID: {process.pid})")
            
            # 等待服务启动
            time.sleep(5)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 启动 {service_name} 服务失败: {e}")
            return False
    
    def start_all_services(self, specific_service: str = None, output_dir: str = "output") -> Dict[str, bool]:
        """启动所有服务或指定服务"""
        results = {}
        
        services_to_start = {specific_service: SERVICES[specific_service]} if specific_service else SERVICES
        
        for service_name, service_config in services_to_start.items():
            success = self.start_service(service_name, service_config, output_dir)
            results[service_name] = success
            
            if not success:
                logger.error(f"❌ {service_name} 服务启动失败，停止后续服务")
                break
        
        return results
    
    def stop_service(self, service_name: str):
        """停止单个服务"""
        # 先尝试从内存中的进程停止
        if service_name in self.processes:
            process = self.processes[service_name]
            try:
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"✅ {service_name} 服务已停止")
                return
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning(f"⚠️ {service_name} 服务被强制终止")
                return
            except Exception as e:
                logger.error(f"❌ 停止 {service_name} 服务失败: {e}")
        
        # 如果内存中没有进程信息，通过端口查找并停止
        if service_name in SERVICES:
            port = SERVICES[service_name]['port']
            self._stop_by_port(service_name, port)
    
    def _stop_by_port(self, service_name: str, port: int):
        """通过端口号停止服务"""
        try:
            pids = []
            
            # 方法1: 尝试使用 psutil（如果可用）
            try:
                import psutil
                for conn in psutil.net_connections(kind='tcp'):
                    if conn.laddr.port == port and conn.pid:
                        pids.append(conn.pid)
            except ImportError:
                pass
            except Exception:
                pass
            
            # 方法2: 如果 psutil 不可用，尝试使用 /proc/net/tcp
            if not pids:
                try:
                    with open('/proc/net/tcp', 'r') as f:
                        lines = f.readlines()[1:]  # 跳过标题行
                        for line in lines:
                            parts = line.split()
                            if len(parts) >= 2:
                                local_addr = parts[1]
                                # 格式: 0100007F:1388 (hex格式的IP:端口)
                                if ':' in local_addr:
                                    hex_port = local_addr.split(':')[1]
                                    if int(hex_port, 16) == port:
                                        # 查找对应的进程
                                        inode = parts[9]
                                        # 通过 inode 查找 PID
                                        for pid_dir in os.listdir('/proc'):
                                            if pid_dir.isdigit():
                                                try:
                                                    for fd in os.listdir(f'/proc/{pid_dir}/fd'):
                                                        fd_path = f'/proc/{pid_dir}/fd/{fd}'
                                                        if os.path.islink(fd_path):
                                                            link = os.readlink(fd_path)
                                                            if f'socket:[{inode}]' in link:
                                                                pids.append(int(pid_dir))
                                                                break
                                                except (PermissionError, FileNotFoundError):
                                                    continue
                except (FileNotFoundError, PermissionError):
                    pass
            
            # 方法3: 尝试使用 netstat（如果可用）
            if not pids:
                try:
                    result = subprocess.run(
                        ['netstat', '-tlnp'], 
                        capture_output=True, text=True, timeout=5
                    )
                    for line in result.stdout.split('\n'):
                        if f':{port}' in line and 'LISTEN' in line:
                            # 尝试提取 PID
                            parts = line.split()
                            for part in parts:
                                if '/' in part and part.split('/')[0].isdigit():
                                    pids.append(int(part.split('/')[0]))
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass
            
            # 去重
            pids = list(set(pids))
            
            if pids:
                for pid in pids:
                    try:
                        os.kill(int(pid), 9)
                        logger.info(f"✅ {service_name} 服务已停止 (PID: {pid}, 端口: {port})")
                    except ProcessLookupError:
                        logger.warning(f"进程 {pid} 已不存在")
                    except Exception as e:
                        logger.error(f"停止进程 {pid} 失败: {e}")
            else:
                # 如果找不到 PID，至少尝试通过健康检查确认服务是否真的在运行
                if self.check_service_health(service_name, port):
                    logger.warning(f"⚠️ 无法找到 {service_name} 服务的 PID，但服务似乎仍在运行")
                    logger.warning(f"   请手动检查端口 {port} 的进程: ps aux | grep {port}")
                else:
                    logger.info(f"ℹ️ {service_name} 服务未运行 (端口: {port})")
        except Exception as e:
            logger.error(f"❌ 通过端口停止 {service_name} 服务失败: {e}")
    
    def stop_all_services(self):
        """停止所有服务"""
        for service_name in SERVICES:
            self.stop_service(service_name)
    
    def check_service_health(self, service_name: str, port: int) -> bool:
        """检查服务健康状态（单次检查）"""
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def wait_for_service_healthy(self, service_name: str, port: int, max_wait: int = 30, interval: int = 5) -> bool:
        """等待服务健康，带重试机制
        
        Args:
            service_name: 服务名称
            port: 服务端口
            max_wait: 最大等待时间（秒）
            interval: 检查间隔（秒）
        
        Returns:
            服务是否健康
        """
        waited = 0
        while waited < max_wait:
            if self.check_service_health(service_name, port):
                return True
            logger.info(f"⏳ 等待 {service_name} 服务就绪... ({waited}/{max_wait}秒)")
            time.sleep(interval)
            waited += interval
        return self.check_service_health(service_name, port)
    
    def check_all_services_health(self, with_wait: bool = False) -> Dict[str, bool]:
        """检查所有服务健康状态
        
        Args:
            with_wait: 是否等待服务就绪（启动后使用）
        """
        health_status = {}
        
        # 不同服务的最大等待时间配置
        max_wait_times = {
            'preprocess': 30,
            'osediff': 120,  # osediff 需要加载大模型，等待更长
            'quality': 60
        }
        
        for service_name, service_config in SERVICES.items():
            port = service_config['port']
            
            if with_wait:
                max_wait = max_wait_times.get(service_name, 30)
                is_healthy = self.wait_for_service_healthy(service_name, port, max_wait=max_wait)
            else:
                is_healthy = self.check_service_health(service_name, port)
            
            health_status[service_name] = is_healthy
            
            status = "✅ 健康" if is_healthy else "❌ 异常"
            logger.info(f"{service_name} 服务 ({port}): {status}")
        
        return health_status
    
    def show_logs(self, service_name: str, lines: int = 20):
        """显示服务日志"""
        if service_name in self.log_files:
            log_file = self.log_files[service_name]
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        log_lines = f.readlines()
                        recent_lines = log_lines[-lines:] if len(log_lines) > lines else log_lines
                        print(f"\n=== {service_name} 服务日志 (最近 {len(recent_lines)} 行) ===")
                        for line in recent_lines:
                            print(line.rstrip())
                except Exception as e:
                    logger.error(f"读取日志失败: {e}")
            else:
                logger.warning(f"日志文件不存在: {log_file}")
        else:
            logger.error(f"未知服务: {service_name}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='API服务管理器（共享左图版本）')
    parser.add_argument('--action', choices=['start', 'stop', 'restart', 'status', 'logs'], 
                       default='start', help='操作类型')
    parser.add_argument('--service', type=str, help='指定服务名称')
    parser.add_argument('--output_dir', type=str, default='output', help='输出根目录')
    parser.add_argument('--log_lines', type=int, default=20, help='显示日志行数')
    args = parser.parse_args()
    
    # 获取当前目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    manager = ServiceManager(base_dir)
    
    if args.action == 'start':
        if args.service:
            logger.info(f"启动 {args.service} API服务...")
            results = manager.start_all_services(args.service, args.output_dir)
            
            # 等待服务健康（带重试机制）
            if args.service in SERVICES:
                port = SERVICES[args.service]['port']
                max_wait_times = {'preprocess': 30, 'osediff': 120, 'quality': 60}
                max_wait = max_wait_times.get(args.service, 30)
                
                logger.info(f"等待 {args.service} 服务就绪（最长 {max_wait} 秒）...")
                is_healthy = manager.wait_for_service_healthy(args.service, port, max_wait=max_wait)
                
                if is_healthy:
                    logger.info(f"✅ {args.service} API服务启动成功并健康")
                else:
                    logger.error(f"❌ {args.service} API服务异常，请查看日志: {args.service}_shared_left_log.txt")
            else:
                logger.error(f"❌ 未知服务: {args.service}")
        else:
            logger.info("启动所有API服务...")
            results = manager.start_all_services(output_dir=args.output_dir)
            
            # 等待所有服务健康（带重试机制）
            logger.info("等待所有服务就绪...")
            health_status = manager.check_all_services_health(with_wait=True)
            
            all_healthy = all(health_status.values())
            if all_healthy:
                logger.info("✅ 所有API服务启动成功并健康")
            else:
                logger.error("❌ 部分API服务异常")
                for service, is_healthy in health_status.items():
                    if not is_healthy:
                        logger.error(f"  - {service} 服务异常，请查看日志: {service}_shared_left_log.txt")
    
    elif args.action == 'stop':
        if args.service:
            logger.info(f"停止 {args.service} 服务...")
            manager.stop_service(args.service)
        else:
            logger.info("停止所有API服务...")
            manager.stop_all_services()
    
    elif args.action == 'restart':
        logger.info("重启所有API服务...")
        manager.stop_all_services()
        time.sleep(5)
        results = manager.start_all_services(output_dir=args.output_dir)
        
        # 等待所有服务健康
        logger.info("等待所有服务就绪...")
        health_status = manager.check_all_services_health(with_wait=True)
        
        all_healthy = all(health_status.values())
        if all_healthy:
            logger.info("✅ 所有API服务重启成功并健康")
        else:
            logger.error("❌ 部分API服务异常")
            for service, is_healthy in health_status.items():
                if not is_healthy:
                    logger.error(f"  - {service} 服务异常，请查看日志: {service}_shared_left_log.txt")
    
    elif args.action == 'status':
        logger.info("检查API服务状态...")
        health_status = manager.check_all_services_health()
        
        all_healthy = all(health_status.values())
        if all_healthy:
            logger.info("✅ 所有API服务都健康")
        else:
            logger.error("❌ 部分API服务异常")
    
    elif args.action == 'logs':
        if args.service:
            manager.show_logs(args.service, args.log_lines)
        else:
            logger.error("使用 --logs 时必须指定 --service")

if __name__ == '__main__':
    main()
