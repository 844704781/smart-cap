import datetime
import logging
import os
import shelve
import time
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional, List

import yaml
from moviepy.editor import VideoFileClip
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from proglog import TqdmProgressBarLogger, ProgressBarLogger
import re
import threading

from srt.ksasr import KuaiShouASR

# 视频文件扩展名
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']

# 全局变量
SOURCE_DIR = ""
TARGET_DIR = ""
DB_PATH = ""

# 统计信息全局变量
total_videos_found = 0
total_videos_processed = 0
total_videos_skipped = 0
new_videos_detected = 0
pending_videos = 0

logger = logging.getLogger(__name__)


def load_config():
    """从配置文件加载配置，支持不同环境(dev/prod)"""
    global SOURCE_DIR, TARGET_DIR, DB_PATH

    try:
        # 获取应用根目录
        app_dir = os.path.dirname(os.path.abspath(__file__))

        # 首先读取主配置文件确定环境
        main_config_path = os.path.join(app_dir, "application.yaml")
        if not os.path.exists(main_config_path):
            raise FileNotFoundError(f"未找到主配置文件: {main_config_path}")

        with open(main_config_path, 'r', encoding='utf-8') as f:
            main_config = yaml.safe_load(f)

        # 确定当前环境
        profile = main_config.get('service', {}).get('profile', 'dev')
        logger.info(f"当前运行环境: {profile}")

        # 根据环境加载对应的配置文件
        env_config_path = os.path.join(app_dir, f"application-{profile}.yaml")
        if not os.path.exists(env_config_path):
            raise FileNotFoundError(f"未找到环境配置文件: {env_config_path}")

        with open(env_config_path, 'r', encoding='utf-8') as f:
            env_config = yaml.safe_load(f)
        
        # 从环境配置中获取目录设置
        SOURCE_DIR = env_config.get('paths', {}).get('source_dir')
        TARGET_DIR = env_config.get('paths', {}).get('target_dir')

        if not SOURCE_DIR or not TARGET_DIR:
            raise ValueError("配置文件中缺少必要的路径配置(source_dir 或 target_dir)")

        # 创建.cache目录并将数据库文件存储在其中
        cache_dir = os.path.join(os.path.dirname(TARGET_DIR), ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        DB_PATH = os.path.join(cache_dir, ".video_srt_index")

        logger.info(f"配置加载成功: SOURCE_DIR={SOURCE_DIR}, TARGET_DIR={TARGET_DIR}")
        logger.info(f"数据库文件位置: {DB_PATH}")
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise


class MoviePyProgressLogger(ProgressBarLogger):
    """
    自定义MoviePy进度日志记录器
    每次更新打印新的一行
    """

    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        # 初始化进度计数器
        self.chunk_total = 1
        self.chunk_index = 0
        self.t_total = 1
        self.t_index = 0
        # 仅用于内部跟踪
        self._last_percent = -1
        self.operation_name = "音频提取"
        
    def set_operation_name(self, name):
        """设置当前操作名称"""
        self.operation_name = name

    def bars_callback(self, bar, attr, value, old_value=None):
        """进度条回调函数，跟踪chunk和t两种进度条"""
        # 更新对应的计数器
        if bar == 'chunk':
            if attr == 'total':
                self.chunk_total = value
            elif attr == 'index':
                self.chunk_index = value
        elif bar == 't':
            if attr == 'total':
                self.t_total = value
            elif attr == 'index':
                self.t_index = value
                
        # 计算总体进度百分比
        chunk_progress = self.chunk_index / self.chunk_total if self.chunk_total > 0 else 0
        t_progress = self.t_index / self.t_total if self.t_total > 0 else 0
        
        if self.t_total > 1 and self.chunk_total > 1:
            progress = (chunk_progress + t_progress) / 2
        elif self.chunk_total > 1:
            progress = chunk_progress
        else:
            progress = t_progress
            
        # 转换为百分比
        percent = int(progress * 100)
        
        # 每改变1%就更新一次显示
        if percent != self._last_percent:
            self._last_percent = percent
            if self.logger:
                self.logger.info(f"MoviePy {self.operation_name}进度: {percent}%")
    
    def callback(self, **changes):
        """处理其他类型的回调消息"""
        # 调用父类方法
        super().callback(**changes)
        
        # 记录重要消息
        if 'message' in changes and self.logger:
            message = changes['message']
            if isinstance(message, str):
                if 'Writing audio' in message:
                    self.logger.info(f"MoviePy: {message}")
                    # 当开始写入音频时更新操作名称
                    self.set_operation_name("音频写入")
                elif 'video decoded' in message or 'audio decoded' in message:
                    self.logger.info(f"MoviePy: {message}")
                    self.set_operation_name("解码")
                elif 'error' in message.lower() or 'exception' in message.lower():
                    self.logger.error(f"MoviePy错误: {message}")
                else:
                    self.logger.debug(f"MoviePy: {message}")


def extract_audio(video_path: str, audio_path: str) -> bool:
    """从视频文件中提取音频，使用改进的进度记录器"""
    try:
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        
        logger.info(f"开始从视频提取音频: {video_path} -> {audio_path}")
        
        # 创建并配置进度记录器
        progress_logger = MoviePyProgressLogger(logger)
        progress_logger.set_operation_name("音频提取")
        
        # 使用进度记录器处理视频
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(
            audio_path,
            logger=progress_logger,
            verbose=False  # 关闭MoviePy的内置进度输出
        )
        video.close()
        
        logger.info(f"音频提取完成: {audio_path}")
        return True
    except Exception as e:
        logger.error(f"从 {video_path} 提取音频时出错: {e}", exc_info=True)
        return False


def generate_srt(audio_path: str, srt_path: str) -> bool:
    """从音频生成SRT字幕文件。"""
    try:
        asr = KuaiShouASR(audio_path)
        asr_result = asr.run()
        asr_data = asr_result.segments
        with open(srt_path, "w", encoding="utf-8") as f:
            for index, asr in enumerate(asr_data):
                f.write(asr.text + "\n\n")
        return True
    except Exception as e:
        logger.error(f"为 {audio_path} 生成SRT时出错: {e}")
        return False


def is_processed(video_path: str) -> bool:
    """检查视频是否已经处理过。"""
    with shelve.open(DB_PATH) as db:
        return video_path in db


def get_all_processed_videos() -> List[str]:
    """获取所有已处理的视频列表"""
    with shelve.open(DB_PATH) as db:
        return list(db.keys())


def update_db(video_path: str, srt_path: Optional[str] = None) -> None:
    """更新数据库的处理信息。"""
    with shelve.open(DB_PATH) as db:
        db[video_path] = {
            "video_path": video_path,
            "srt_path": srt_path
        }


def is_file_ready(file_path: str, timeout: int = 60, check_interval: float = 1.0) -> bool:
    """
    检查文件是否已经完全写入（上传完成）
    
    通过监控文件大小在一段时间内是否保持稳定来判断文件是否上传完成
    
    参数:
        file_path: 文件路径
        timeout: 等待超时时间（秒）
        check_interval: 检查间隔（秒）
        
    返回:
        bool: 如果文件准备好可以处理，返回True；否则返回False
    """
        # 初始化文件大小监控
    previous_size = -1
    stable_count = 0
    max_stable_count = 5  # 文件大小连续稳定的次数阈值
    logger.info(f"检查文件是否正在上传: {file_path}, 检查时间大约 {max_stable_count} s")
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在或已被移除: {file_path}")
            return False
            
        # 获取当前文件大小
        try:
            current_size = os.path.getsize(file_path)
        except OSError as e:
            logger.warning(f"无法获取文件大小: {file_path}, 错误: {e}")
            time.sleep(check_interval)
            continue
        
        # 检查文件大小是否稳定
        if current_size == previous_size and current_size > 0:
            stable_count += 1
            if stable_count >= max_stable_count:
                logger.info(f"文件上传完成，大小稳定在 {current_size} 字节: {file_path}")
                return True
        else:
            stable_count = 0
            
        previous_size = current_size
        
        # 打印进度
        elapsed = time.time() - start_time
        logger.debug(f"正在等待文件上传: {file_path}, 当前大小: {current_size} 字节, 已等待: {elapsed:.1f}秒")
        
        time.sleep(check_interval)
    
    logger.warning(f"等待文件上传超时: {file_path}")
    return False


def get_target_paths(video_path: str) -> Dict[str, str]:
    """
    根据视频文件路径生成对应的音频和SRT字幕文件路径
    
    参数:
        video_path: 视频文件的完整路径
        
    返回:
        包含音频和SRT文件路径的字典
    """
    # 获取视频文件相对于SOURCE_DIR的路径
    rel_path = os.path.relpath(video_path, SOURCE_DIR)
    
    # 获取文件名（不含扩展名）
    file_dir, file_name = os.path.split(rel_path)
    file_base, _ = os.path.splitext(file_name)
    
    # 构建目标目录
    target_subdir = os.path.join(TARGET_DIR, file_dir)
    
    # 确保目标目录存在
    os.makedirs(target_subdir, exist_ok=True)
    
    # 构建音频和SRT文件路径
    audio_path = os.path.join(target_subdir, f"{file_base}.mp3")
    srt_path = os.path.join(target_subdir, f"{file_base}.srt")
    
    return {
        "audio_path": audio_path,
        "srt_path": srt_path
    }


def process_video(video_path: str) -> None:
    """处理视频以生成SRT字幕文件。"""
    # 如果已处理过则跳过
    if is_processed(video_path):
        logger.info(f"跳过已处理的视频: {video_path}")
        return
    
    logger.info(f"准备处理视频: {video_path}")
    
    # 确保文件已经完全上传
    if not is_file_ready(video_path):
        logger.error(f"文件未准备好，跳过处理: {video_path}")
        return
    
    # 生成目标路径
    target_paths = get_target_paths(video_path)
    audio_path = target_paths["audio_path"]
    srt_path = target_paths["srt_path"]
    
    # 检查SRT文件是否已存在
    if os.path.exists(srt_path):
        logger.info(f"SRT文件已存在: {srt_path}")
        update_db(video_path, srt_path)
        return
    
    # 提取音频
    if not extract_audio(video_path, audio_path):
        update_db(video_path, None)
        return
    
    # 生成SRT
    if generate_srt(audio_path, srt_path):
        update_db(video_path, srt_path)
        logger.info(f"成功生成SRT: {srt_path}")
        
        # 删除音频文件以节省空间
        try:
            os.remove(audio_path)
            logger.info(f"已删除临时音频文件: {audio_path}")
        except Exception as e:
            logger.warning(f"删除音频文件 {audio_path} 时出错: {e}")
    else:
        update_db(video_path, None)
        logger.error(f"为 {video_path} 生成SRT失败")


def scan_directory(directory: str) -> None:
    """递归扫描目录中的视频文件并处理它们。"""
    global total_videos_found, pending_videos

    logger.info(f"开始扫描目录: {directory}")

    # 收集所有视频文件
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file_path)
            if ext.lower() in VIDEO_EXTENSIONS:
                video_files.append(file_path)

    total_videos_found = len(video_files)
    logger.info(f"发现视频文件总数: {total_videos_found}")

    # 获取已处理视频列表
    processed_videos = get_all_processed_videos()
    pending_videos = sum(1 for v in video_files if v not in processed_videos)
    logger.info(f"待处理视频数量: {pending_videos}")

    # 处理所有视频文件
    for index, file_path in enumerate(video_files, 1):
        logger.info(f"处理第 {index}/{total_videos_found} 个视频文件: {os.path.basename(file_path)}")
        process_video(file_path)

    logger.info(
        f"目录扫描完成: 共发现 {total_videos_found} 个视频, 处理 {total_videos_processed} 个, 跳过 {total_videos_skipped} 个")


class VideoHandler(FileSystemEventHandler):
    """监视新视频文件并处理它们。"""

    def __init__(self):
        super().__init__()
        self.new_videos = []

    def on_created(self, event):
        global new_videos_detected, pending_videos

        if not event.is_directory:
            file_path = event.src_path
            _, ext = os.path.splitext(file_path)
            if ext.lower() in VIDEO_EXTENSIONS:
                new_videos_detected += 1
                pending_videos += 1
                self.new_videos.append(file_path)
                logger.info("-" * 70)
                logger.info(f"检测到新视频 [{new_videos_detected}]: {file_path}")
                logger.info(f"当前累计未处理视频数: {pending_videos}")
                process_video(file_path)
                pending_videos -= 1

    def on_modified(self, event):
        # 某些系统会同时触发created和modified事件
        # 仅当数据库中不存在时才处理
        if not event.is_directory:
            file_path = event.src_path
            _, ext = os.path.splitext(file_path)
            if ext.lower() in VIDEO_EXTENSIONS and not is_processed(file_path) and file_path not in self.new_videos:
                logger.info(f"检测到修改的视频: {file_path}")
                process_video(file_path)


def print_statistics():
    """打印当前统计信息"""
    logger.info("=" * 50)
    logger.info("当前统计信息:")
    logger.info(f"发现视频总数: {total_videos_found}")
    logger.info(f"已处理视频数: {total_videos_processed}")
    logger.info(f"已跳过视频数: {total_videos_skipped}")
    logger.info(f"新增视频数量: {new_videos_detected}")
    logger.info(f"待处理视频数: {pending_videos}")
    logger.info("=" * 50)


def setup_logging(log_dir=None):
    """
    设置日志配置，使用TimedRotatingFileHandler确保日志实时刷新
    
    参数:
        log_dir: 日志目录，如果为None则使用默认目录(项目根目录下的logs)
    """
    # 如果没有指定日志目录，则使用默认目录
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 配置日志
    log_file = os.path.join(log_dir, "smart_cap.log")

    # 设置日志级别
    logger.setLevel(logging.INFO)

    # 清除已有的处理器，防止重复添加
    if logger.handlers:
        logger.handlers.clear()

    # 基础格式化器
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 使用TimedRotatingFileHandler，按天轮转
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_file,
        when='D',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.INFO)
    
    # 创建标准控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    console_handler.setLevel(logging.INFO)

    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"日志系统初始化完成，日志文件: {log_file}")

# 确保程序退出时正确关闭日志处理器
import atexit
def close_logger():
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)
atexit.register(close_logger)


def main():
    """主入口点。"""
    # 初始化基本日志配置（仅控制台输出）
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
    
    # 加载配置
    load_config()
    
    # 现在可以设置日志到TARGET_DIR上一层的.cache/logs目录
    logs_dir = os.path.join(os.path.dirname(TARGET_DIR), ".cache", "logs")
    setup_logging(logs_dir)
    
    # 确保目标目录存在
    os.makedirs(TARGET_DIR, exist_ok=True)

    # 处理现有视频
    scan_directory(SOURCE_DIR)

    # 设置watchdog观察者
    event_handler = VideoHandler()
    observer = Observer()
    observer.schedule(event_handler, SOURCE_DIR, recursive=True)
    observer.start()

    logger.info(f"监视 {SOURCE_DIR} 的变化")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
