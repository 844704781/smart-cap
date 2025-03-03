import atexit
import logging
import os
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import TimedRotatingFileHandler
from typing import Dict, Optional, List

import yaml
from moviepy.editor import VideoFileClip
from proglog import ProgressBarLogger
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from srt.ksasr import KuaiShouASR

# 视频文件扩展名
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']

# 全局变量
SOURCE_DIR = ""
TARGET_DIR = ""
CACHE_DIR = ""
DB_PATH = ""

# 统计信息全局变量
total_videos_found = 0
total_videos_processed = 0
total_videos_skipped = 0
new_videos_detected = 0
pending_videos = 0

# 创建线程锁，用于保护全局变量的并发访问
stats_lock = threading.Lock()

# 创建线程池
video_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="video_processor")

# 添加一个新的全局变量来跟踪正在处理的视频和任务
processing_videos = set()
processing_lock = threading.Lock()
running_tasks = {}  # 用于跟踪正在运行的任务，键为视频路径，值为Future对象

logger = logging.getLogger(__name__)


def load_config():
    """从配置文件加载配置，支持不同环境(dev/prod)"""
    global SOURCE_DIR, TARGET_DIR, CACHE_DIR, DB_PATH

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
        CACHE_DIR = env_config.get('paths', {}).get('cache_dir')

        if not SOURCE_DIR or not TARGET_DIR or not CACHE_DIR:
            raise ValueError("配置文件中缺少必要的路径配置(source_dir 或 target_dir 或 cache_dir)")

        # 创建.cache目录并将数据库文件存储在其中
        cache_dir = CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        DB_PATH = os.path.join(cache_dir, ".video_srt_index")

        logger.info(f"配置加载成功: SOURCE_DIR={SOURCE_DIR}, TARGET_DIR={TARGET_DIR}, CACHE_DIR={CACHE_DIR}")
        logger.info(f"数据库文件位置: {DB_PATH}.json")
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
    db_data = _load_db()
    return video_path in db_data


def get_all_processed_videos() -> List[str]:
    """获取所有已处理的视频列表"""
    db_data = _load_db()
    return list(db_data.keys())


def update_db(video_path: str, srt_path: Optional[str] = None) -> None:
    """更新数据库的处理信息。"""
    db_data = _load_db()
    db_data[video_path] = {
        "video_path": video_path,
        "srt_path": srt_path
    }
    _save_db(db_data)


def _load_db() -> Dict:
    """从JSON文件加载数据库。"""
    try:
        if os.path.exists(DB_PATH + '.json'):
            with open(DB_PATH + '.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"加载数据库时出错: {e}")
        return {}


def _save_db(db_data: Dict) -> None:
    """保存数据库到JSON文件。"""
    try:
        with open(DB_PATH + '.json', 'w', encoding='utf-8') as f:
            json.dump(db_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存数据库时出错: {e}")


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
    # 建议添加文件存在检查
    if not os.path.exists(file_path):
        logger.warning(f"文件不存在: {file_path}")
        return False

    # 初始化文件大小监控
    previous_size = -1
    stable_count = 0
    max_stable_count = 5  # 文件大小连续稳定的次数阈值
    logger.info(f"检查文件是否正在上传: {file_path}, 检查时间大约 {max_stable_count} s")

    start_time = time.time()

    while time.time() - start_time < timeout:
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
    srt_path = os.path.join(target_subdir, f"{file_base}.txt")

    return {
        "audio_path": audio_path,
        "srt_path": srt_path
    }


def process_video(video_path: str) -> None:
    """处理视频，提取音频并生成字幕文件"""
    global total_videos_processed, pending_videos, total_videos_skipped

    try:
        # 如果已处理过则跳过
        if is_processed(video_path):
            logger.info(f"跳过已处理的视频: {video_path}")
            with stats_lock:
                total_videos_skipped += 1
            return

        logger.info(f"准备处理视频: {video_path}")

        # 确保文件已经完全上传
        if not is_file_ready(video_path):
            logger.error(f"文件未准备好，跳过处理: {video_path}")
            with stats_lock:
                total_videos_skipped += 1
            return

        # 生成目标路径
        target_paths = get_target_paths(video_path)
        audio_path = target_paths["audio_path"]
        srt_path = target_paths["srt_path"]

        # 检查SRT文件是否已存在
        if os.path.exists(srt_path):
            logger.info(f"SRT文件已存在: {srt_path}")
            update_db(video_path, srt_path)
            with stats_lock:
                total_videos_skipped += 1
            return

        # 提取音频
        if not extract_audio(video_path, audio_path):
            update_db(video_path, None)
            with stats_lock:
                total_videos_skipped += 1
            return

        # 生成SRT
        if generate_srt(audio_path, srt_path):
            update_db(video_path, srt_path)
            logger.info(f"成功生成SRT: {srt_path}")
            with stats_lock:
                total_videos_processed += 1
                if pending_videos > 0:
                    pending_videos -= 1
        else:
            update_db(video_path, None)
            logger.error(f"为 {video_path} 生成SRT失败")
            with stats_lock:
                total_videos_skipped += 1
                if pending_videos > 0:
                    pending_videos -= 1

        logger.info(f"视频处理完成: {video_path}")
    except Exception as e:
        logger.error(f"处理视频时出错: {e}")
        # 处理失败也视为已跳过
        with stats_lock:
            total_videos_skipped += 1
            if pending_videos > 0:
                pending_videos -= 1


def scan_directory(directory: str) -> None:
    """扫描目录，处理所有未处理的视频文件"""
    global total_videos_found, total_videos_skipped, total_videos_processed, pending_videos

    # 获取所有视频文件
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file_path)
            if ext.lower() in VIDEO_EXTENSIONS:
                video_files.append(file_path)

    # 获取已处理的视频
    processed_videos = get_all_processed_videos()

    # 重置计数并更新统计信息
    with stats_lock:
        total_videos_found = len(video_files)
        pending_videos = 0
        total_videos_processed = len(processed_videos)
        # 计算已跳过数量 = 已处理数量
        total_videos_skipped = total_videos_processed

    # 处理所有未处理的视频
    for video_file in video_files:
        if video_file not in processed_videos:
            with stats_lock:
                pending_videos += 1
                # 减少已跳过计数
                if total_videos_skipped > 0:
                    total_videos_skipped -= 1
            # 提交任务到任务池，检查运行中的任务数量
            submit_task(video_file, process_video)

    logger.info(f"目录扫描完成: 共发现 {total_videos_found} 个视频文件已提交到线程池处理")


class VideoHandler(FileSystemEventHandler):
    """监视新视频文件并处理它们。"""

    def __init__(self):
        super().__init__()
        self.new_videos = []

    def on_created(self, event):
        """处理新创建的文件事件"""
        global new_videos_detected, pending_videos

        if not event.is_directory and os.path.exists(event.src_path):
            file_path = event.src_path
            # 检查是否是视频文件且未处理
            if any(file_path.lower().endswith(ext) for ext in VIDEO_EXTENSIONS) and not is_processed(file_path):
                # 检查是否已在处理队列中
                with processing_lock:
                    if file_path in processing_videos:
                        logger.info(f"视频已在处理队列中，跳过: {file_path}")
                        return

                # 更新统计信息
                with stats_lock:
                    new_videos_detected += 1
                    pending_videos += 1

                logger.info("-" * 70)
                logger.info(f"检测到新视频 [{new_videos_detected}]: {file_path}")
                logger.info(f"当前累计未处理视频数: {pending_videos}")

                # 在线程池中执行视频处理
                submit_task(file_path, self._process_video_wrapper)

    def on_modified(self, event):
        """处理文件修改事件"""
        # 某些系统会同时触发created和modified事件
        # 仅当数据库中不存在且不在处理队列中时才处理
        if not event.is_directory and os.path.exists(event.src_path):
            file_path = event.src_path
            if any(file_path.lower().endswith(ext) for ext in VIDEO_EXTENSIONS) and not is_processed(file_path):
                # 检查是否已在处理队列中
                with processing_lock:
                    if file_path in processing_videos:
                        logger.info(f"视频已在处理队列中，跳过: {file_path}")
                        return

                logger.info(f"检测到修改的视频: {file_path}")
                # 更新待处理视频计数
                with stats_lock:
                    global pending_videos
                    pending_videos += 1
                # 在线程池中执行视频处理
                submit_task(file_path, self._process_video_wrapper)

    def _process_video_wrapper(self, file_path):
        """处理视频的包装函数，确保不会重复处理同一个视频"""
        global pending_videos

        # 检查视频是否已经在处理中
        with processing_lock:
            if file_path in processing_videos:
                logger.info(f"视频已在处理队列中，跳过: {file_path}")
                return
            processing_videos.add(file_path)

        try:
            process_video(file_path)
        except Exception as e:
            logger.error(f"处理视频 {file_path} 时发生错误: {e}")
            # 更新统计信息
            with stats_lock:
                if pending_videos > 0:
                    pending_videos -= 1
        finally:
            # 确保无论成功还是失败，都减少待处理视频计数并从处理中列表移除
            with stats_lock:
                if pending_videos > 0:
                    pending_videos -= 1

            with processing_lock:
                processing_videos.discard(file_path)


def print_statistics():
    """打印当前统计信息"""
    logger.info("=" * 50)
    logger.info("当前统计信息:")
    with stats_lock:
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
    handler = TimedRotatingFileHandler(
        log_file,
        when='midnight',
        backupCount=30,  # 保留30天日志
        encoding='utf-8'
    )
    # 可以考虑添加日志大小限制
    # handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    handler.setFormatter(log_format)
    handler.setLevel(logging.INFO)

    # 创建标准控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    console_handler.setLevel(logging.INFO)

    # 添加处理器到logger
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    logger.info(f"日志系统初始化完成，日志文件: {log_file}")


def close_logger():
    """关闭日志和线程池。"""

    # 打印最终统计信息
    logger.info("=" * 70)
    logger.info("程序结束，最终统计信息:")
    with stats_lock:
        logger.info(f"发现视频总数: {total_videos_found}")
        logger.info(f"成功处理视频数: {total_videos_processed}")
        logger.info(f"跳过视频数: {total_videos_skipped}")
        logger.info(f"检测到新视频数: {new_videos_detected}")
        logger.info(f"未处理视频数: {pending_videos}")
    logger.info("=" * 70)

    # 关闭线程池
    logger.info("正在关闭线程池...")
    video_executor.shutdown(wait=True)
    logger.info("线程池已关闭")

    # 关闭日志处理器
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


atexit.register(close_logger)


def reset_statistics():
    """重置并初始化所有统计计数器"""
    global total_videos_found, total_videos_processed, total_videos_skipped, new_videos_detected, pending_videos

    # 获取当前处理状态
    all_videos = []
    for root, _, files in os.walk(SOURCE_DIR):
        for file in files:
            if any(file.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                all_videos.append(os.path.join(root, file))

    processed_videos = get_all_processed_videos()

    # 重置所有计数器
    with stats_lock:
        total_videos_found = len(all_videos)
        total_videos_processed = len(processed_videos)
        total_videos_skipped = total_videos_processed  # 已处理的视频也算作已跳过
        new_videos_detected = 0
        pending_videos = 0

    logger.info("统计计数器已重置")


def get_all_videos() -> List[str]:
    """获取所有符合条件的视频文件路径"""
    all_videos = []
    for root, _, files in os.walk(SOURCE_DIR):
        for file in files:
            if any(file.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                all_videos.append(os.path.join(root, file))
    return all_videos


def check_processing_state():
    """检查处理状态，清理可能卡住的文件"""
    with processing_lock:
        stuck_files = []
        for file_path in processing_videos:
            # 检查此文件是否已在数据库中（可能处理完成但没从列表移除）
            if is_processed(file_path):
                stuck_files.append(file_path)
                logger.warning(f"发现已处理但未从处理列表移除的视频: {file_path}")

        # 清理卡住的文件
        for file_path in stuck_files:
            processing_videos.discard(file_path)

        if stuck_files:
            logger.info(f"清理了 {len(stuck_files)} 个卡住的文件")


def submit_task(video_path, task_func):
    """
    智能提交任务到线程池，避免过多任务积压
    
    Args:
        video_path: 视频文件路径
        task_func: 要执行的任务函数
    """
    global running_tasks
    
    # 清理已完成的任务
    completed_tasks = [path for path, future in running_tasks.items() if future.done()]
    for path in completed_tasks:
        future = running_tasks.pop(path)
        try:
            # 检查任务是否有异常
            if future.exception():
                logger.error(f"任务处理 {path} 失败: {future.exception()}")
        except Exception:
            pass
    
    # 检查当前正在运行的任务数量
    running_count = len(running_tasks)
    max_workers = video_executor._max_workers
    
    if running_count < max_workers:
        # 线程池未满，直接提交任务
        logger.info(f"提交新任务处理视频: {video_path} (运行中: {running_count}/{max_workers})")
        future = video_executor.submit(task_func, video_path)
        running_tasks[video_path] = future
    else:
        # 线程池已满，定期尝试提交
        logger.info(f"线程池已满 ({running_count}/{max_workers})，将延迟处理视频: {video_path}")
        
        # 启动一个守护线程来等待空闲线程并提交任务
        def wait_and_submit():
            while True:
                time.sleep(1)  # 每秒检查一次
                
                # 再次检查任务是否已在运行
                if video_path in running_tasks:
                    return
                
                # 清理已完成的任务
                completed = [p for p, f in running_tasks.items() if f.done()]
                for p in completed:
                    running_tasks.pop(p)
                
                # 检查是否有空闲线程
                if len(running_tasks) < max_workers:
                    logger.info(f"线程池有空闲，现在提交延迟的视频处理: {video_path}")
                    future = video_executor.submit(task_func, video_path)
                    running_tasks[video_path] = future
                    return
        
        # 启动守护线程
        submit_thread = threading.Thread(target=wait_and_submit, daemon=True)
        submit_thread.start()


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

    # 初始化统计计数器
    reset_statistics()

    # 处理现有视频
    scan_directory(SOURCE_DIR)

    # 设置watchdog观察者
    event_handler = VideoHandler()
    observer = Observer()
    observer.schedule(event_handler, SOURCE_DIR, recursive=True)
    observer.start()

    logger.info(f"监视 {SOURCE_DIR} 的变化")

    # 建议添加定期统计和资源监控
    def periodic_stats():
        while True:
            time.sleep(6)  # 每小时打印一次统计信息
            print_statistics()
            # 可以添加内存使用监控
            # log_memory_usage()

    stats_thread = threading.Thread(target=periodic_stats, daemon=True)
    stats_thread.start()

    try:
        last_stats_time = time.time()
        stats_interval = 6  # 每10分钟打印一次统计信息

        # 每小时执行一次计数校验
        last_check_time = time.time()
        check_interval = 3600  # 1小时

        while True:
            time.sleep(1)

            # 定期打印统计信息
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                last_stats_time = current_time

            # 定期校验计数
            if current_time - last_check_time >= check_interval:
                # 校验待处理视频计数
                with stats_lock:
                    # 获取真实待处理数量
                    all_videos = get_all_videos()
                    processed_videos = get_all_processed_videos()
                    actual_pending = sum(1 for v in all_videos if v not in processed_videos)

                    # 如果计数不一致，进行修正
                    if actual_pending != pending_videos:
                        logger.warning(f"待处理视频计数不一致，修正: {pending_videos} -> {actual_pending}")
                        pending_videos = actual_pending

                last_check_time = current_time

            # 每小时检查一次处理状态
            if current_time - last_check_time >= check_interval:
                check_processing_state()
                last_check_time = current_time
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        traceback.print_exc()
    finally:
        # 确保资源正确释放
        logger.info("正在关闭资源...")
        video_executor.shutdown(wait=True)
        close_logger()


if __name__ == "__main__":
    main()
