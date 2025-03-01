# Smart-Cap: 智能视频字幕批量生成工具

> 这是一个能够自动监控和处理视频文件的工具，可以批量从视频中提取音频并生成SRT格式的字幕文件。

> 广告: 本人接活,如有需求,请邮: 844704781@qq.com

## 功能特点

- 支持批量处理视频文件，自动提取音频并生成字幕
- 实时监控源目录，自动处理新增视频文件
- 语音识别，生成高质量字幕
- 支持多种视频格式（.mp4, .avi, .mov, .mkv）
- 支持不同环境配置(开发/生产)
- 详细的日志记录系统，实时显示处理进度

## 系统要求

- Python 3.8 或更高版本
- 支持的操作系统: Windows, Linux, macOS

## 安装步骤

1. 克隆或下载本项目

2. 创建并激活虚拟环境(可选但推荐)
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # Linux/macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. 安装依赖库
   ```bash
   pip install -r requirements.txt
   ```

## 配置说明

项目使用YAML文件进行配置管理，包含三个配置文件:

1. `application.yaml`: 主配置文件，用于设置当前环境(dev/prod)
   ```yaml
   service:
     profile: dev  # 切换为prod使用生产环境配置
   ```

2. `application-dev.yaml`: 开发环境配置
   ```yaml
   paths:
     source_dir: C:\path\to\your\videos  # 源视频目录
     target_dir: C:\path\to\your\subtitles  # 字幕输出目录
   ```

3. `application-prod.yaml`: 生产环境配置
   ```yaml
   paths:
     source_dir: /path/to/your/videos  # 源视频目录
     target_dir: /path/to/your/subtitles  # 字幕输出目录
   ```

## 使用方法

1. 修改配置文件中的目录设置，指向你的视频源目录和字幕输出目录

2. 启动程序
   ```bash
   python app.py
   ```

3. 程序将自动:
   - 扫描源目录中的所有视频文件
   - 对未处理的视频文件进行音频提取和字幕生成
   - 持续监控源目录，处理新增的视频文件
   - 在目标目录中生成对应的SRT字幕文件

4. 查看日志
   - 日志存储在 `[TARGET_DIR上级目录]/.cache/logs/` 目录中
   - 在PowerShell中可以使用以下命令实时查看日志:
     ```powershell
     Get-Content -Path "path\to\cache\logs\smart_cap.log" -Tail 10 -Wait -Encoding utf8
     ```

## Docker部署

### 构建Docker镜像

1. 确保您已安装Docker

2. 在项目根目录下构建Docker镜像
   ```bash
   docker build -t smart-cap .
   ```

### 使用Docker运行

1. 使用以下命令运行容器，通过环境变量配置源目录和目标目录

   **Linux 环境：**
   ```bash
   docker run -d \
     --name smart-cap \
     --restart unless-stopped \
     -e SOURCE_DIR=/app/videos \
     -e TARGET_DIR=/app/subtitles \
     -e CACHE_DIR=/app/cache \
     -v /本地源视频目录:/app/videos \
     -v /本地字幕输出目录:/app/subtitles \
     -v /本地缓存目录:/app/cache \
     smart-cap
   ```

   **Windows WSL 环境：**
   ```bash
   # 必须使用WSL路径格式，如/mnt/c/开头
   docker run -d \
     --name smart-cap \
     --restart unless-stopped \
     -e SOURCE_DIR=/app/videos \
     -e TARGET_DIR=/app/subtitles \
     -e CACHE_DIR=/app/cache \
     -v /mnt/c/Users/用户名/视频路径:/app/videos \
     -v /mnt/c/Users/用户名/字幕路径:/app/subtitles \
     -v /mnt/c/Users/用户名/缓存路径:/app/cache \
     smart-cap
   ```

   例如:
   ```bash
   docker run -d \
     --name smart-cap \
     -e SOURCE_DIR=/app/videos \
     -e TARGET_DIR=/app/subtitles \
     -e CACHE_DIR=/app/cache \
     -v /home/user/videos:/app/videos \
     -v /home/user/subtitles:/app/subtitles \
     -v /home/user/cache:/app/cache \
     smart-cap
   ```

   **注意**：如果路径包含中文字符或空格，可能需要使用以下方法:
   ```bash
   # 在WSL中创建符号链接到不含特殊字符的路径
   mkdir -p ~/temp-videos ~/temp-subtitles ~/temp-cache
   ln -sf "/mnt/c/带中文的路径" ~/temp-videos/source
   ln -sf "/mnt/c/带中文的输出路径" ~/temp-subtitles/target
   ln -sf "/mnt/c/带中文的缓存路径" ~/temp-cache/cache
   
   # 然后使用符号链接挂载
   docker run -d \
     --name smart-cap \
     -e SOURCE_DIR=/app/videos \
     -e TARGET_DIR=/app/subtitles \
     -e CACHE_DIR=/app/cache \
     -v ~/temp-videos/source:/app/videos \
     -v ~/temp-subtitles/target:/app/subtitles \
     -v ~/temp-cache/cache:/app/cache \
     smart-cap
   ```

2. 查看容器日志
   ```bash
   docker logs -f smart-cap
   ```

### 排查容器问题

如果遇到容器运行问题:

1. 使用以下命令检查容器状态:
   ```bash
   docker ps -a
   ```

2. 使用以下命令查看容器日志:
   ```bash
   docker logs smart-cap
   ```

3. 如需进入容器内部调试:
   ```bash
   docker exec -it smart-cap bash
   ```

4. 如需重新启动容器:
   ```bash
   docker restart smart-cap
   ```

### Docker Compose部署

也可以使用Docker Compose来简化部署:

1. 创建`docker-compose.yml`文件:
   ```yaml
   version: '3'
   services:
     smart-cap:
       build: .
       restart: unless-stopped
       environment:
         - SOURCE_DIR=/app/videos
         - TARGET_DIR=/app/subtitles
         - CACHE_DIR=/app/cache
       volumes:
         - ./videos:/app/videos
         - ./subtitles:/app/subtitles
         - ./cache:/app/cache
   ```

2. 启动服务
   ```bash
   docker-compose up -d
   ```

### Docker开机自启动设置

Docker提供了多种方式确保容器开机自动启动:

1. **使用`--restart`参数（推荐）**:
   ```bash
   # 已添加到上述所有docker run命令中
   --restart unless-stopped
   ```
   
   参数选项说明:
   - `no`: 默认值，不自动重启容器
   - `on-failure`: 容器非正常退出时(退出状态非0)才重启
   - `always`: 无论容器以何种状态退出都重启
   - `unless-stopped`: 容器退出时总是重启，除非容器被显式停止

2. **确保Docker服务自启动**:
   ```bash
   # Ubuntu/Debian
   sudo systemctl enable docker
   
   # CentOS/RHEL
   sudo systemctl enable docker
   
   # Windows
   # 在"任务管理器" -> "启动" -> 启用Docker Desktop
   ```

3. **创建系统服务（高级用户）**:
   创建`/etc/systemd/system/smart-cap.service`文件:
   ```
   [Unit]
   Description=Smart-Cap Container
   After=docker.service
   Requires=docker.service

   [Service]
   Type=oneshot
   RemainAfterExit=yes
   ExecStart=/usr/bin/docker start smart-cap
   ExecStop=/usr/bin/docker stop smart-cap

   [Install]
   WantedBy=multi-user.target
   ```
   
   启用服务:
   ```bash
   sudo systemctl enable smart-cap.service
   ```

### 环境变量说明

Docker镜像支持以下环境变量:

| 环境变量 | 说明 | 默认值 |
|---------|-----|-------|
| SOURCE_DIR | 源视频目录路径 | /app/videos |
| TARGET_DIR | 字幕输出目录路径 | /app/subtitles |
| CACHE_DIR | 缓存目录路径(日志和数据库文件) | /app/cache |
| SERVICE_PROFILE | 运行环境配置(注意：在Docker中总是强制为prod) | prod |

**注意**: Docker容器内部无论如何设置`SERVICE_PROFILE`，都会被强制设为`prod`环境，以确保正确的路径处理。

## 日志和临时文件

- 程序生成的所有日志文件存储在 `[TARGET_DIR上级目录]/.cache/logs/` 目录中
- 处理记录数据库文件存储在 `[TARGET_DIR上级目录]/.cache/` 目录中
- 临时音频文件在处理完成后会自动删除

## 故障排除

如果遇到问题:

1. 检查日志文件了解详细错误信息
2. 确保配置的目录路径正确且有足够的读写权限
3. 检查网络连接是否正常(字幕生成需要访问在线API)
4. 确保视频文件格式为支持的类型(.mp4, .avi, .mov, .mkv)

## 许可证

默认使用MIT许可证
