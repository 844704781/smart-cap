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

1. 创建适当的配置文件，修改source_dir和target_dir路径为容器内路径

2. 使用以下命令运行容器，挂载视频源目录和字幕输出目录
   ```bash
   docker run -d \
     --name smart-cap \
     -v /本地源视频目录:/容器内源视频目录 \
     -v /本地字幕输出目录:/容器内字幕输出目录 \
     smart-cap
   ```

   例如:
   ```bash
   docker run -d \
     --name smart-cap \
     -v /home/user/videos:/app/videos \
     -v /home/user/subtitles:/app/subtitles \
     smart-cap
   ```

3. 查看容器日志
   ```bash
   docker logs -f smart-cap
   ```

### Docker Compose部署

也可以使用Docker Compose来简化部署:

1. 创建`docker-compose.yml`文件:
   ```yaml
   version: '3'
   services:
     smart-cap:
       build: .
       volumes:
         - /本地源视频目录:/容器内源视频目录
         - /本地字幕输出目录:/容器内字幕输出目录
       restart: unless-stopped
   ```

2. 启动服务
   ```bash
   docker-compose up -d
   ```

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
