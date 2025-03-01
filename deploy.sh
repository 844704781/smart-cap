#!/bin/bash
set -e

# 定义变量
IMAGE_NAME="smart-cap"
IMAGE_VERSION="1.0.0"
TAR_GZ_NAME="${IMAGE_NAME}-${IMAGE_VERSION}.tar.gz"
REMOTE_PATH="/volume1/Download"

# 定义挂载目录
VIDEO_DIR="/volume1/Bililive/抖音/霞湖世家官方旗舰店"
SUBTITLE_DIR="/volume1/Bililive/抖音/字幕/data"
CACHE_DIR="/volume1/Bililive/抖音/字幕/.cache"

echo "===== 开始Smart-Cap服务端部署 ====="

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "错误: Docker未安装，请先安装Docker"
    exit 1
fi

# 检查tar包是否存在
if [ ! -f "$REMOTE_PATH/$TAR_GZ_NAME" ]; then
    echo "错误: tar包不存在: $REMOTE_PATH/$TAR_GZ_NAME"
    echo "请确保先上传tar包到服务器"
    exit 1
fi

# 确保所有必要的目录都存在
echo "检查并创建必要的目录..."
echo "执行命令: mkdir -p \"$VIDEO_DIR\""
mkdir -p "$VIDEO_DIR"
echo "执行命令: mkdir -p \"$SUBTITLE_DIR\""
mkdir -p "$SUBTITLE_DIR"
echo "执行命令: mkdir -p \"$CACHE_DIR\""
mkdir -p "$CACHE_DIR"
echo "目录创建完成"

# 加载Docker镜像
echo "执行命令: sudo docker load -i $REMOTE_PATH/$TAR_GZ_NAME"
if ! sudo docker load -i "$REMOTE_PATH/$TAR_GZ_NAME"; then
    echo "加载Docker镜像失败"
    exit 1
fi
echo "Docker镜像加载成功"

# 停止并删除现有容器（如果存在）
echo "执行命令: sudo docker rm -f smart-cap 2>/dev/null || true"
sudo docker rm -f smart-cap 2>/dev/null || true
echo "已清理旧容器"

# 启动新容器
echo "执行命令: sudo docker run -d --name smart-cap --restart unless-stopped ... $IMAGE_NAME:$IMAGE_VERSION"
if ! sudo docker run -d \
  --name smart-cap \
  --restart unless-stopped \
  -e SOURCE_DIR=/app/videos \
  -e TARGET_DIR=/app/subtitles \
  -e CACHE_DIR=/app/cache \
  -v "$VIDEO_DIR":/app/videos \
  -v "$SUBTITLE_DIR":/app/subtitles \
  -v "$CACHE_DIR":/app/cache \
  $IMAGE_NAME:$IMAGE_VERSION; then
    echo "启动容器失败"
    exit 1
fi

# 检查容器是否成功启动
echo "执行命令: sudo docker ps | grep -q smart-cap"
if sudo docker ps | grep -q smart-cap; then
    echo "容器成功启动"
else
    echo "容器未成功启动，请检查错误日志"
    sudo docker logs smart-cap
    exit 1
fi

# 显示容器日志以便检查启动状态
echo "执行命令: sudo docker logs --tail 20 smart-cap"
echo "容器最新日志:"
sudo docker logs --tail 20 smart-cap

echo "执行命令: sudo docker ps | grep smart-cap"
echo "容器已启动，查看容器状态:"
sudo docker ps | grep smart-cap

echo "===== 部署完成 ====="
echo "Smart-Cap已成功部署"
echo "您可以使用以下命令查看容器日志:"
echo "  sudo docker logs -f smart-cap" 