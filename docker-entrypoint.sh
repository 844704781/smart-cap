#!/bin/bash
set -e

# 设置默认值
SOURCE_DIR=${SOURCE_DIR:-"/app/videos"}
TARGET_DIR=${TARGET_DIR:-"/app/subtitles"}
CACHE_DIR=${CACHE_DIR:-"/app/cache"}

# 确保路径存在
mkdir -p "$SOURCE_DIR"
mkdir -p "$TARGET_DIR"
mkdir -p "$CACHE_DIR/logs"

# 强制设置为生产环境
export SERVICE_PROFILE=prod

# 生成主配置文件application.yaml
cat > /app/application.yaml <<EOF
# 应用主配置文件
service:
  # 当前环境配置 (dev=开发环境, prod=生产环境)
  profile: prod
EOF

echo "已生成主配置文件，强制使用生产环境"

# 生成application-prod.yaml配置文件
cat > /app/application-prod.yaml <<EOF
# 生产环境配置 (自动生成)
paths:
  # 生产环境源视频目录
  source_dir: $SOURCE_DIR
  # 生产环境字幕输出目录
  target_dir: $TARGET_DIR
  # 生产环境缓存目录
  cache_dir: $CACHE_DIR
EOF

echo "已生成配置文件，使用以下路径:"
echo "源视频目录: $SOURCE_DIR"
echo "字幕输出目录: $TARGET_DIR"
echo "缓存目录: $CACHE_DIR"
echo "环境模式: $SERVICE_PROFILE"

# 启动应用
exec python app.py