#!/bin/bash
set -e

# 定义变量
IMAGE_NAME="smart-cap"
IMAGE_VERSION="1.0.0"
OUTPUT_DIR="./output"
TAR_NAME="${IMAGE_NAME}-${IMAGE_VERSION}.tar"

echo "===== 开始Smart-Cap构建 ====="

# 创建输出目录(如果不存在)
if [ ! -d "$OUTPUT_DIR" ]; then
  echo "创建输出目录: $OUTPUT_DIR"
  echo "执行命令: mkdir -p \"$OUTPUT_DIR\""
  mkdir -p "$OUTPUT_DIR"
fi

# 构建Docker镜像
echo "正在构建Docker镜像: $IMAGE_NAME:$IMAGE_VERSION"
echo "执行命令: docker build -t \"$IMAGE_NAME:$IMAGE_VERSION\" ."
docker build -t "$IMAGE_NAME:$IMAGE_VERSION" .

# 标记latest镜像
echo "标记为latest版本"
echo "执行命令: docker tag \"$IMAGE_NAME:$IMAGE_VERSION\" \"$IMAGE_NAME:latest\""
docker tag "$IMAGE_NAME:$IMAGE_VERSION" "$IMAGE_NAME:latest"

# 保存镜像为tar包
echo "正在将镜像保存为tar包: $TAR_NAME"
echo "执行命令: docker save -o \"$OUTPUT_DIR/$TAR_NAME\" \"$IMAGE_NAME:$IMAGE_VERSION\""
docker save -o "$OUTPUT_DIR/$TAR_NAME" "$IMAGE_NAME:$IMAGE_VERSION"

# 压缩tar包以节省空间
echo "压缩tar包"
echo "执行命令: gzip -f \"$OUTPUT_DIR/$TAR_NAME\""
gzip -f "$OUTPUT_DIR/$TAR_NAME"
TAR_GZ_NAME="${TAR_NAME}.gz"

# 计算文件大小
echo "执行命令: du -h \"$OUTPUT_DIR/$TAR_GZ_NAME\" | cut -f1"
TAR_SIZE=$(du -h "$OUTPUT_DIR/$TAR_GZ_NAME" | cut -f1)
echo "tar包大小: $TAR_SIZE"

echo "===== 构建完成 ====="
echo "Docker镜像已成功构建并保存为: $OUTPUT_DIR/$TAR_GZ_NAME (大小: $TAR_SIZE)"
echo "您可以使用以下命令将其上传到服务器:"
echo "  scp -P <端口> \"$OUTPUT_DIR/$TAR_GZ_NAME\" <用户名>@<服务器地址>:<目标路径>/"