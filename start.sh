#!/bin/bash

# 显示欢迎信息
echo "================================="
echo "  ChatMaster 智能英语外教系统   "
echo "================================="
echo "正在启动服务..."

# 检查是否有conda环境
if command -v conda &> /dev/null; then
    echo "检测到conda环境，正在激活chat环境..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate chat || echo "chat环境不存在，请先运行: conda create -n chat python=3.10"
else
    echo "未检测到conda，使用当前Python环境..."
fi

# 检查依赖
echo "正在检查依赖..."
pip install -r requirements.txt > /dev/null

# 设置工作目录
ROOT_DIR=$(pwd)

# 启动后端服务
echo "正在启动后端服务..."
cd $ROOT_DIR
python -m server.rag_service.main > backend.log 2>&1 &
BACKEND_PID=$!
echo "后端服务已启动，PID: $BACKEND_PID"

# 等待后端启动
echo "等待后端服务就绪..."
sleep 5

# 检查后端是否成功启动
if curl -s http://localhost:8000/health > /dev/null; then
    echo "后端服务已就绪!"
else
    echo "后端服务启动失败，请检查backend.log文件"
    exit 1
fi

# 启动前端服务
echo "正在启动前端服务..."
cd $ROOT_DIR
streamlit run frontend/app.py > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "前端服务已启动，PID: $FRONTEND_PID"

# 记录进程ID
echo "$BACKEND_PID $FRONTEND_PID" > .pids

echo ""
echo "========================================"
echo "所有服务已启动！"
echo "访问前端: http://localhost:8501"
echo "API文档: http://localhost:8000/docs"
echo ""
echo "要停止服务，请运行: ./stop.sh"
echo "========================================" 