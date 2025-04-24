#!/bin/bash

echo "正在停止ChatMaster服务..."

# 检查进程ID文件
if [ -f .pids ]; then
    # 读取进程ID
    read BACKEND_PID FRONTEND_PID < .pids
    
    # 停止后端服务
    if ps -p $BACKEND_PID > /dev/null; then
        echo "停止后端服务 (PID: $BACKEND_PID)..."
        kill $BACKEND_PID
        echo "后端服务已停止"
    else
        echo "后端服务不在运行"
    fi
    
    # 停止前端服务
    if ps -p $FRONTEND_PID > /dev/null; then
        echo "停止前端服务 (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID
        echo "前端服务已停止"
    else
        echo "前端服务不在运行"
    fi
    
    # 删除进程ID文件
    rm .pids
else
    # 尝试通过端口查找进程
    echo "未找到进程ID文件，尝试通过端口查找进程..."
    
    # 查找占用8000端口的进程
    BACKEND_PID=$(lsof -ti:8000)
    if [ ! -z "$BACKEND_PID" ]; then
        echo "停止在端口8000上运行的后端服务 (PID: $BACKEND_PID)..."
        kill $BACKEND_PID
        echo "后端服务已停止"
    else
        echo "未找到在端口8000上运行的后端服务"
    fi
    
    # 查找占用8501端口的进程
    FRONTEND_PID=$(lsof -ti:8501)
    if [ ! -z "$FRONTEND_PID" ]; then
        echo "停止在端口8501上运行的前端服务 (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID
        echo "前端服务已停止"
    else
        echo "未找到在端口8501上运行的前端服务"
    fi
fi

echo "所有服务已停止" 