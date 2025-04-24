#!/bin/bash

# 启动RAG服务(后台运行)
cd server
echo "启动RAG服务..."
python run_rag_service.py &
RAG_PID=$!
echo "RAG服务已启动，PID: $RAG_PID"
cd ..

# 等待RAG服务启动
echo "等待RAG服务启动..."
sleep 5

# 启动前端应用
cd frontend
echo "启动Streamlit前端..."
streamlit run app.py

# 收到中断信号时，终止所有进程
trap 'echo "收到中断信号，终止所有进程..."; kill $RAG_PID; exit' INT

# 等待前端退出
wait 