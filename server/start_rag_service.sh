#!/bin/bash

# 设置Python路径
export PYTHONPATH=$PYTHONPATH:$(cd .. && pwd)

# 启动RAG服务
uvicorn rag_service.main:app --reload --host 0.0.0.0 --port 8000 