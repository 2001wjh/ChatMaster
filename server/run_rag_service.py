"""
RAG服务启动脚本
处理路径问题并启动FastAPI服务
"""

import os
import sys
import uvicorn

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../"))

if __name__ == "__main__":
    # 启动FastAPI服务
    uvicorn.run(
        "rag_service.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        reload_dirs=["rag_service"]
    ) 