"""
RAG服务主入口文件
提供基于FastAPI的RESTful API接口
集成检索增强生成的文档问答功能
"""

import os
import sys
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 添加服务器根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 导入路由
from server.rag_service.router import qa_router, document_router, knowledge_base_router, scene_router, retrieval_router

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="RAG服务API",
    description="提供检索增强生成的文档问答功能",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(qa_router.router, prefix="/api/qa", tags=["问答服务"])
app.include_router(document_router.router, prefix="/api/document", tags=["文档服务"])
app.include_router(knowledge_base_router.router, prefix="/api/kb", tags=["知识库服务"])
app.include_router(scene_router.router, prefix="/api/scene", tags=["场景服务"])
app.include_router(retrieval_router.router, prefix="/api/retrieval", tags=["检索服务"])

# 健康检查
@app.get("/health", tags=["健康检查"])
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}

# 主页
@app.get("/", tags=["主页"])
async def root():
    """API主页"""
    return {
        "message": "欢迎使用RAG服务API",
        "documentation": "/docs",
        "health_check": "/health"
    }

# 入口点
if __name__ == "__main__":
    # 获取配置信息
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    
    # 启动服务
    logger.info(f"启动服务: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port) 