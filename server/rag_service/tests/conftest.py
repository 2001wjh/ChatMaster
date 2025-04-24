"""
测试配置文件
设置测试环境和共享资源
"""

import os
import pytest
from typing import Dict, Any

# 测试数据目录
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

# 确保测试数据目录存在
os.makedirs(TEST_DATA_DIR, exist_ok=True)

@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """测试配置"""
    return {
        "model_name": "gpt-3.5-turbo",
        "vector_store_type": "faiss",
        "embedding_model": "text-embedding-ada-002",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "top_k": 5,
        "enable_advanced_features": True,
        "vector_weight": 0.7,
        "keyword_weight": 0.3,
        "enable_context_optimization": True,
        "knowledge_base_dir": TEST_DATA_DIR
    }

@pytest.fixture(scope="session")
def test_index_name() -> str:
    """测试索引名称"""
    return "test_index"

@pytest.fixture(scope="session")
def test_document_path() -> str:
    """测试文档路径"""
    return os.path.join(TEST_DATA_DIR, "test_document.txt") 