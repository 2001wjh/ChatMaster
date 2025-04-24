# RAG服务

基于检索增强生成（Retrieval-Augmented Generation）的知识库问答服务。

## 功能特点

- **混合检索**：结合向量检索和关键词检索，提高召回质量
- **上下文优化**：考虑用户之前的查询，优化结果排序
- **多种文档处理**：支持PDF、Word、TXT、Markdown等常见文档格式
- **知识库管理**：提供完整的知识库创建、查询和管理功能
- **推荐问题生成**：自动生成与当前问题相关的推荐问题

## 快速开始

### 环境要求

- Python 3.8+
- OpenAI API Key (用于嵌入和LLM生成)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 设置环境变量

```bash
export OPENAI_API_KEY="your-api-key"
```

### 启动服务

```bash
cd server
uvicorn rag_service.main:app --reload --host 0.0.0.0 --port 8000
```

服务启动后，可以通过 http://localhost:8000/docs 访问API文档。

## API接口

### 知识库管理

- `POST /api/kb`: 创建知识库
- `GET /api/kb`: 获取知识库列表
- `GET /api/kb/{knowledge_base_id}`: 获取知识库详情
- `DELETE /api/kb/{knowledge_base_id}`: 删除知识库
- `POST /api/kb/{knowledge_base_id}/reindex`: 重建知识库索引

### 文档管理

- `POST /api/document/upload`: 上传文档
- `GET /api/document/list/{knowledge_base_id}`: 获取文档列表
- `GET /api/document/{document_id}`: 获取文档元数据
- `DELETE /api/document/{document_id}`: 删除文档

### 问答功能

- `POST /api/qa/ask`: 问答接口
- `POST /api/qa/feedback`: 提交反馈
- `POST /api/qa/clear_history/{conversation_id}`: 清除会话历史
- `GET /api/qa/recommended_questions`: 获取推荐问题

## 系统架构

RAG服务由以下几个核心组件构成：

1. **文档服务**：负责文档解析、处理和存储
2. **知识库服务**：管理知识库的创建和维护
3. **索引服务**：处理文档索引和检索
4. **检索服务**：实现混合检索和上下文优化
5. **问答服务**：生成回答和推荐问题

## 高级配置

可以通过修改配置参数来优化系统性能：

- `vector_weight`：向量检索权重 (默认: 0.7)
- `keyword_weight`：关键词检索权重 (默认: 0.3)
- `enable_context_optimization`：是否启用上下文优化 (默认: true)
- `top_k`：返回的检索结果数量 (默认: 5)
- `chunk_size`：文档分块大小 (默认: 500)
- `chunk_overlap`：文档分块重叠大小 (默认: 50)

## 使用示例

### Python客户端示例

```python
import requests

# 创建知识库
response = requests.post(
    "http://localhost:8000/api/kb",
    json={
        "id": "company_kb",
        "name": "公司知识库",
        "description": "包含公司产品、服务和政策的知识库"
    }
)
print(response.json())

# 上传文档
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/document/upload",
        files={"file": f},
        data={
            "knowledge_base_id": "company_kb",
            "document_type": "pdf",
            "description": "公司产品手册"
        }
    )
print(response.json())

# 提问
response = requests.post(
    "http://localhost:8000/api/qa/ask",
    json={
        "question": "公司的退款政策是什么?",
        "knowledge_base_id": "company_kb",
        "top_k": 5,
        "search_type": "hybrid"
    }
)
print(response.json())
```

## 许可证

MIT 