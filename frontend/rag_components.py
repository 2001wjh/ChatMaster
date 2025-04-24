"""
RAG组件模块
为Streamlit前端提供文档上传和知识库管理功能
"""

import os
import tempfile
import requests
import streamlit as st
from typing import List, Dict, Any, Optional
import uuid

# RAG服务API地址
RAG_API_BASE = "http://localhost:8000/api"

def render_knowledge_base_management():
    """
    渲染知识库管理界面
    """
    st.header("知识库管理")
    
    # 获取所有知识库
    try:
        response = requests.get(f"{RAG_API_BASE}/kb")
        if response.status_code != 200:
            st.error(f"获取知识库列表失败: {response.text}")
            knowledge_bases = []
        else:
            knowledge_bases = response.json().get("knowledge_bases", [])
    except Exception as e:
        st.error(f"连接RAG服务失败: {str(e)}")
        knowledge_bases = []
    
    # 创建新知识库
    with st.expander("创建新知识库", expanded=True):
        new_kb_id = st.text_input("知识库ID (仅限英文字母、数字和下划线)")
        new_kb_name = st.text_input("知识库名称")
        new_kb_desc = st.text_area("知识库描述")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("创建知识库", use_container_width=True):
                if not new_kb_id or not new_kb_name:
                    st.error("知识库ID和名称不能为空")
                else:
                    try:
                        response = requests.post(
                            f"{RAG_API_BASE}/kb",
                            json={
                                "id": new_kb_id,
                                "name": new_kb_name,
                                "description": new_kb_desc
                            }
                        )
                        if response.status_code == 200:
                            st.success("知识库创建成功！")
                            # 刷新页面
                            st.experimental_rerun()
                        else:
                            st.error(f"创建知识库失败: {response.text}")
                    except Exception as e:
                        st.error(f"连接RAG服务失败: {str(e)}")
    
    # 显示现有知识库
    if knowledge_bases:
        st.subheader("现有知识库")
        
        for kb in knowledge_bases:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{kb['name']}** (ID: {kb['id']})")
                st.markdown(f"描述: {kb.get('description', '无')}")
            
            with col2:
                if st.button("管理文档", key=f"manage_{kb['id']}", use_container_width=True):
                    st.session_state["current_view"] = "document_management"
                    st.session_state["selected_kb_id"] = kb["id"]
                    st.session_state["selected_kb_name"] = kb["name"]
                    st.experimental_rerun()
            
            with col3:
                if st.button("删除", key=f"delete_{kb['id']}", use_container_width=True):
                    # 弹出确认对话框
                    st.warning(f"确定要删除知识库 '{kb['name']}' 吗？这将删除所有相关文档。")
                    if st.button("确定删除", key=f"confirm_delete_{kb['id']}"):
                        try:
                            response = requests.delete(f"{RAG_API_BASE}/kb/{kb['id']}")
                            if response.status_code == 200:
                                st.success("知识库已删除！")
                                # 刷新页面
                                st.experimental_rerun()
                            else:
                                st.error(f"删除知识库失败: {response.text}")
                        except Exception as e:
                            st.error(f"连接RAG服务失败: {str(e)}")
            
            st.markdown("---")
    else:
        st.info("还没有创建任何知识库。请先创建一个知识库。")


def render_document_management(kb_id: str, kb_name: str):
    """
    渲染文档管理界面
    
    Args:
        kb_id: 知识库ID
        kb_name: 知识库名称
    """
    st.header(f"文档管理 - {kb_name}")
    
    # 返回按钮
    if st.button("返回知识库列表"):
        st.session_state["current_view"] = "knowledge_base_management"
        st.experimental_rerun()
    
    # 上传文档区域
    with st.expander("上传新文档", expanded=True):
        uploaded_file = st.file_uploader("选择文件", type=["pdf", "docx", "txt", "md"])
        document_type = st.selectbox("文档类型", ["教材", "文章", "对话", "语法", "词汇", "其他"])
        description = st.text_area("文档描述")
        
        if uploaded_file and st.button("上传文档"):
            # 显示上传中状态
            with st.spinner("文档上传中..."):
                try:
                    # 保存临时文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        temp_path = tmp.name
                    
                    # 上传到RAG服务
                    files = {"file": (uploaded_file.name, open(temp_path, "rb"), f"application/{uploaded_file.name.split('.')[-1]}")}
                    data = {
                        "knowledge_base_id": kb_id,
                        "document_type": document_type,
                        "description": description
                    }
                    
                    response = requests.post(
                        f"{RAG_API_BASE}/document/upload",
                        files=files,
                        data=data
                    )
                    
                    # 删除临时文件
                    os.unlink(temp_path)
                    
                    if response.status_code == 200:
                        st.success("文档上传成功！")
                    else:
                        st.error(f"上传失败: {response.text}")
                except Exception as e:
                    st.error(f"上传过程出错: {str(e)}")
    
    # 获取文档列表
    try:
        response = requests.get(f"{RAG_API_BASE}/document/list/{kb_id}")
        if response.status_code != 200:
            st.error(f"获取文档列表失败: {response.text}")
            documents = []
        else:
            documents = response.json().get("documents", [])
    except Exception as e:
        st.error(f"连接RAG服务失败: {str(e)}")
        documents = []
    
    # 显示文档列表
    if documents:
        st.subheader("文档列表")
        
        for doc in documents:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{doc['filename']}**")
                st.markdown(f"类型: {doc.get('document_type', '未知')}")
                st.markdown(f"描述: {doc.get('description', '无')}")
                st.markdown(f"状态: {doc.get('status', '未知')}")
                st.markdown(f"分块数量: {doc.get('chunks_count', 0)}")
            
            with col2:
                if st.button("删除", key=f"delete_doc_{doc['document_id']}", use_container_width=True):
                    # 弹出确认对话框
                    st.warning(f"确定要删除文档 '{doc['filename']}' 吗？")
                    if st.button("确定删除", key=f"confirm_delete_doc_{doc['document_id']}"):
                        try:
                            response = requests.delete(
                                f"{RAG_API_BASE}/document/{doc['document_id']}",
                                params={"knowledge_base_id": kb_id}
                            )
                            if response.status_code == 200:
                                st.success("文档已删除！")
                                # 刷新页面
                                st.experimental_rerun()
                            else:
                                st.error(f"删除文档失败: {response.text}")
                        except Exception as e:
                            st.error(f"连接RAG服务失败: {str(e)}")
            
            st.markdown("---")
    else:
        st.info("该知识库中还没有任何文档。请上传文档。")


def get_knowledge_bases() -> List[Dict[str, Any]]:
    """
    获取所有知识库
    
    Returns:
        知识库列表
    """
    try:
        response = requests.get(f"{RAG_API_BASE}/kb")
        if response.status_code != 200:
            return []
        return response.json().get("knowledge_bases", [])
    except Exception as e:
        return []


def display_rag_result(result: Dict[str, Any]):
    """
    显示RAG查询结果
    
    Args:
        result: 查询结果
    """
    if not result:
        return
    
    # 显示答案
    st.markdown("### 回答")
    st.markdown(result.get("answer", ""))
    
    # 显示来源文档
    sources = result.get("sources", [])
    if sources:
        with st.expander("参考来源", expanded=False):
            for i, source in enumerate(sources):
                st.markdown(f"**来源 {i+1}**")
                st.markdown(f"**内容:** {source.get('content', '')}")
                
                # 显示元数据
                metadata = source.get("metadata", {})
                if metadata:
                    st.markdown("**元数据:**")
                    for key, value in metadata.items():
                        if key not in ["chunk_id", "chunk_index"]:
                            st.markdown(f"- {key}: {value}")
                
                # 显示关键词匹配（如果有）
                keyword_matches = source.get("keyword_matches", [])
                if keyword_matches:
                    st.markdown("**关键词匹配:**")
                    matches_text = ", ".join([match.get("text", "") for match in keyword_matches])
                    st.markdown(matches_text)
                
                st.markdown("---")
    
    # 显示推荐问题
    recommended_questions = result.get("recommended_questions", [])
    if recommended_questions:
        st.markdown("### 你可能还想问")
        for question in recommended_questions:
            if st.button(question, key=f"rec_q_{hash(question)}"):
                # 设置该问题为用户输入并提交
                st.session_state.user_input = question
                st.session_state.submit_question = True
                st.experimental_rerun()


def render_advanced_search_options():
    """
    渲染高级搜索选项
    """
    with st.expander("高级检索设置", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            search_type = st.selectbox(
                "检索方式",
                options=["hybrid", "vector", "keyword", "auto"],
                format_func=lambda x: {
                    "hybrid": "混合检索 (向量+关键词)",
                    "vector": "向量检索 (语义相似度)",
                    "keyword": "关键词检索 (精确匹配)",
                    "auto": "自动选择 (根据意图)"
                }[x],
                index=0
            )
            
            top_k = st.slider(
                "返回结果数量",
                min_value=1,
                max_value=10,
                value=5,
                step=1
            )
            
        with col2:
            enable_context = st.checkbox("启用上下文优化", value=True, 
                                        help="基于历史查询优化结果排序")
            
            enable_diversity = st.checkbox("启用结果多样性", value=True,
                                         help="确保结果来源多样化，避免单一来源")
            
            enable_expansion = st.checkbox("启用查询扩展", value=True,
                                         help="自动扩展查询关键词提高查询效果")
    
    with st.expander("过滤器设置", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            use_time_filter = st.checkbox("启用时间过滤", value=False)
            if use_time_filter:
                time_filter = st.text_input("时间表达式 (如: 2023年, 上个月)", "")
            
            use_author_filter = st.checkbox("启用作者过滤", value=False)
            if use_author_filter:
                author_filter = st.text_input("作者名称", "")
                
        with col2:
            use_location_filter = st.checkbox("启用位置过滤", value=False)
            if use_location_filter:
                location_filter = st.text_input("位置名称", "")
    
    # 构建过滤条件
    filters = {}
    
    if use_time_filter and time_filter:
        filters["time_range"] = [time_filter]
        
    if use_author_filter and author_filter:
        filters["author"] = [author_filter]
        
    if use_location_filter and location_filter:
        filters["location"] = [location_filter]
    
    # 构建高级设置
    advanced_settings = {
        "search_type": search_type,
        "top_k": top_k,
        "filters": filters,
        "enable_context_optimization": enable_context,
        "enable_source_diversification": enable_diversity,
        "enable_query_expansion": enable_expansion
    }
    
    return advanced_settings


def render_retrieval_analysis(result: Dict[str, Any]):
    """
    渲染检索分析结果
    
    Args:
        result: 查询结果
    """
    if not result:
        return
    
    with st.expander("查询分析", expanded=False):
        # 显示意图和置信度
        intent = result.get("intent", "未知")
        intent_confidence = result.get("intent_confidence", 0.0)
        
        intent_map = {
            "command": "指令",
            "learning_question": "学习问题",
            "information_seeking": "信息检索",
            "casual_chat": "日常闲聊"
        }
        
        st.markdown(f"**查询意图:** {intent_map.get(intent, intent)} (置信度: {intent_confidence:.2f})")
        
        # 显示实体
        entities = result.get("entities", [])
        if entities:
            st.markdown("**识别实体:**")
            for entity in entities:
                entity_type = entity.get("type", "")
                entity_text = entity.get("text", "")
                
                entity_type_map = {
                    "PERSON": "人物",
                    "LOCATION": "地点",
                    "TIME": "时间",
                    "ORGANIZATION": "组织",
                    "EVENT": "事件"
                }
                
                st.markdown(f"- {entity_text} ({entity_type_map.get(entity_type, entity_type)})")
        
        # 显示检索统计
        sources = result.get("sources", [])
        if sources:
            # 计算不同检索类型的数量
            search_types = {}
            for source in sources:
                search_type = source.get("search_type", "unknown")
                if search_type in search_types:
                    search_types[search_type] += 1
                else:
                    search_types[search_type] = 1
            
            st.markdown("**检索方式分布:**")
            for search_type, count in search_types.items():
                search_type_map = {
                    "vector": "向量检索",
                    "keyword": "关键词检索",
                    "hybrid": "混合检索"
                }
                st.markdown(f"- {search_type_map.get(search_type, search_type)}: {count}个结果")
            
            # 显示得分分布
            scores = [source.get("score", 0) for source in sources]
            if scores:
                st.markdown(f"**相关度分数范围:** {min(scores):.2f} - {max(scores):.2f}")


def analyze_query(query: str):
    """
    分析查询，获取意图和实体信息
    
    Args:
        query: 用户查询
        
    Returns:
        查询分析结果
    """
    try:
        response = requests.post(
            f"{RAG_API_BASE}/qa/analyze_query",
            json={"query": query}
        )
        
        if response.status_code != 200:
            return None
            
        return response.json()
    except Exception as e:
        st.error(f"查询分析失败: {str(e)}")
        return None


def render_enhanced_rag_interface():
    """
    渲染增强版RAG界面
    
    包含高级检索设置、查询分析和结果可视化
    """
    st.header("增强型知识检索")
    
    # 获取所有知识库
    knowledge_bases = get_knowledge_bases()
    kb_options = [{"id": kb["id"], "name": kb["name"]} for kb in knowledge_bases]
    
    if not kb_options:
        st.warning("没有可用的知识库。请先创建一个知识库。")
        if st.button("前往知识库管理"):
            st.session_state["current_view"] = "knowledge_base_management"
            st.experimental_rerun()
        return
    
    # 选择知识库
    kb_names = [kb["name"] for kb in kb_options]
    kb_ids = [kb["id"] for kb in kb_options]
    
    selected_kb_index = st.selectbox(
        "选择知识库:",
        range(len(kb_names)),
        format_func=lambda i: kb_names[i],
        index=0
    )
    
    # 保存选择的知识库ID
    selected_kb_id = kb_ids[selected_kb_index]
    
    # 输入查询
    user_query = st.text_input("输入您的问题:", key="enhanced_rag_query")
    
    # 如果有输入，显示查询分析
    if user_query:
        with st.spinner("分析查询中..."):
            query_analysis = analyze_query(user_query)
            
            if query_analysis:
                intent = query_analysis.get("intent", "")
                is_question = query_analysis.get("is_question", False)
                
                # 根据分析结果给出提示
                if intent == "command":
                    st.info("检测到指令类查询，将使用关键词检索以提高精确度。")
                elif intent == "information_seeking":
                    st.info("检测到信息检索类查询，将使用向量检索以提高相关性。")
                elif intent == "learning_question":
                    st.info("检测到学习类问题，将使用混合检索以获得全面结果。")
                
                if not is_question:
                    st.info("您的输入不是一个问句，可能会影响回答的准确性。")
    
    # 高级检索设置
    advanced_settings = render_advanced_search_options()
    
    # 设置会话ID
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    
    # 执行检索
    col1, col2 = st.columns([1, 3])
    with col1:
        search_button = st.button("开始检索", use_container_width=True)
    
    if search_button and user_query:
        with st.spinner("正在检索相关内容..."):
            # 执行RAG查询
            result = rag_query(
                question=user_query,
                knowledge_base_id=selected_kb_id,
                conversation_id=st.session_state.conversation_id,
                top_k=advanced_settings["top_k"],
                search_type=advanced_settings["search_type"],
                filters=advanced_settings["filters"],
                advanced_settings={
                    "enable_context_optimization": advanced_settings["enable_context_optimization"],
                    "enable_source_diversification": advanced_settings["enable_source_diversification"],
                    "enable_query_expansion": advanced_settings["enable_query_expansion"]
                }
            )
            
            if result:
                # 显示检索分析
                render_retrieval_analysis(result)
                
                # 显示RAG结果
                display_rag_result(result)
            else:
                st.error("检索失败，请重试。")
    
    # 显示历史记录
    with st.expander("对话历史", expanded=False):
        if st.button("清除对话历史"):
            try:
                response = requests.post(f"{RAG_API_BASE}/qa/clear_history/{st.session_state.conversation_id}")
                if response.status_code == 200:
                    st.success("对话历史已清除！")
                    # 创建新的会话ID
                    st.session_state.conversation_id = str(uuid.uuid4())
                else:
                    st.error("清除对话历史失败")
            except Exception as e:
                st.error(f"清除对话历史失败: {str(e)}")


def rag_query(
    question: str, 
    knowledge_base_id: str, 
    conversation_id: Optional[str] = None,
    top_k: int = 3,
    search_type: str = "hybrid",
    filters: Optional[Dict[str, Any]] = None,
    advanced_settings: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    查询RAG系统
    
    Args:
        question: 用户问题
        knowledge_base_id: 知识库ID
        conversation_id: 对话ID (可选)
        top_k: 返回的结果数量
        search_type: 搜索类型 (vector, keyword, hybrid)
        filters: 过滤条件
        advanced_settings: 高级设置
        
    Returns:
        查询结果
    """
    try:
        # 构建请求参数
        payload = {
            "question": question,
            "knowledge_base_id": knowledge_base_id,
            "conversation_id": conversation_id,
            "top_k": top_k,
            "search_type": search_type,
            "filters": filters or {}
        }
        
        # 合并高级设置
        if advanced_settings:
            # 添加高级设置到请求中
            for key, value in advanced_settings.items():
                payload[key] = value
        
        response = requests.post(
            f"{RAG_API_BASE}/qa/ask",
            json=payload
        )
        
        if response.status_code != 200:
            st.error(f"RAG查询失败: {response.text}")
            return None
            
        return response.json()
    except Exception as e:
        st.error(f"RAG查询失败: {str(e)}")
        return None 