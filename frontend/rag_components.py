"""
RAG组件模块
为Streamlit前端提供文档上传和知识库管理功能
"""

import os
import tempfile
import requests
import streamlit as st
from typing import List, Dict, Any, Optional

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


def rag_query(
    question: str, 
    knowledge_base_id: str, 
    conversation_id: Optional[str] = None,
    top_k: int = 3,
    search_type: str = "hybrid"
) -> Dict[str, Any]:
    """
    查询RAG系统
    
    Args:
        question: 用户问题
        knowledge_base_id: 知识库ID
        conversation_id: 对话ID (可选)
        top_k: 返回的结果数量
        search_type: 搜索类型 (vector, keyword, hybrid)
        
    Returns:
        查询结果
    """
    try:
        response = requests.post(
            f"{RAG_API_BASE}/qa/ask",
            json={
                "question": question,
                "knowledge_base_id": knowledge_base_id,
                "conversation_id": conversation_id,
                "top_k": top_k,
                "search_type": search_type
            }
        )
        
        if response.status_code != 200:
            return {
                "error": True,
                "message": f"查询失败: {response.text}"
            }
        
        return {
            "error": False,
            "data": response.json()
        }
    except Exception as e:
        return {
            "error": True,
            "message": f"连接RAG服务失败: {str(e)}"
        }


def display_rag_result(result: Dict[str, Any]):
    """
    显示RAG查询结果
    
    Args:
        result: RAG查询结果
    """
    if result.get("error", False):
        st.error(result["message"])
        return
    
    data = result["data"]
    
    # 显示回答
    st.markdown(f"{data['answer']}")
    
    # 显示参考来源
    if data.get("sources"):
        with st.expander("查看参考来源", expanded=False):
            for i, source in enumerate(data["sources"]):
                st.markdown(f"**来源 {i+1}**")
                if "metadata" in source and "filename" in source["metadata"]:
                    st.markdown(f"文档: {source['metadata']['filename']}")
                st.markdown(f"相关内容: {source['content']}")
                st.markdown("---")
    
    # 显示推荐问题
    if data.get("recommended_questions"):
        st.markdown("### 你可能还想问:")
        cols = st.columns(len(data["recommended_questions"]))
        for i, question in enumerate(data["recommended_questions"]):
            with cols[i]:
                if st.button(question, key=f"rec_q_{i}"):
                    # 将问题设置为用户输入
                    st.session_state["user_input"] = question
                    # 自动提交表单
                    st.session_state["submit_question"] = True 