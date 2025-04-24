"""
ChatMaster 主页面
展示系统概览和主要功能
"""

import streamlit as st
from typing import Dict, List, Any
import altair as alt
import pandas as pd
import numpy as np
import datetime

def render_welcome_section():
    """渲染欢迎区域"""
    st.title("欢迎使用 ChatMaster 智能英语系统")
    
    st.markdown("""
    <div style="background-color: #f0f7ff; padding: 20px; border-radius: 10px; margin-bottom: 20px">
    <h3>智能英语学习与知识检索平台</h3>
    <p>ChatMaster集成了智能对话、文档检索和知识管理功能，为您提供全方位的英语学习和知识获取体验。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 添加最近更新信息
    st.sidebar.markdown("### 最近更新")
    st.sidebar.markdown(f"**版本**: 2.1.0")
    st.sidebar.markdown(f"**更新日期**: {datetime.date.today().strftime('%Y-%m-%d')}")
    st.sidebar.markdown("**新功能**:")
    st.sidebar.markdown("- ✨ 增强型检索功能")
    st.sidebar.markdown("- 📊 性能仪表盘")
    st.sidebar.markdown("- 🔍 混合检索优化")
    st.sidebar.markdown("- 🧠 上下文优化算法")

def render_feature_cards():
    """渲染功能卡片"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="border: 1px solid #ddd; padding: 20px; border-radius: 10px; height: 200px;">
        <h3>💬 智能对话练习</h3>
        <p>与AI外教进行自然英语对话，支持文字和语音模式。根据不同场景和难度级别进行针对性练习。</p>
        <button style="background-color: #4CAF50; color: white; border: none; padding: 8px 15px; text-align: center; text-decoration: none; display: inline-block; font-size: 14px; border-radius: 5px; cursor: pointer;">开始对话</button>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="border: 1px solid #ddd; padding: 20px; border-radius: 10px; margin-top: 20px; height: 200px;">
        <h3>📚 知识库管理</h3>
        <p>创建和管理个性化知识库，上传和组织文档资源，为检索系统提供专业知识支持。</p>
        <button style="background-color: #2196F3; color: white; border: none; padding: 8px 15px; text-align: center; text-decoration: none; display: inline-block; font-size: 14px; border-radius: 5px; cursor: pointer;">管理知识库</button>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="border: 1px solid #ddd; padding: 20px; border-radius: 10px; height: 200px;">
        <h3>🔍 增强型检索系统</h3>
        <p>采用最新的混合检索技术，结合语义理解和关键词匹配，智能优化上下文和结果多样性，提供精准查询结果。</p>
        <button style="background-color: #FF9800; color: white; border: none; padding: 8px 15px; text-align: center; text-decoration: none; display: inline-block; font-size: 14px; border-radius: 5px; cursor: pointer;">体验增强检索</button>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="border: 1px solid #ddd; padding: 20px; border-radius: 10px; margin-top: 20px; height: 200px;">
        <h3>📊 性能分析仪表盘</h3>
        <p>直观展示不同检索策略的性能指标，比较各种优化方法的效果，帮助您了解系统如何工作。</p>
        <button style="background-color: #9C27B0; color: white; border: none; padding: 8px 15px; text-align: center; text-decoration: none; display: inline-block; font-size: 14px; border-radius: 5px; cursor: pointer;">查看性能分析</button>
        </div>
        """, unsafe_allow_html=True)

def render_retrieval_overview():
    """渲染检索系统概览"""
    st.header("检索系统技术概览")
    
    # 创建示例数据
    search_types = ["向量检索", "关键词检索", "混合检索", "高级混合检索"]
    metrics = ["精确度", "召回率", "F1分数", "延迟(ms)"]
    
    # 生成样本数据
    np.random.seed(42)
    data = {
        "向量检索": [0.82, 0.76, 0.79, 85],
        "关键词检索": [0.78, 0.72, 0.75, 45],
        "混合检索": [0.88, 0.83, 0.85, 95],
        "高级混合检索": [0.92, 0.89, 0.90, 110]
    }
    
    # 创建DataFrame
    df = pd.DataFrame(data, index=metrics)
    
    # 转换为长格式用于Altair
    df_long = df.reset_index().melt(id_vars=['index'], var_name='检索类型', value_name='值')
    
    # 创建性能指标图表
    performance_chart = alt.Chart(df_long[df_long['index'] != '延迟(ms)']).mark_bar().encode(
        x=alt.X('检索类型:N', title='检索类型'),
        y=alt.Y('值:Q', title='得分', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('检索类型:N', scale=alt.Scale(scheme='category10')),
        column=alt.Column('index:N', title=None)
    ).properties(
        width=150,
        height=200,
        title='检索性能指标对比'
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )
    
    # 创建延迟图表
    latency_data = df_long[df_long['index'] == '延迟(ms)']
    latency_chart = alt.Chart(latency_data).mark_bar().encode(
        x=alt.X('检索类型:N', title='检索类型'),
        y=alt.Y('值:Q', title='延迟(ms)'),
        color=alt.Color('检索类型:N', scale=alt.Scale(scheme='category10'))
    ).properties(
        width=600,
        height=200,
        title='检索延迟对比'
    )
    
    # 显示图表
    st.altair_chart(performance_chart, use_container_width=True)
    st.altair_chart(latency_chart, use_container_width=True)
    
    st.markdown("""
    👆 **图表说明**：以上图表展示了不同检索方法的性能对比。高级混合检索通过结合向量和关键词检索并应用多种优化策略，
    在精确度、召回率和F1分数方面表现最佳，但延迟略高。您可以在性能仪表盘中查看更详细的分析。
    """)

def render_usage_stats():
    """渲染使用统计信息"""
    st.header("系统使用统计")
    
    # 创建示例数据
    dates = pd.date_range(end=pd.Timestamp.now(), periods=7).strftime('%m-%d')
    
    # 查询和对话数据
    queries = [142, 156, 178, 165, 189, 210, 231]
    chats = [89, 95, 110, 102, 115, 127, 138]
    
    # 创建DataFrame
    df = pd.DataFrame({
        '日期': dates,
        '知识检索': queries,
        '对话交互': chats
    })
    
    # 转换为长格式用于Altair
    df_long = df.melt(id_vars=['日期'], var_name='类型', value_name='次数')
    
    # 创建图表
    chart = alt.Chart(df_long).mark_line(point=True).encode(
        x=alt.X('日期:N', title='日期'),
        y=alt.Y('次数:Q', title='使用次数'),
        color=alt.Color('类型:N', scale=alt.Scale(scheme='set1')),
        tooltip=['日期', '类型', '次数']
    ).properties(
        width=600,
        height=300,
        title='过去7天使用统计'
    )
    
    st.altair_chart(chart, use_container_width=True)

def render_retrieval_features():
    """渲染检索功能特点"""
    st.header("检索系统核心功能")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 智能混合检索
        - 🧠 **向量语义检索**：理解查询的语义含义
        - 🔤 **关键词匹配**：捕捉重要术语和专业词汇
        - ⚖️ **自适应权重调整**：根据查询类型自动调整检索权重
        """)
        
        st.markdown("""
        ### 上下文优化
        - 📝 **上下文窗口调整**：智能选择最相关的文本片段
        - 🔄 **连贯性处理**：保证文本片段之间的逻辑连贯
        - 📊 **相关性排序**：根据与查询的相关性对结果排序
        """)
    
    with col2:
        st.markdown("""
        ### 结果多样性
        - 🌈 **多样性算法**：避免冗余结果，确保信息覆盖面
        - 📑 **来源多样化**：从不同文档获取信息
        - 🔄 **观点平衡**：提供多角度的信息视角
        """)
        
        st.markdown("""
        ### 高级分析功能
        - 🔍 **意图识别**：理解用户查询的目的
        - 📌 **实体提取**：识别查询中的关键实体
        - 📈 **性能分析**：提供检索过程的详细性能指标
        """)

def main():
    """主函数"""
    # 使用自定义CSS设置页面样式
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 渲染欢迎区域
    render_welcome_section()
    
    # 渲染功能卡片
    render_feature_cards()
    
    # 在不同标签中显示不同内容
    tabs = st.tabs(["检索系统概览", "技术特点", "使用统计"])
    
    with tabs[0]:
        render_retrieval_overview()
    
    with tabs[1]:
        render_retrieval_features()
    
    with tabs[2]:
        render_usage_stats()
    
    # 添加页脚
    st.markdown("---")
    st.markdown("© 2023-2024 ChatMaster | 智能英语外教系统 | 版本 2.1.0")

if __name__ == "__main__":
    main()
