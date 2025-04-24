"""
检索服务性能仪表盘
展示不同检索策略的效果对比和性能分析
"""
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import json
import random
import time
import requests
from datetime import datetime, timedelta

# API基础URL
RAG_API_BASE = "http://localhost:8000/api"

def load_sample_data():
    """加载样本数据"""
    # 这里模拟从API获取的数据
    # 实际应用中应该从真实的API获取
    
    # 模拟查询数据
    search_types = ["vector", "keyword", "hybrid"]
    query_types = ["command", "learning_question", "information_seeking", "casual_chat"]
    
    data = []
    
    for i in range(50):
        search_type = random.choice(search_types)
        query_type = random.choice(query_types)
        
        # 根据搜索类型和查询类型设置基础精度
        base_precision = 0
        if search_type == "vector" and query_type == "information_seeking":
            base_precision = 0.85
        elif search_type == "keyword" and query_type == "command":
            base_precision = 0.9
        elif search_type == "hybrid":
            base_precision = 0.8
        else:
            base_precision = 0.7
            
        # 添加随机波动
        precision = min(0.98, max(0.5, base_precision + random.uniform(-0.15, 0.15)))
        recall = min(0.98, max(0.5, base_precision + random.uniform(-0.2, 0.1)))
        latency = random.uniform(0.2, 2.0)
        
        # 上下文优化
        context_optimization = random.choice([True, False])
        if context_optimization:
            precision += random.uniform(0, 0.1)
            latency += random.uniform(0.1, 0.3)
        
        # 多样性优化    
        diversity_optimization = random.choice([True, False])
        if diversity_optimization:
            recall += random.uniform(0, 0.1)
            
        # 查询扩展
        query_expansion = random.choice([True, False])
        if query_expansion:
            recall += random.uniform(0, 0.15)
            precision -= random.uniform(0, 0.05)
            
        data.append({
            "id": i,
            "timestamp": (datetime.now() - timedelta(days=random.randint(0, 14), 
                                                  hours=random.randint(0, 23))).isoformat(),
            "query": f"模拟查询 {i}",
            "query_type": query_type,
            "search_type": search_type,
            "precision": round(min(0.99, precision), 2),
            "recall": round(min(0.99, recall), 2),
            "f1_score": round(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0, 2),
            "latency": round(latency, 2),
            "result_count": random.randint(1, 10),
            "context_optimization": context_optimization,
            "diversity_optimization": diversity_optimization,
            "query_expansion": query_expansion
        })
    
    return pd.DataFrame(data)

def render_metrics_overview(data):
    """渲染主要指标概览"""
    st.header("检索性能指标")
    
    # 计算平均指标
    avg_precision = data["precision"].mean()
    avg_recall = data["recall"].mean()
    avg_f1 = data["f1_score"].mean()
    avg_latency = data["latency"].mean()
    
    # 显示指标卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="平均精确度", value=f"{avg_precision:.2f}")
    
    with col2:
        st.metric(label="平均召回率", value=f"{avg_recall:.2f}")
    
    with col3:
        st.metric(label="平均F1分数", value=f"{avg_f1:.2f}")
    
    with col4:
        st.metric(label="平均延迟(秒)", value=f"{avg_latency:.2f}")

def render_search_type_comparison(data):
    """渲染不同搜索类型的对比"""
    st.header("搜索类型对比")
    
    # 按搜索类型分组计算平均指标
    search_type_metrics = data.groupby("search_type").agg({
        "precision": "mean",
        "recall": "mean",
        "f1_score": "mean",
        "latency": "mean"
    }).reset_index()
    
    # 长格式转换以便于可视化
    search_metrics_long = pd.melt(
        search_type_metrics, 
        id_vars=["search_type"],
        value_vars=["precision", "recall", "f1_score"],
        var_name="metric",
        value_name="value"
    )
    
    # 创建聚合条形图
    chart = alt.Chart(search_metrics_long).mark_bar().encode(
        x=alt.X("search_type:N", title="搜索类型", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("value:Q", title="得分"),
        color=alt.Color("metric:N", 
                        title="指标",
                        scale=alt.Scale(
                            domain=["precision", "recall", "f1_score"],
                            range=["#5470c6", "#91cc75", "#fac858"]
                        )),
        tooltip=["search_type", "metric", "value"]
    ).properties(
        width=600,
        height=400,
        title="不同搜索类型的性能指标对比"
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # 创建延迟对比图
    latency_chart = alt.Chart(search_type_metrics).mark_bar().encode(
        x=alt.X("search_type:N", title="搜索类型", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("latency:Q", title="平均延迟(秒)"),
        color=alt.Color("search_type:N", legend=None),
        tooltip=["search_type", "latency"]
    ).properties(
        width=600,
        height=300,
        title="不同搜索类型的平均延迟"
    )
    
    st.altair_chart(latency_chart, use_container_width=True)

def render_optimization_impact(data):
    """渲染优化策略的影响"""
    st.header("优化策略的影响")
    
    # 上下文优化的影响
    context_impact = data.groupby("context_optimization").agg({
        "precision": "mean",
        "recall": "mean",
        "f1_score": "mean",
        "latency": "mean"
    }).reset_index()
    
    context_impact["context_optimization"] = context_impact["context_optimization"].map({
        True: "启用上下文优化", 
        False: "禁用上下文优化"
    })
    
    # 多样性优化的影响
    diversity_impact = data.groupby("diversity_optimization").agg({
        "precision": "mean",
        "recall": "mean",
        "f1_score": "mean"
    }).reset_index()
    
    diversity_impact["diversity_optimization"] = diversity_impact["diversity_optimization"].map({
        True: "启用多样性增强", 
        False: "禁用多样性增强"
    })
    
    # 查询扩展的影响
    expansion_impact = data.groupby("query_expansion").agg({
        "precision": "mean",
        "recall": "mean",
        "f1_score": "mean"
    }).reset_index()
    
    expansion_impact["query_expansion"] = expansion_impact["query_expansion"].map({
        True: "启用查询扩展", 
        False: "禁用查询扩展"
    })
    
    # 创建对比图表
    col1, col2 = st.columns(2)
    
    with col1:
        context_chart = alt.Chart(pd.melt(
            context_impact, 
            id_vars=["context_optimization"],
            value_vars=["precision", "recall", "f1_score"],
            var_name="metric",
            value_name="value"
        )).mark_bar().encode(
            x=alt.X("context_optimization:N", title="", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("value:Q", title="得分"),
            color=alt.Color("metric:N", title="指标"),
            column=alt.Column("metric:N", title="上下文优化影响")
        ).properties(
            width=200,
            height=200
        )
        
        st.altair_chart(context_chart, use_container_width=True)
    
    with col2:
        diversity_chart = alt.Chart(pd.melt(
            diversity_impact, 
            id_vars=["diversity_optimization"],
            value_vars=["precision", "recall", "f1_score"],
            var_name="metric",
            value_name="value"
        )).mark_bar().encode(
            x=alt.X("diversity_optimization:N", title="", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("value:Q", title="得分"),
            color=alt.Color("metric:N", title="指标"),
            column=alt.Column("metric:N", title="多样性增强影响")
        ).properties(
            width=200,
            height=200
        )
        
        st.altair_chart(diversity_chart, use_container_width=True)
    
    expansion_chart = alt.Chart(pd.melt(
        expansion_impact, 
        id_vars=["query_expansion"],
        value_vars=["precision", "recall", "f1_score"],
        var_name="metric",
        value_name="value"
    )).mark_bar().encode(
        x=alt.X("query_expansion:N", title="", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("value:Q", title="得分"),
        color=alt.Color("metric:N", title="指标"),
        column=alt.Column("metric:N", title="查询扩展影响")
    ).properties(
        width=200,
        height=200
    )
    
    st.altair_chart(expansion_chart, use_container_width=True)

def render_query_intent_analysis(data):
    """渲染查询意图分析"""
    st.header("查询意图分析")
    
    # 按查询类型和搜索类型分组
    intent_search_metrics = data.groupby(["query_type", "search_type"]).agg({
        "f1_score": "mean"
    }).reset_index()
    
    # 创建热力图
    heatmap_data = intent_search_metrics.pivot(
        index="query_type", 
        columns="search_type", 
        values="f1_score"
    ).reset_index()
    
    # 转换为长格式
    heatmap_long = pd.melt(
        heatmap_data,
        id_vars=["query_type"],
        var_name="search_type",
        value_name="f1_score"
    )
    
    # 查询类型映射
    query_type_order = ["command", "information_seeking", "learning_question", "casual_chat"]
    query_type_names = {
        "command": "指令查询",
        "information_seeking": "信息检索",
        "learning_question": "学习问题",
        "casual_chat": "日常闲聊"
    }
    
    heatmap_long["query_type_name"] = heatmap_long["query_type"].map(query_type_names)
    
    # 创建热力图
    heatmap = alt.Chart(heatmap_long).mark_rect().encode(
        x=alt.X("search_type:N", title="搜索类型", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("query_type:N", 
                title="查询意图", 
                sort=query_type_order,
                axis=alt.Axis(labelExpr="datum.value == 'command' ? '指令查询' : " +
                              "datum.value == 'information_seeking' ? '信息检索' : " +
                              "datum.value == 'learning_question' ? '学习问题' : '日常闲聊'")),
        color=alt.Color("f1_score:Q", 
                        title="F1分数",
                        scale=alt.Scale(scheme="blueorange", domain=[0.5, 1])),
        tooltip=["query_type_name", "search_type", "f1_score"]
    ).properties(
        width=600,
        height=300,
        title="不同查询意图与搜索类型的性能关系"
    )
    
    st.altair_chart(heatmap, use_container_width=True)
    
    # 创建分组柱状图
    bar_chart = alt.Chart(heatmap_long).mark_bar().encode(
        x=alt.X("query_type:N", 
                title="查询意图",
                sort=query_type_order,
                axis=alt.Axis(labelExpr="datum.value == 'command' ? '指令查询' : " +
                              "datum.value == 'information_seeking' ? '信息检索' : " +
                              "datum.value == 'learning_question' ? '学习问题' : '日常闲聊'")),
        y=alt.Y("f1_score:Q", title="F1分数"),
        color=alt.Color("search_type:N", title="搜索类型"),
        tooltip=["query_type_name", "search_type", "f1_score"]
    ).properties(
        width=600,
        height=300,
        title="不同查询意图下各搜索类型的性能"
    )
    
    st.altair_chart(bar_chart, use_container_width=True)

def render_latency_analysis(data):
    """渲染延迟分析"""
    st.header("延迟分析")
    
    # 创建结果数量与延迟的散点图
    scatter = alt.Chart(data).mark_circle(size=60).encode(
        x=alt.X("result_count:Q", title="结果数量"),
        y=alt.Y("latency:Q", title="延迟(秒)"),
        color=alt.Color("search_type:N", title="搜索类型"),
        tooltip=["query", "search_type", "result_count", "latency", "context_optimization"]
    ).properties(
        width=600,
        height=400,
        title="结果数量与延迟的关系"
    )
    
    # 添加趋势线
    trend_line = scatter.transform_regression(
        "result_count", "latency", groupby=["search_type"]
    ).mark_line()
    
    st.altair_chart(scatter + trend_line, use_container_width=True)
    
    # 各搜索类型的延迟分布
    box_plot = alt.Chart(data).mark_boxplot().encode(
        x=alt.X("search_type:N", title="搜索类型"),
        y=alt.Y("latency:Q", title="延迟(秒)"),
        color=alt.Color("search_type:N", legend=None)
    ).properties(
        width=600,
        height=300,
        title="各搜索类型的延迟分布"
    )
    
    st.altair_chart(box_plot, use_container_width=True)

def render_hybrid_search_optimization(data):
    """渲染混合搜索优化分析"""
    st.header("混合搜索优化")
    
    # 创建模拟的混合权重数据
    weights = np.linspace(0, 1, 11)
    vector_weights = []
    keyword_weights = []
    f1_scores = []
    
    base_vector_f1 = data[data["search_type"] == "vector"]["f1_score"].mean()
    base_keyword_f1 = data[data["search_type"] == "keyword"]["f1_score"].mean()
    
    for w in weights:
        vector_weights.append(w)
        keyword_weights.append(1 - w)
        
        # 模拟不同权重下的F1分数
        # 在实际应用中，这些数据应该来自真实的实验
        hybrid_f1 = w * base_vector_f1 + (1 - w) * base_keyword_f1
        # 添加混合效应
        if 0.3 <= w <= 0.8:
            hybrid_f1 += np.sin((w - 0.3) * 5) * 0.05
            
        f1_scores.append(hybrid_f1)
    
    # 创建权重优化数据框
    weights_df = pd.DataFrame({
        "vector_weight": vector_weights,
        "keyword_weight": keyword_weights,
        "f1_score": f1_scores
    })
    
    # 创建权重优化曲线
    line_chart = alt.Chart(weights_df).mark_line(point=True).encode(
        x=alt.X("vector_weight:Q", title="向量检索权重"),
        y=alt.Y("f1_score:Q", title="F1分数"),
        tooltip=["vector_weight", "keyword_weight", "f1_score"]
    ).properties(
        width=600,
        height=400,
        title="不同混合权重下的性能"
    )
    
    st.altair_chart(line_chart, use_container_width=True)
    
    # 添加权重调整器
    st.subheader("混合权重调整")
    
    col1, col2 = st.columns(2)
    
    with col1:
        vector_weight = st.slider("向量检索权重", 0.0, 1.0, 0.7, 0.1)
        keyword_weight = 1.0 - vector_weight
        st.write(f"关键词检索权重: {keyword_weight:.1f}")
    
    with col2:
        # 计算预估性能
        # 在实际应用中，这个预估应该基于真实的模型或数据
        estimated_precision = base_vector_f1 * vector_weight + base_keyword_f1 * keyword_weight
        if 0.3 <= vector_weight <= 0.8:
            estimated_precision += np.sin((vector_weight - 0.3) * 5) * 0.05
            
        st.metric(label="预估F1分数", value=f"{estimated_precision:.2f}")
        
        if st.button("应用权重设置"):
            st.success(f"已将向量检索权重设置为 {vector_weight:.1f}，关键词检索权重设置为 {keyword_weight:.1f}")

def main():
    st.title("检索服务性能仪表盘")
    
    # 加载样本数据
    data = load_sample_data()
    
    # 设置过滤期间
    st.sidebar.header("数据过滤")
    date_options = ["最近7天", "最近30天", "全部数据"]
    selected_period = st.sidebar.selectbox("选择时间范围", date_options)
    
    # 过滤数据
    if selected_period == "最近7天":
        cutoff_date = datetime.now() - timedelta(days=7)
        data = data[pd.to_datetime(data["timestamp"]) >= cutoff_date]
    elif selected_period == "最近30天":
        cutoff_date = datetime.now() - timedelta(days=30)
        data = data[pd.to_datetime(data["timestamp"]) >= cutoff_date]
    
    # 添加其他过滤选项
    search_types = ["全部"] + sorted(data["search_type"].unique().tolist())
    selected_search_type = st.sidebar.selectbox("搜索类型", search_types)
    
    if selected_search_type != "全部":
        data = data[data["search_type"] == selected_search_type]
    
    query_types = ["全部"] + sorted(data["query_type"].unique().tolist())
    selected_query_type = st.sidebar.selectbox("查询意图", query_types)
    
    if selected_query_type != "全部":
        data = data[data["query_type"] == selected_query_type]
    
    # 渲染各部分仪表盘
    render_metrics_overview(data)
    
    st.markdown("---")
    
    render_search_type_comparison(data)
    
    st.markdown("---")
    
    render_query_intent_analysis(data)
    
    st.markdown("---")
    
    render_optimization_impact(data)
    
    st.markdown("---")
    
    render_latency_analysis(data)
    
    st.markdown("---")
    
    render_hybrid_search_optimization(data)
    
    st.sidebar.markdown("---")
    st.sidebar.info("注意: 此仪表盘使用模拟数据进行展示。在实际部署中，应该连接到真实的API获取性能数据。")

if __name__ == "__main__":
    main() 