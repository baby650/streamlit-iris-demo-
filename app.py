import streamlit as st
import time
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.tree import _tree, DecisionTreeClassifier
from sklearn.decomposition import PCA
import graphviz
from sklearn.datasets import load_iris
from typing import Tuple, List, Any

# -------------------------------------------------------------
# Constants & Config
# -------------------------------------------------------------
CLASS_COLORS = ["#ffe899", "#86d3ff", "#c59cff"]
GRAPH_BG_COLOR = "#111111"
PLOT_BG_COLOR = "#222222"
FONT_COLOR = "white"

# -------------------------------------------------------------
# Data & Model Logic
# -------------------------------------------------------------
@st.cache_data
def load_data() -> Tuple[Any, Any, List[str], List[str]]:
    """Irisデータセットをロードする"""
    iris = load_iris()
    return iris.data, iris.target, iris.feature_names, iris.target_names.tolist()

def train_model(X: np.ndarray, y: np.ndarray, max_depth: int) -> DecisionTreeClassifier:
    """決定木モデルを学習する"""
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X, y)
    return model

# -------------------------------------------------------------
# Visualization Functions
# -------------------------------------------------------------
def render_decision_tree_graphviz(decision_tree: DecisionTreeClassifier, feature_names: List[str], class_names: List[str]) -> graphviz.Digraph:
    """決定木をGraphvizで描画する"""
    tree_ = decision_tree.tree_
    dot = graphviz.Digraph()

    dot.attr('graph', bgcolor=GRAPH_BG_COLOR, rankdir="TB", ranksep="0.8", nodesep="0.6")
    dot.attr('node', shape='box', style="rounded,filled", color="#FFFFFF", fontcolor="#FFFFFF", 
             penwidth="2", fontname="Meiryo", fontsize="10", height="0.4")
    dot.attr('edge', color="#FFFFFF", fontcolor="#FFFFFF")

    colors = {
        'setosa': CLASS_COLORS[0],
        'versicolor': CLASS_COLORS[1],
        'virginica': CLASS_COLORS[2]
    }

    def format_value(node_id):
        raw = tree_.value[node_id][0]
        samples = tree_.n_node_samples[node_id]
        return (raw * samples).astype(int).tolist()

    def recurse(node_id):
        # 内部ノード
        if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
            feature = feature_names[tree_.feature[node_id]]
            threshold = tree_.threshold[node_id]
            label = (f"ID = {node_id}\n{feature} ≤ {threshold:.2f}\n"
                     f"gini = {tree_.impurity[node_id]:.3f}\n"
                     f"samples = {tree_.n_node_samples[node_id]}\n"
                     f"value = {format_value(node_id)}")
            dot.node(str(node_id), label, fillcolor="#333333")
            
            left = tree_.children_left[node_id]
            right = tree_.children_right[node_id]
            dot.edge(str(node_id), str(left), label="True")
            dot.edge(str(node_id), str(right), label="False")
            
            recurse(left)
            recurse(right)
        # 葉ノード
        else:
            class_id = tree_.value[node_id].argmax()
            cls = class_names[class_id]
            label = (f"ID = {node_id}\nclass = {cls}\n"
                     f"gini = {tree_.impurity[node_id]:.3f}\n"
                     f"samples = {tree_.n_node_samples[node_id]}\n"
                     f"value = {format_value(node_id)}")
            dot.node(str(node_id), label, fillcolor=colors[cls], fontcolor="#000000")

    recurse(0)
    return dot

def plot_pca_2d(model: DecisionTreeClassifier, X: np.ndarray, y: np.ndarray, 
                input_data: np.ndarray, class_names: List[str]) -> plt.Figure:
    """PCA 2次元マップを描画する"""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    input_pca = pca.transform(input_data)

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_original = pca.inverse_transform(grid_points)
    Z = model.predict(grid_original).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(GRAPH_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)
    
    # タイトル削除
    fig.suptitle("")
    ax.set_title("")

    ax.contourf(xx, yy, Z, alpha=0.25, levels=[-0.5, 0.5, 1.5, 2.5], colors=CLASS_COLORS)
    ax.contour(xx, yy, Z, levels=[0.5, 1.5], colors="white", linewidths=1, alpha=0.8)

    for i, color in enumerate(CLASS_COLORS):
        mask = (y == i)
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=class_names[i],
                   edgecolors="#000000", s=80, alpha=0.85)

    ax.scatter(input_pca[0, 0], input_pca[0, 1], c='red', marker='*', s=300,
               label='Input Data', edgecolors='white', linewidths=1.5)

    ax.set_xlabel("PC1", color=FONT_COLOR)
    ax.set_ylabel("PC2", color=FONT_COLOR)
    ax.tick_params(colors=FONT_COLOR)
    ax.grid(True, linestyle='--', alpha=0.25, color=FONT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(FONT_COLOR)

    legend = ax.legend(facecolor="#333333", edgecolor="white")
    for text in legend.get_texts():
        text.set_color("white")

    return fig

def plot_pca_3d(X: np.ndarray, y: np.ndarray, input_data: np.ndarray, class_names: List[str]) -> go.Figure:
    """PCA 3次元マップを描画する"""
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    df_pca['Species'] = [class_names[i] for i in y]
    input_pca = pca.transform(input_data)

    fig = px.scatter_3d(
        df_pca, x='PC1', y='PC2', z='PC3', color='Species',
        color_discrete_map={'setosa': CLASS_COLORS[0], 'versicolor': CLASS_COLORS[1], 'virginica': CLASS_COLORS[2]},
        opacity=0.7, title=""
    )

    fig.update_layout(
        title="",
        scene=dict(
            xaxis=dict(backgroundcolor=GRAPH_BG_COLOR, color=FONT_COLOR, gridcolor="gray"),
            yaxis=dict(backgroundcolor=GRAPH_BG_COLOR, color=FONT_COLOR, gridcolor="gray"),
            zaxis=dict(backgroundcolor=GRAPH_BG_COLOR, color=FONT_COLOR, gridcolor="gray"),
        ),
        paper_bgcolor=GRAPH_BG_COLOR,
        font=dict(color=FONT_COLOR),
        height=600
    )
    fig.update_traces(marker=dict(size=5, line=dict(width=1, color='white')))
    
    fig.add_trace(go.Scatter3d(
        x=[input_pca[0, 0]], y=[input_pca[0, 1]], z=[input_pca[0, 2]],
        mode='markers',
        marker=dict(size=15, color='red', symbol='diamond', line=dict(width=2, color='white')),
        name='Input Data'
    ))
    return fig

def plot_feature_importance(model: DecisionTreeClassifier, feature_names: List[str]) -> go.Figure:
    """特徴量重要度を描画する"""
    importance = model.feature_importances_
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importance}).sort_values("Importance", ascending=True)

    fig = px.bar(
        df_imp, x="Importance", y="Feature", orientation='h',
        title="", color="Importance", color_continuous_scale="Blues"
    )
    fig.update_layout(
        paper_bgcolor=GRAPH_BG_COLOR, plot_bgcolor=GRAPH_BG_COLOR,
        font=dict(color=FONT_COLOR), height=500, title=""
    )
    fig.update_xaxes(showgrid=True, gridcolor="gray", zerolinecolor="gray", color=FONT_COLOR)
    fig.update_yaxes(color=FONT_COLOR)
    fig.update_traces(marker=dict(line=dict(color="white", width=1)))
    return fig

# -------------------------------------------------------------
# Main Application
# -------------------------------------------------------------
def main():
    st.title("Iris分類　決定木モデル")

    # Sidebar
    st.sidebar.header("モデル設定")
    max_depth = st.sidebar.slider("木の深さ (max_depth)", min_value=1, max_value=10, value=3)
    viz_mode = st.sidebar.radio(
        "可視化モード",
        ["決定木 (Tree)", "PCA 2次元マップ", "PCA 3次元マップ", "特徴量重要度"]
    )

    # Load Data & Train
    X, y, feature_names, class_names = load_data()
    model = train_model(X, y, max_depth)

    # User Input
    sepal_length = st.number_input("がく片の長さ", 0.0, 10.0, 5.0)
    sepal_width = st.number_input("がく片の幅", 0.0, 10.0, 3.5)
    petal_length = st.number_input("花弁の長さ", 0.0, 10.0, 1.4)
    petal_width = st.number_input("花弁の幅", 0.0, 10.0, 0.2)
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Session State for Rate Limiting
    if "last_submit_time" not in st.session_state:
        st.session_state.last_submit_time = 0

    if st.button("予測"):
        now = time.time()
        if now - st.session_state.last_submit_time < 3:
            st.warning("3秒に1回だけ実行できます。")
        else:
            st.session_state.last_submit_time = now
            pred = model.predict(input_data)[0]
            st.success(f"予測結果: {class_names[pred]}")

            # Visualization Dispatch
            if viz_mode == "決定木 (Tree)":
                st.subheader("決定木フローチャート")
                graph = render_decision_tree_graphviz(model, feature_names, class_names)
                st.graphviz_chart(graph.source)

            elif viz_mode == "PCA 2次元マップ":
                st.subheader("PCA 2次元マップ")
                fig = plot_pca_2d(model, X, y, input_data, class_names)
                st.pyplot(fig)

            elif viz_mode == "PCA 3次元マップ":
                st.subheader("PCA 3次元マップ")
                fig = plot_pca_3d(X, y, input_data, class_names)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_mode == "特徴量重要度":
                st.subheader("特徴量重要度")
                fig = plot_feature_importance(model, feature_names)
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
