import streamlit as st
import time
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.tree import _tree, DecisionTreeClassifier
from sklearn.decomposition import PCA
import graphviz
from sklearn.datasets import load_iris


# -------------------------------------------------------------
# データロード & モデル学習（動的）
# -------------------------------------------------------------
# サイドバーでハイパーパラメータ設定
st.sidebar.header("モデル設定")
max_depth = st.sidebar.slider("木の深さ (max_depth)", min_value=1, max_value=10, value=3)

# 可視化モード選択
viz_mode = st.sidebar.radio("可視化モード", ["決定木 (Tree)", "PCA (2D Map)", "PCA (3D Map)"])

iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names.tolist()

# モデル作成と学習
model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
model.fit(X, y)



# -------------------------------------------------------------
# カスタム Graphviz 生成関数（Excel風）
# -------------------------------------------------------------
def export_excel_style_tree(decision_tree, feature_names, class_names):
    tree_ = decision_tree.tree_

    dot = graphviz.Digraph()
    dot.attr('graph', bgcolor="#f7f7f7", ranksep="0.2", nodesep="0.2", size="10,10")
    dot.attr('node', shape='box', style="rounded,filled", color="#444444", penwidth="2", fontname="Meiryo", fontsize="10", height="0.4")

    colors = {
        'setosa': "#ffe899",
        'versicolor': "#86d3ff",
        'virginica': "#c59cff"
    }

    def recurse(node_id):
        if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node_id]]
            threshold = tree_.threshold[node_id]
            condition = f"{name} ≤ {threshold:.2f}"

            label = f"ID = {node_id}\n" \
                    f"{condition}\n" \
                    f"gini = {tree_.impurity[node_id]:.3f}\n" \
                    f"samples = {tree_.n_node_samples[node_id]}\n" \
                    f"value = {tree_.value[node_id].astype(int).tolist()[0]}"

            dot.node(str(node_id), label, fillcolor="#ffffff")

            left_id = tree_.children_left[node_id]
            right_id = tree_.children_right[node_id]

            dot.edge(str(node_id), str(left_id), label="True")
            dot.edge(str(node_id), str(right_id), label="False")

            recurse(left_id)
            recurse(right_id)

        else:
            class_id = tree_.value[node_id].argmax()
            cls = class_names[class_id]

            label = f"ID = {node_id}\n" \
                    f"class = {cls}\n" \
                    f"gini = {tree_.impurity[node_id]:.3f}\n" \
                    f"samples = {tree_.n_node_samples[node_id]}\n" \
                    f"value = {tree_.value[node_id].astype(int).tolist()[0]}"

            dot.node(str(node_id), label, fillcolor=colors[cls])

    recurse(0)
    return dot


# -------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------
st.title("Iris 決定木モデル 予測アプリ（Excel風決定木表示）")

sepal_length = st.number_input("がく片の長さ", 0.0, 10.0, 5.0)
sepal_width = st.number_input("がく片の幅", 0.0, 10.0, 3.5)
petal_length = st.number_input("花弁の長さ", 0.0, 10.0, 1.4)
petal_width = st.number_input("花弁の幅", 0.0, 10.0, 0.2)

if "last_submit_time" not in st.session_state:
    st.session_state.last_submit_time = 0


if st.button("予測"):
    now = time.time()
    if now - st.session_state.last_submit_time < 3:
        st.warning("3秒に1回だけ実行できます。")
    else:
        st.session_state.last_submit_time = now

        pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
        st.success(f"予測結果: {class_names[pred]}")

        # ---------------------------------------------------------
        # 可視化処理
        # ---------------------------------------------------------
        if viz_mode == "決定木 (Tree)":
            st.subheader("決定木フローチャート")
            # Excel風グラフ生成
            graph = export_excel_style_tree(model, iris.feature_names, class_names)
            st.graphviz_chart(graph.source)

        elif viz_mode == "PCA (2D Map)":
            st.subheader("PCA 2次元マップ")
            
            # PCAで2次元に圧縮
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            # 入力データも同様に変換
            input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
            input_pca = pca.transform(input_data)

            # グラフ描画
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # クラスごとの色定義（決定木と合わせる）
            colors = ["#ffe899", "#86d3ff", "#c59cff"]
            
            # 全データをプロット
            for i, color in enumerate(colors):
                mask = (y == i)
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                           c=color, label=class_names[i], edgecolors='k', s=80, alpha=0.8)

            # 入力データを★でプロット
            ax.scatter(input_pca[0, 0], input_pca[0, 1], 
                       c='red', marker='*', s=300, label='Input Data', edgecolors='white', linewidths=1.5)

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("Iris Dataset PCA Projection")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            st.pyplot(fig)

        elif viz_mode == "PCA (3D Map)":
            st.subheader("PCA 3次元マップ")
            
            # PCAで3次元に圧縮
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X)
            
            # DataFrame作成（Plotly用）
            df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
            df_pca['Species'] = [class_names[i] for i in y]
            
            # 入力データも同様に変換
            input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
            input_pca = pca.transform(input_data)

            # 3Dグラフ描画 (Plotly)
            fig = px.scatter_3d(
                df_pca, x='PC1', y='PC2', z='PC3', color='Species',
                color_discrete_map={'setosa': '#ffe899', 'versicolor': '#86d3ff', 'virginica': '#c59cff'},
                opacity=0.7,
                title="Iris Dataset PCA 3D Projection"
            )
            
            # マーカーサイズ調整
            fig.update_traces(marker=dict(size=5, line=dict(width=1, color='DarkSlateGrey')))

            # 入力データを★で追加
            fig.add_trace(go.Scatter3d(
                x=[input_pca[0, 0]],
                y=[input_pca[0, 1]],
                z=[input_pca[0, 2]],
                mode='markers',
                marker=dict(size=15, color='red', symbol='diamond', line=dict(width=2, color='white')),
                name='Input Data'
            ))

            # レイアウト調整
            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=40),
                scene=dict(
                    xaxis_title='PC1',
                    yaxis_title='PC2',
                    zaxis_title='PC3'
                ),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
