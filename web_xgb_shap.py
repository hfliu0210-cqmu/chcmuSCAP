# 基础库导入
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import seaborn as sns
import shap
import xgboost as xgb
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from io import BytesIO
import streamlit as st
from PIL import Image

# ========== Streamlit强制首行页面配置 ==========
st.set_page_config(page_title="儿童重症社区获得性肺炎风险计算系统（SCAP Risk Calculation System - SpRCS）", layout="wide")

# 网页无GUI绘图后端
plt.switch_backend("Agg")

# ===================== 全局配置 =====================
CONFIG = {
    "dpi": 300,
    "formats": ["png"],
    # 字体
    "font_family_en": "Times New Roman",
    "font_family_zh": "Arial Unicode MS",
    "label_fontsize": 14,
    "title_fontsize": 16,
    "tick_fontsize": 14,
    # Fig9专用配置
    "fig9_max_instances": 1,
    "fig9_figsize": (22, 5),
    "fig9_color_pos": "#f8b030",
    "fig9_color_neg": "#5c2a71",
    "fig9_contribution_threshold": 0.0,
    # 配色与随机种子
    "shap_cmap": mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["#4a106b", "#fec325"]),
    "random_state": 42,
    # 模型保存路径
    "model_save_path": "fixed_xgb_pipeline.joblib",
    "explainer_save_path": "shap_explainer.joblib",
    # 你的原始训练数据文件名
    "train_data_name": "Pearson保留线性+分类(病原学)的副本(1).xlsx",
    # 特征映射：{原始模型特征名: 页面中文显示名称+单位说明}
    "feature_map": [
        ("Age", "年龄（Age，单位：月）"),
        ("Wheeze", "喘息（Wheeze，1=有，0=无）"),
        ("Underlying condition", "基础病（Underlying condition，1=有，0=无）"),
        ("Lymphocyte count", "淋巴细胞计数（Lymphocyte count，10^9/L）"),
        ("MCHC", "平均红细胞血红蛋白浓度（MCHC，g/L）"),
        ("RDW-CV", "红细胞分布宽度-CV（RDW-CV，%）"),
        ("Albumin", "白蛋白（Albumin，g/L）"),
        ("K+", "血清钾（K+，mmol/L）"),
        ("Cl-", "血清氯（Cl-，mmol/L）"),
        ("NLR", "中性-淋巴细胞比（NLR）")
    ]
}

# ===================== 全局字体初始化 =====================
en_font = CONFIG["font_family_en"]
zh_font = CONFIG["font_family_zh"]
font_fallback_list = [en_font, zh_font, 'sans-serif']
plt.rcParams['font.sans-serif'] = font_fallback_list
plt.rcParams['font.serif'] = font_fallback_list
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="ticks", rc={
    "font.sans-serif": font_fallback_list,
    "font.serif": font_fallback_list,
    "font.family": font_fallback_list,
    "axes.unicode_minus": False
})

# ===================== 工具函数：内存生成图片 =====================
def fig2pil(fig):
    buf = BytesIO()
    fig.savefig(buf, dpi=CONFIG["dpi"], format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img

# ===================== 分析类：固定模型推理 =====================
class FixedXGBInfer:
    def __init__(self):
        self.random_state = CONFIG["random_state"]
        self.cv = 5
        self.pipeline = None
        self.model = None
        self.explainer = None
        self.feature_names = None
        self.train_feature_median = {}

    def _preprocess_steps(self, scale: bool):
        steps = [
            ('imputer', SimpleImputer(strategy="median")),
            ('var', VarianceThreshold(threshold=0.0)),
        ]
        if scale:
            steps.append(('scaler', StandardScaler()))
        return steps

    # 首次运行：上传原始训练Excel训练并保存模型
    def train_and_save_model(self, train_df):
        st.info("正在使用原始SCAP数据集训练固定预测模型，仅首次运行执行一次...")
        train_df.columns = [str(col).strip() for col in train_df.columns]
        y = train_df.iloc[:, 0]
        X = train_df.iloc[:, 1:]
        X.columns = [str(col).strip() for col in X.columns]
        self.feature_names = X.columns.tolist()
        self.train_feature_median = X.median().to_dict()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        xgb_reg = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=self.random_state,
            tree_method='hist',
            n_jobs=-1
        )
        param_dist = {
            'reg__n_estimators': [300, 500, 800, 1000],
            'reg__max_depth': [3, 4, 6, 8],
            'reg__learning_rate': [0.01, 0.05, 0.1],
            'reg__subsample': [0.7, 0.85, 1.0],
            'reg__colsample_bytree': [0.7, 0.85, 1.0],
            'reg__reg_alpha': [0, 0.1, 0.5],
            'reg__reg_lambda': [1.0, 1.5, 2.0]
        }
        pipe = Pipeline(self._preprocess_steps(scale=False) + [('reg', xgb_reg)])
        rscv = RandomizedSearchCV(
            pipe, param_distributions=param_dist, n_iter=28,
            scoring='r2', cv=self.cv, n_jobs=-1, random_state=self.random_state, verbose=0
        )
        rscv.fit(X_train, y_train)
        self.pipeline = rscv.best_estimator_
        self.model = self.pipeline.named_steps['reg']

        preprocessor = Pipeline(self.pipeline.steps[:-1])
        X_train_trans = preprocessor.transform(X_train)
        X_train_proc = pd.DataFrame(X_train_trans, columns=self.feature_names)
        self.explainer = shap.TreeExplainer(self.model)

        save_data = {
            "explainer": self.explainer,
            "feature_names": self.feature_names,
            "feature_median": self.train_feature_median
        }
        joblib.dump(self.pipeline, CONFIG["model_save_path"])
        joblib.dump(save_data, CONFIG["explainer_save_path"])
        st.success(f"SCAP预测模型训练完成，已固化保存，后续仅推理不再更新参数！最优参数：{rscv.best_params_}")

    # 加载保存好的模型【兼容旧文件无feature_median】
    def load_saved_model(self):
        self.pipeline = joblib.load(CONFIG["model_save_path"])
        explainer_data = joblib.load(CONFIG["explainer_save_path"])
        self.explainer = explainer_data["explainer"]
        self.feature_names = explainer_data["feature_names"]
        self.train_feature_median = explainer_data.get("feature_median", {})
        self.model = self.pipeline.named_steps['reg']

    # 单条输入数据转为DataFrame并推理
    def infer_single_sample(self, input_dict):
        df = pd.DataFrame([input_dict])
        preprocessor = Pipeline(self.pipeline.steps[:-1])
        X_trans = preprocessor.transform(df)
        X_proc = pd.DataFrame(X_trans, columns=self.feature_names)
        shap_raw = self.explainer.shap_values(X_proc, check_additivity=False)
        if isinstance(shap_raw, list):
            shap_raw = shap_raw[-1]
        if shap_raw.ndim == 3:
            shap_raw = shap_raw[:, :, -1]
        base_val = self.explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = np.array(base_val).ravel()[-1]
        return X_proc, shap_raw, base_val

    # 绘制Fig9单样本力图
    def plot_single_force(self, X_proc, shap_all, base_val):
        feat_series = pd.Series(
            X_proc.iloc[0, :].round(3).values,
            index=self.feature_names
        )
        shap_vals = shap_all[0, :]
        fig = shap.force_plot(
            base_val,
            shap_vals,
            feat_series,
            feature_names=self.feature_names,
            matplotlib=True,
            show=False,
            figsize=CONFIG["fig9_figsize"],
            contribution_threshold=CONFIG["fig9_contribution_threshold"]
        )
        fig.set_size_inches(CONFIG["fig9_figsize"])
        ax = fig.gca()
        for t in ax.texts:
            t.set_fontfamily(CONFIG["font_family_en"])
            t.set_fontsize(max(CONFIG["tick_fontsize"] - 2, 9))

        feature_texts = [t for t in ax.texts if '=' in t.get_text()]
        feature_texts.sort(key=lambda t: t.get_position()[0])
        lines = ax.get_lines()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_span = xlim[1] - xlim[0]
        y_span = ylim[1] - ylim[0]
        x_threshold = x_span * 0.10
        step = y_span * 0.10
        levels = [0.0, -step, -step*2, -step*3, -step*4, -step*5, -step*6, -step*7]
        last_x_at_level = {l:-float("inf") for l in levels}
        min_y_attained = ylim[0]
        for txt in feature_texts:
            x, y = txt.get_position()
            choose_lvl = levels[0]
            for lvl in levels:
                if x - last_x_at_level[lvl] > x_threshold:
                    choose_lvl = lvl
                    break
            last_x_at_level[choose_lvl] = x
            new_y = y + choose_lvl
            min_y_attained = min(min_y_attained, new_y)
            if choose_lvl != 0:
                txt.set_position((x, new_y))
                for line in lines:
                    xd = line.get_xdata()
                    yd = line.get_ydata()
                    if len(xd) == 2 and abs(xd[0]-x) <1e-3 and abs(xd[1]-x)<1e-3:
                        if abs(yd[0]-y) < abs(yd[1]-y):
                            yd[0] = new_y
                        else:
                            yd[1] = new_y
                        line.set_ydata(yd)
        ax.set_ylim(bottom=min_y_attained - step * 1.5)

        pos_target = mcolors.to_rgb("#FF0051")
        neg_target = mcolors.to_rgb("#008BFB")
        pos_color = CONFIG["fig9_color_pos"]
        neg_color = CONFIG["fig9_color_neg"]
        def match_rgb(c):
            if c is None:
                return None
            try:
                crgb = mcolors.to_rgb(c)
                if sum((a-b)**2 for a,b in zip(crgb, pos_target)) <0.05:
                    return pos_color
                if sum((a-b)**2 for a,b in zip(crgb, neg_target)) <0.05:
                    return neg_color
            except:
                pass
            return None
        for obj in ax.findobj():
            if hasattr(obj, "get_color") and hasattr(obj, "set_color"):
                nc = match_rgb(obj.get_color())
                if nc: obj.set_color(nc)
            if hasattr(obj, "get_facecolor") and hasattr(obj, "set_facecolor"):
                fc = obj.get_facecolor()
                if isinstance(fc, np.ndarray) and fc.size >=3:
                    fc = fc[0] if fc.ndim ==2 else fc
                nc = match_rgb(fc)
                if nc: obj.set_facecolor(nc)
            if hasattr(obj, "get_edgecolor") and hasattr(obj, "set_edgecolor"):
                ec = obj.get_edgecolor()
                if isinstance(ec, np.ndarray) and ec.size >=3:
                    ec = ec[0] if ec.ndim ==2 else ec
                nc = match_rgb(ec)
                if nc: obj.set_edgecolor(nc)
        return fig2pil(fig)

# ===================== 网页主界面 =====================
def main():
    # 顶部左右分栏：左侧放单位图片，右侧放标题
    top_col1, top_col2 = st.columns([1, 2])
    with top_col1:
        # 读取同目录下 unit.png 图片
        if os.path.exists("unit.png"):
            logo_img = Image.open("unit.png")
            st.image(logo_img, use_column_width=True)
        else:
            st.text("请将unit.png放入程序同文件夹")
    with top_col2:
        st.title("儿童重症社区获得性肺炎风险计算系统（SCAP Risk Calculation System - SpRCS）")
    st.divider()

    infer_tool = FixedXGBInfer()
    model_exist = os.path.exists(CONFIG["model_save_path"]) and os.path.exists(CONFIG["explainer_save_path"])

    # 首次使用：上传原始训练文件生成模型
    if not model_exist:
        st.warning("检测到无预训练SCAP预测模型，请上传原始训练文件：" + CONFIG["train_data_name"])
        train_upload = st.file_uploader("上传SCAP原始训练Excel", type=["xlsx"], key="train_file")
        if train_upload is not None:
            train_df = pd.read_excel(train_upload)
            train_df.columns = [str(col).strip() for col in train_df.columns]
            st.subheader("原始SCAP训练数据预览")
            st.dataframe(train_df.head(8), use_container_width=True)
            if st.button("一键训练并固化SCAP预测模型（仅执行一次）", type="primary"):
                infer_tool.train_and_save_model(train_df)
                st.rerun()
        return

    # 已加载固定模型：手动输入表单
    infer_tool.load_saved_model()
    st.success("✅ 已加载SCAP预训练预测模型，请在下方录入患者各项临床指标（The SCAP pre-training prediction model has been loaded. Please enter the patient's indicators below）")
    st.divider()

    input_data = {}
    feat_map = CONFIG["feature_map"]
    half = len(feat_map) // 2
    col1, col2 = st.columns(2)

    with col1:
        for raw_name, show_name in feat_map[:half]:
            input_data[raw_name] = st.number_input(label=show_name, value=0.0, step=0.01)
    with col2:
        for raw_name, show_name in feat_map[half:]:
            input_data[raw_name] = st.number_input(label=show_name, value=0.0, step=0.01)

    st.divider()
    run_btn = st.button("生成SCAP预测SHAP个体力图（Generate SCAP-predicted SHAP Force Plot）", type="primary")

    if run_btn:
        with st.spinner("模型预测中，正在生成SHAP贡献力图..."):
            X_proc, shap_all, base_val = infer_tool.infer_single_sample(input_data)
            img = infer_tool.plot_single_force(X_proc, shap_all, base_val)
        st.subheader("当前患者样本SCAP预测SHAP贡献力图结果（SCAP-Predicted SHAP Force Plot for the Patient）")
        st.image(img, caption="SCAP样本SHAP力图（SCAP-Predicted SHAP Force Plot）", use_column_width=True)
        # 图片下载
        buf = BytesIO()
        img.save(buf, format="png")
        st.download_button(
            label="下载本次SCAP预测图（Download SCAP_Predicted_SHAP_Force_Plot）",
            data=buf.getvalue(),
            file_name="SCAP_Predicted_SHAP_Force_Plot.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
