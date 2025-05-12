"""
AutoML and SHAP Analysis Assistant Agent
使用agno和streamlit构建的AutoML和SHAP分析智能体
"""
import streamlit as st

from agno.agent import Agent
from agno.models.google import Gemini

from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
import shap

import pandas as pd
import numpy as np
import os
import tempfile
from typing import Dict, Any, Tuple
import pickle
import traceback

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 自动机器学习函数
def run_auto_ml(data: dict, 
                target: str, 
                problem_type: str = 'binary', 
                test_size: float = 0.2,
                random_state: int = 42,
                time_limit: int = 240) -> Dict[str, Any]:
    """
    使用AutoGluon进行自动机器学习建模
    
    参数:
        data: 输入数据集，pandas DataFrame格式
        target: 目标变量名称
        problem_type: 问题类型，可选值为'binary'(二分类)、'multiclass'(多分类)或'regression'(回归)
        test_size: 测试集比例
        random_state: 随机种子
        
    返回:
        包含模型性能、最佳模型和特征重要性的字典
    """
    # 指定模型保存路径
    save_path = f'agModels-{target}'
    
    # 检查是否已有训练好的模型
    if os.path.exists(save_path):
        print(f"发现已有训练模型: {save_path}，直接加载")
        try:
            predictor = TabularPredictor.load(save_path)
            print(f"成功加载已有模型，目标变量: {predictor.label}")
            
            # 评估模型性能
            # 划分训练集和测试集
            train_data, test_data = train_test_split(
                data, 
                test_size=test_size, 
                random_state=random_state, 
                stratify=data[target] if problem_type != 'regression' else None
            )
            
            leaderboard = predictor.leaderboard(test_data)
            performance = predictor.evaluate(test_data)
            best_model = predictor.model_best
            
            # 返回结果
            results = {
                "performance_metrics": performance,
                "best_model": best_model,
                "model_path": save_path,
                "leaderboard": leaderboard,
                "loaded_from_cache": True
            }
            
            return results
        except Exception as e:
            print(f"加载已有模型失败: {str(e)}，将重新训练")
    
    # 划分训练集和测试集
    train_data, test_data = train_test_split(
        data, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=data[target] if problem_type != 'regression' else None
    )
    
    print(f"\n训练集大小: {train_data.shape[0]}行")
    print(f"测试集大小: {test_data.shape[0]}行")
    
    # 初始化预测器
    predictor = TabularPredictor(
        label=target,
        problem_type=problem_type,
        path=save_path
    ).fit(
        train_data,  # 使用训练集进行训练
        save_space=True,  # 只保存最优模型
        presets='best_quality',  # 使用最佳质量预设
        time_limit=time_limit,  # 训练时间限制（秒）
    )
    
    # 评估模型性能
    leaderboard = predictor.leaderboard(test_data)  # 在测试集上生成排行榜
    print("\n模型排行榜 (测试集):")
    print(leaderboard)
    
    # 获取性能指标
    performance = predictor.evaluate(test_data)  # 在测试集上评估性能
    print("\n模型性能指标 (测试集):")
    print(performance)
    
    # 获取最佳模型
    best_model = predictor.model_best
    print("\n最佳模型名称:", best_model)
    st.session_state.model_path = save_path
    print(f"预测器保存路径: {st.session_state.model_path}")

    # 返回结果
    results = {
        "performance_metrics": performance,
        "best_model": best_model if 'best_model' in locals() else None,
        "model_path": save_path,
        "leaderboard": leaderboard,
        "loaded_from_cache": False
    }
    
    return results

# SHAP分析函数
def run_shap_analysis(data: dict, 
                      target: str = None, 
                      model_path: str = None,
                      problem_type: str = 'binary',
                      max_display: int = 10,
                      sample_size: int = 20) -> Dict[str, Any]:
    """
    对训练好的AutoGluon模型进行SHAP可解释性分析
    
    参数:
        data: 输入数据集，pandas DataFrame格式
        target: 目标变量名称
        model_path: 模型保存路径
        problem_type: 问题类型，可选值为'binary'(二分类)、'multiclass'(多分类)或'regression'(回归)
        max_display: 显示的最大特征数量
        sample_size: 用于SHAP分析的样本数量，较大的数据集可以减少样本数量以提高性能
        
    返回:
        包含SHAP分析结果的字典
    """
    # data = st.session_state.data

    print(f"加载模型: {model_path}")
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    # 加载训练好的模型
    predictor = TabularPredictor.load(model_path)
    print(f"已加载模型，目标变量: {predictor.label}")
    
    # 准备特征数据
    X = data.drop(columns=[target])
    
    # 如果数据集较大，随机抽样以提高性能
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
        print(f"数据集较大，随机抽样 {sample_size} 条记录进行SHAP分析")
    else:
        X_sample = X
        print(f"使用全部 {len(X)} 条记录进行SHAP分析")
    
    # 选择最佳模型
    best_model = predictor.model_best
    
    print(f"使用模型 '{best_model}' 进行SHAP分析")

    # 计算SHAP values
    try:
        # 使用KernelExplainer作为备选（适用于任何模型）
        # 创建一个预测函数
        def model_predict(X):
            return predictor.predict_proba(pd.DataFrame(X, columns=X_sample.columns))
        
        # 使用背景数据集
        background = shap.sample(X_sample, 50)  # 使用50个样本作为背景
        explainer = shap.KernelExplainer(model_predict, background)
        shap_values = explainer.shap_values(X_sample)
        shap_values_file_path = "./shap-values/shap_values.pkl"
        # 确保目标文件夹存在
        os.makedirs("./shap-values", exist_ok=True)
        
        # 保存 shap_values 到文件
        with open(shap_values_file_path, "wb") as f:
            pickle.dump(shap_values, f)
            print(f"已保存SHAP值到文件: {shap_values_file_path}")
        
        # 处理多分类情况
        if problem_type == 'multiclass':
            if isinstance(shap_values, list): # KernelExplainer for multiclass returns a list of arrays
                shap_values_list = shap_values
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3: # Other explainers might return a 3D array
                # shap_values is (n_classes, n_samples, n_features) or (n_samples, n_features, n_classes)
                # We need a list of (n_samples, n_features) arrays, one for each class
                if shap_values.shape[0] == len(X_sample) : # (n_samples, n_features, n_classes)
                    shap_values_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
                elif shap_values.shape[1] == len(X_sample): # (n_classes, n_samples, n_features)
                     shap_values_list = [shap_values[i, :, :] for i in range(shap_values.shape[0])]
                else:
                    raise ValueError(f"无法处理的SHAP值形状: {shap_values.shape}")
            else:
                raise ValueError(f"多分类场景下无法处理的SHAP值类型或形状: {type(shap_values)}")
            multi_class = True
        else: # Binary or regression
            shap_values_list = [shap_values] # Treat as a list with one element for consistency
            multi_class = False
            
    except Exception as e:
        raise Exception(f"SHAP分析失败: {str(e)}")
    
    # 生成SHAP可视化
    # 条形摘要图
    # For multiclass, shap.summary_plot can take a list of shap_values arrays
    # For binary/regression, shap_values_list will contain a single array
    plt.figure(figsize=(10, 8))
    summary_plot_path = "./figures/shap_summary.png" # 定义路径变量
    if multi_class:
        shap.summary_plot(shap_values_list, X_sample, plot_type="bar", max_display=max_display, show=False, class_names=predictor.class_labels)
    else:
        shap.summary_plot(shap_values_list[0], X_sample, plot_type="bar", max_display=max_display, show=False)
    plt.title("特征重要性摘要图")
    plt.tight_layout()
    plt.savefig(summary_plot_path) # 使用路径变量保存
    plt.close()
    print(f"已保存特征重要性摘要图: {summary_plot_path}")

    # 生成SHAP依赖图（针对最重要的特征）
    all_top_features = set()
    dependence_plots_by_class = {}  # 添加这行来存储每个类别的依赖图信息

    for class_idx, shap_values_for_class in enumerate(shap_values_list):
        class_name = predictor.class_labels[class_idx] if multi_class and predictor.class_labels and class_idx < len(predictor.class_labels) else f"class_{class_idx}"
        print(f"为类别 '{class_name}' 生成依赖图...")
        
        # 初始化当前类别的依赖图列表
        dependence_plots_by_class[class_name] = []

        # 获取当前类别的特征重要性排序
        if shap_values_for_class.ndim == 1: # Workaround for single output regression from KernelExplainer
            feature_importance_for_class = np.abs(shap_values_for_class)
        else:
            feature_importance_for_class = np.abs(shap_values_for_class).mean(0)
        
        indices = np.argsort(feature_importance_for_class)
        
        # 获取最重要的特征 (top 3 for each class)
        num_top_features_to_plot = min(3, len(X_sample.columns))
        top_indices_for_class = indices[-num_top_features_to_plot:]
        top_features_for_class = X_sample.columns[top_indices_for_class]
        all_top_features.update(top_features_for_class)

        for feature in top_features_for_class:
            plt.figure(figsize=(10, 7))
            shap.dependence_plot(feature, shap_values_for_class, X_sample, show=False, interaction_index="auto")
            title = f"特征依赖图: {feature} (类别: {class_name})"
            plt.title(title)
            plt.tight_layout()
            
            # 保存图片
            safe_feature_name = feature.replace('/', '_').replace('\\', '_')
            img_path = f"./figures/shap_dependence_{safe_feature_name}_class_{class_name}.png"
            plt.savefig(img_path)
            plt.close()
            print(f"  已保存依赖图: {img_path}")
            
            # 将依赖图信息添加到对应类别的列表中
            dependence_plots_by_class[class_name].append({
                "feature": feature,
                "path": img_path,
                "title": title
            })

    # calculate overall feature importance based on SHAP values
    # For overall feature importance, average absolute SHAP values across all classes if multiclass
    if multi_class:
        # Stack shap values for all classes and then average
        # Ensure all arrays in shap_values_list have the same shape before stacking
        # This might need adjustment if KernelExplainer output for multiclass has varying shapes (unlikely for tabular)
        abs_shap_values_stacked = np.abs(np.stack(shap_values_list, axis=0))
        overall_feature_importance = abs_shap_values_stacked.mean(axis=(0,1)) # Mean over classes and samples
    else:
        if shap_values_list[0].ndim == 1:
             overall_feature_importance = np.abs(shap_values_list[0])
        else:
            overall_feature_importance = np.abs(shap_values_list[0]).mean(0)
    
    # 修改返回结果，添加dependence_plots_by_class
    results = {
        "feature_importance": dict(zip(X_sample.columns, overall_feature_importance)),
        "top_features_overall": list(all_top_features), 
        "multi_class": multi_class,
        "summary_plot_path": summary_plot_path,
        "dependence_plots_by_class": dependence_plots_by_class  # 添加依赖图信息到返回结果
    }
    
    return results

# 创建智能体
def create_agents(api_key: str) -> Tuple[Agent, Agent]:
    # 自动机器学习智能体
    auto_ml_agent = Agent(
        model=Gemini(
            id="gemini-2.0-flash", 
            api_key=api_key,
            system_prompt="""你是一位自动机器学习专家，擅长使用AutoGluon框架。你的任务是：
            1. 解释AutoGluon模型训练的结果
            2. 分析模型性能指标
            3. 解释最佳模型的选择原因
            4. 分析特征重要性
            5. 提供清晰的结果解释和建议
            请确保解释专业且易于理解。"""
        ),
        markdown=True
    )
    
    # SHAP分析智能体
    shap_agent = Agent(
        model=Gemini(
            id="gemini-2.0-flash",
            api_key=api_key,
            system_prompt="""你是一位模型可解释性专家，擅长使用SHAP进行模型解释，并且你能够理解和分析图像。你的任务是：
            你需要基于SHAP图提供详细、清晰、专业且易于理解的图文结合的解释。"""
        ),
        markdown=True
    )
    
    return auto_ml_agent, shap_agent

# 构建Streamlit应用
def build_app():
    """构建AutoML和可解释性分析的Streamlit应用"""
    st.title("🤖 AutoML and SHAP Analysis Assistant Agent")
    
    # 初始化会话状态
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'data_path' not in st.session_state:
        st.session_state.data_path = None
    if 'target' not in st.session_state:
        st.session_state.target = None
    if 'model_path' not in st.session_state:
        st.session_state.model_path = None
    if 'problem_type' not in st.session_state:
        st.session_state.problem_type = "binary"
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'summary_plot_base64' not in st.session_state:
        st.session_state.summary_plot_base64 = None
    if 'summary_plot_path' not in st.session_state:
        st.session_state.summary_plot_path = None
    

    # 侧边栏配置
    with st.sidebar:
        st.title("⚙ 配置")
        st.session_state.api_key = st.text_input("Gemini API密钥", 
                                               value=st.session_state.api_key,
                                               type="password")

        # 上传数据文件
        uploaded_file = st.file_uploader("上传数据文件", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                # 创建临时文件保存上传的数据
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                
                # 创建一个持久的临时文件
                temp_dir = tempfile.gettempdir()
                temp_filename = f"uploaded_data_{uploaded_file.name}"
                temp_filepath = os.path.join(temp_dir, temp_filename)
                
                # 保存上传的数据到临时文件
                with open(temp_filepath, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                # 根据文件扩展名读取数据
                if file_extension == '.csv':
                    # 尝试不同的编码方式读取CSV文件
                    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'iso-8859-1']
                    for encoding in encodings:
                        try:
                            data = pd.read_csv(temp_filepath, encoding=encoding)
                            st.success(f"成功使用{encoding}编码读取CSV文件")
                            break
                        except UnicodeDecodeError:
                            continue
                        except Exception as e:
                            st.error(f"使用{encoding}编码读取CSV文件时出错: {str(e)}")
                            continue
                    else:
                        # 如果所有编码都失败，使用二进制方式读取
                        st.warning("无法确定CSV文件的编码，尝试使用二进制方式读取")
                        data = pd.read_csv(temp_filepath, encoding='latin-1', on_bad_lines='skip')
                elif file_extension == '.xlsx':
                    data = pd.read_excel(temp_filepath)
                    st.session_state.data = pd.DataFrame(data)
                else:
                    st.error(f"不支持的文件格式: {file_extension}")
                    data = None
                
                if data is not None:
                    st.session_state.data = pd.DataFrame(data)
                    st.session_state.data_path = temp_filepath
                    st.success(f"成功加载数据: {data.shape[0]}行 x {data.shape[1]}列")
            except Exception as e:
                st.error(f"加载数据时出错: {str(e)}")
    
    # 主界面
    if st.session_state.data is not None:
        # 显示数据预览
        st.subheader("📊 数据预览")
        st.dataframe(st.session_state.data.head())
        
        # 选择目标变量
        target_options = st.session_state.data.columns.tolist()
        st.session_state.target = st.selectbox("选择目标变量", options=target_options)
        
        # 选择问题类型
        problem_type = st.radio(
            "选择问题类型",
            options=["binary", "multiclass", "regression"],
            index=0
        )
        st.session_state.problem_type = problem_type
        
        # 设置训练时间限制
        time_limit = st.slider("训练时间限制(秒)", min_value=30, max_value=600, value=120, step=30)
        
        # 创建标签页
        tab1, tab2 = st.tabs(["AutoML", "SHAP分析"])
        
        with tab1:
            st.subheader("AutoML")
            
            if st.button("点击运行"):
                if st.session_state.api_key:
                    with st.spinner("正在训练模型，这可能需要几分钟时间..."):
                        try:
                            auto_ml_resluts = run_auto_ml(
                                data=st.session_state.data,
                                target=st.session_state.target,
                                problem_type=st.session_state.problem_type,
                                time_limit=time_limit  # 添加时间限制参数
                            )
                            st.success("Autogluon执行完成！现在由Gemini进行解释...")
                            auto_ml_agent, _ = create_agents(st.session_state.api_key)
                            data = pd.DataFrame(st.session_state.data)
                            agent_input = f"""
                            请你根据自动机器学习运行的结果，提供详细的结果解释和建议。
                            请使用以下信息：
                            - 数据: {data}
                            - 目标变量: {st.session_state.target}
                            - 问题类型: {st.session_state.problem_type}
                            - 自动机器学习结果: {auto_ml_resluts}
                            """
                            response = auto_ml_agent.run(agent_input)
                            st.markdown(response.content)
                        except Exception as e:
                            st.error(f"运行自动机器学习时出错: {str(e)}")
                else:
                    st.error("请先设置Gemini API密钥")
        
        with tab2:
            st.subheader("SHAP分析")
            model_path_input = st.text_input("请输入训练好的模型路径", value=st.session_state.model_path or "")
            if model_path_input:
                st.session_state.model_path = model_path_input

            if st.button("运行SHAP分析"):
                if not st.session_state.api_key:
                    st.error("请输入Gemini API密钥才能进行SHAP结果解释。")
                elif st.session_state.model_path and st.session_state.data is not None and st.session_state.target:
                    with st.spinner("正在进行SHAP分析并请求Gemini解释..."):
                        try:
                            # 运行SHAP分析
                            shap_results = run_shap_analysis(
                                data=st.session_state.data,
                                target=st.session_state.target,
                                model_path=st.session_state.model_path,
                                problem_type=st.session_state.problem_type
                            )
                            st.success("SHAP分析完成！现在由Gemini进行解释...")
                            
                            # 创建SHAP智能体
                            _, shap_agent = create_agents(st.session_state.api_key)
                            
                            # 准备图像路径列表
                            image_paths = []
                            
                            # 添加摘要图
                            if shap_results.get("summary_plot_path"):
                                image_paths.append(shap_results["summary_plot_path"])
                            
                            # 添加所有依赖图
                            if shap_results.get("dependence_plots_by_class"):
                                for class_name, plots_data_list in shap_results["dependence_plots_by_class"].items():
                                    for plot_data in plots_data_list:
                                        if plot_data.get("path"):
                                            image_paths.append(plot_data["path"])
                           
                            # 显示分析结果
                            st.markdown("### SHAP分析结果解释")
                            st.markdown("#### SHAP特征摘要图")

                            prompt = "分析SHAP图，给出专业的，详细的，并且有见解的解释，得出的分析要有价值，而不是泛泛而谈。"
                            # 显示摘要图及其解释
                            if shap_results.get("summary_plot_path"):
                                st.image(shap_results["summary_plot_path"], caption="特征重要性摘要图")
                                print("Summary plot path:", shap_results["summary_plot_path"])
                                response = shap_agent.run(prompt, images=[{"filepath": shap_results["summary_plot_path"]}] )
                                st.markdown(response.content)

                            # 显示依赖图
                            if shap_results.get("dependence_plots_by_class"):
                                for class_name, plots_data_list in shap_results["dependence_plots_by_class"].items():
                                    st.markdown(f"#### 类别: {class_name}下的重要特征依赖图")
                                    for plot_data in plots_data_list:
                                        if plot_data.get("path"):
                                            feature_name = os.path.basename(plot_data["path"]).split("_class_")[0].replace("shap_dependence_", "")
                                            st.image(plot_data["path"], caption=f"特征依赖图: {feature_name}")
                                            response = shap_agent.run(prompt, images=[{"filepath": plot_data["path"]}] )
                                            st.markdown(response.content)
                        
                        except Exception as e:
                            st.error(f"SHAP分析或Gemini解释过程中出错: {str(e)}")
                            st.text(traceback.format_exc())
                else:
                    st.error("请先设置Gemini API密钥")
    else:
        st.info("请先上传数据文件")

if __name__ == "__main__":
    build_app()
