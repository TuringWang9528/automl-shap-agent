# AutoML and SHAP Analysis Assistant

这是一个基于agno框架的AutoML和SHAP分析助理，使用Streamlit实现用户界面。

## 功能特点

目前该应用主要实现以下两个功能：

1. **自动机器学习**：使用AutoGluon框架对用户上传的数据进行自动化机器学习建模，自动选择最佳模型。使用Gemini智能体对结果进行解释。

2. **SHAP分析**：对训练好的模型进行SHAP可解释性分析，生成特征重要性图表和依赖图，帮助用户理解模型决策过程。使用Gemini智能体对图片进行解释。


## 安装指南

### 环境要求

- Python 3.11 或更高版本

### 安装步骤

1. 克隆仓库：

```bash
git clone https://github.com/guifou/automl-shap-agent.git
cd automl-shap-agent
```

2. 创建并激活虚拟环境：

```bash
uv venv
source .venv/bin/activate # macOS/Linux
.venv\Scripts\activate # Windows
```

2. 使用uv安装依赖：

```bash
uv pip install -r requirements.txt
```


## 使用方法

1. 启动应用：

```bash
streamlit run agent.py
```

2. 在浏览器中打开显示的URL（通常是 http://localhost:8501）

3. 使用流程：
   - 上传数据文件
   - 选择目标变量（用于机器学习建模的目标列）
   - 在不同标签页中使用各种分析功能：
     - **自动机器学习**：设置训练时间并训练模型
     - **SHAP分析**：查看模型可解释性分析结果

## 注意事项

- 对于大型数据集，自动机器学习过程可能需要较长时间
- SHAP分析需要先完成自动机器学习模型训练