@echo off
CHCP 65001
setlocal enabledelayedexpansion

echo ======================================================
echo   RAG 知识库系统 - 全自动一键部署工具
echo ======================================================

:: 1. 环境检查
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请先安装 Python 3.11。
    pause
    exit /b
)

:: 2. 创建并配置虚拟环境
if not exist "venv" (
    echo [1/4] 正在创建虚拟环境 (venv)...
    python -m venv venv
)

:: 3. 动态生成并安装依赖 (整合 requirements.txt)
echo [2/4] 正在生成并安装必要的依赖包...
call venv\Scripts\activate

:: 创建临时依赖文件
set REQ_TEMP=requirements_tmp.txt
(
    echo streamlit
    echo pandas
    echo beautifulsoup4
    echo cryptography
    echo langchain
    echo langchain-core
    echo langchain-milvus
    echo langchain-openai
    echo langchain-google-genai
    echo langgraph
    echo pymilvus
    echo langdetect
    echo flashrank
    echo requests
    echo pypdf
) > %REQ_TEMP%

:: 执行安装
python -m pip install --upgrade pip
pip install -r %REQ_TEMP%

:: 安装完成后删除临时文件
del %REQ_TEMP%

:: 4. 模型就位确认 (适配你的 opt 文件夹)
echo [3/4] 正在检查本地模型状态...
if exist "opt\ms-marco-MultiBERT-L-12" (
    echo ✅ 发现本地预存模型 (opt/MultiBERT)。
) else (
    echo [警告] 未发现项目内的 opt 模型文件夹！
    echo 第一次运行对话时，系统会自动联网下载模型。
)

:: 5. 提示启动
echo [4/4] 环境准备就绪！
echo ------------------------------------------------------
echo 启动说明：
echo 请运行目录下的“start.bat”启动RAG系统
echo ------------------------------------------------------
pause