@echo off
CHCP 65001
setlocal enabledelayedexpansion

echo ======================================================
<<<<<<< HEAD
echo   RAG 知识库系统 - 全自动环境部署工具
echo ======================================================

:: 0. 尝试请求管理员权限 (用于安装软件)
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [警告] 当前没有管理员权限！
    echo 如果你需要自动安装 Python 或 Docker，请关闭本窗口。
    echo 然后【右键点击 setup.bat -^> 以管理员身份运行】。
    echo.
    echo 如果你已经装好了环境，按任意键继续...
    pause >nul
)

:: 1. 检查并自动安装 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [1/5] 未检测到 Python，正在调用系统自带的 winget 自动安装 Python 3.11...
    winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
    echo.
    echo ⚠️ [重要提示] Python 安装完成！
    echo ⚠️ 因为环境变量刚刚更新，本窗口尚未生效。
    echo ⚠️ 请【关闭当前黑框】，然后再重新双击运行一次 setup.bat！
    pause
    exit /b
) else (
    echo [1/5] Python 已就绪。
)

:: 2. 检查并自动安装 Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [2/5] 未检测到 Docker，正在自动下载并安装 Docker Desktop...
    echo (这可能需要几分钟，请耐心等待，期间可能会弹窗)
    winget install -e --id Docker.DockerDesktop --accept-package-agreements --accept-source-agreements
    echo.
    echo ⚠️ [重要提示] Docker Desktop 已安装！
    echo ⚠️ 运行 Docker 通常需要重启电脑以启用 WSL2 虚拟化。
    echo ⚠️ 请按照 Docker 的弹窗提示操作，【重启电脑】后再运行本脚本。
    pause
    exit /b
) else (
    echo [2/5] Docker 已就绪。
)

:: 3. 启动 Docker 容器 (如果环境都装好了，才会走到这里)
echo [3/5] 正在通过 docker-compose 启动底层服务...
docker-compose up -d
if %errorlevel% neq 0 (
    echo [错误] 容器启动失败！请确认 Docker Desktop 软件已打开并在右下角托盘运行。
=======
echo   RAG 知识库系统 - 全自动一键部署工具
echo ======================================================

:: 1. 环境检查
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请先安装 Python 3.11。
>>>>>>> d9df69c (fix issue#3 When it is crawling the webpages, the status of log window and control buttons are not correct)
    pause
    exit /b
)

<<<<<<< HEAD
:: 4. 创建虚拟环境并安装项目依赖
if not exist "venv" (
    echo [4/5] 正在创建 Python 虚拟环境...
    python -m venv venv
)
call venv\Scripts\activate

=======
:: 2. 创建并配置虚拟环境
if not exist "venv" (
    echo [1/4] 正在创建虚拟环境 (venv)...
    python -m venv venv
)

:: 3. 动态生成并安装依赖 (整合 requirements.txt)
echo [2/4] 正在生成并安装必要的依赖包...
call venv\Scripts\activate

:: 创建临时依赖文件
>>>>>>> d9df69c (fix issue#3 When it is crawling the webpages, the status of log window and control buttons are not correct)
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

<<<<<<< HEAD
echo 正在安装 Python 依赖包...
python -m pip install --upgrade pip >nul 2>&1
pip install -r %REQ_TEMP% >nul 2>&1
del %REQ_TEMP%

:: 5. 模型就位确认
echo [5/5] 正在检查模型状态...
if exist "opt\ms-marco-MultiBERT-L-12" (
    echo ✅ 发现本地预存模型 (opt/MultiBERT)。
) else (
    echo [提示] 首次对话时系统将自动下载大模型。
)

echo.
echo ======================================================
echo 🎉 部署全部完成！
echo 以后只需双击运行 start.bat 即可启动系统。
echo ======================================================
=======
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
>>>>>>> d9df69c (fix issue#3 When it is crawling the webpages, the status of log window and control buttons are not correct)
pause