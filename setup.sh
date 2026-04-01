#!/bin/bash

echo "======================================================"
echo "  RAG 知识库系统 - 全自动环境部署工具 (Mac/Linux)"
echo "======================================================"

# 获取当前操作系统类型
OS="$(uname -s)"

# 1. 检查并自动安装 Python3
if ! command -v python3 &> /dev/null; then
    echo "[1/5] 未检测到 Python3。"
    if [ "$OS" = "Darwin" ]; then
        echo "正在尝试使用 Homebrew 自动安装 Python 3.11..."
        # 检查是否安装了 brew
        if ! command -v brew &> /dev/null; then
            echo "[错误] 你的 Mac 未安装 Homebrew。请先在终端运行以下命令安装："
            echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            exit 1
        fi
        brew install python@3.11
        echo "⚠️ [重要提示] Python 安装完成！请关闭当前终端窗口，重新打开并再次运行 ./setup.sh"
        exit 1
    else
        echo "[错误] Linux 用户请使用 apt/yum 手动安装 python3 和 python3-venv。"
        exit 1
    fi
else
    echo "[1/5] Python 已就绪。"
fi

# 2. 检查并自动安装 Docker
if ! command -v docker &> /dev/null; then
    echo "[2/5] 未检测到 Docker。"
    if [ "$OS" = "Darwin" ]; then
        echo "正在使用 Homebrew 自动安装 Docker Desktop..."
        brew install --cask docker
        echo "⚠️ [重要提示] Docker 已安装！"
        echo "⚠️ 请在你的 Mac『启动台 (Launchpad)』中找到 Docker 图标并打开它。"
        echo "⚠️ 等待 Docker 在顶部菜单栏显示运行正常后，再次运行本脚本！"
        open -a Docker
        exit 1
    else
        echo "[错误] Linux 用户请参考官方文档安装 Docker。"
        exit 1
    fi
else
    echo "[2/5] Docker 已就绪。"
fi

# 3. 启动 Docker 容器
echo "[3/5] 正在通过 docker-compose 启动底层服务..."
# 检查 Docker 引擎是否真的在后台运行
if ! docker info &> /dev/null; then
    echo "[错误] Docker 引擎未运行！请确保 Docker 软件已打开并在后台运行。"
    if [ "$OS" = "Darwin" ]; then
        open -a Docker
    fi
    exit 1
fi

docker-compose up -d
if [ $? -ne 0 ]; then
    echo "[错误] 容器启动失败！请检查 docker-compose.yml 配置。"
    exit 1
fi

# 4. 创建虚拟环境并安装依赖
if [ ! -d "venv" ]; then
    echo "[4/5] 正在创建 Python 虚拟环境..."
    python3 -m venv venv
fi

# 激活环境
source venv/bin/activate

# 动态生成临时依赖文件
cat << EOF > requirements_tmp.txt
streamlit
pandas
beautifulsoup4
cryptography
langchain
langchain-core
langchain-milvus
langchain-openai
langchain-google-genai
langgraph
pymilvus
langdetect
flashrank
requests
pypdf
EOF

echo "正在安装 Python 依赖包 (可能需要几分钟)..."
python3 -m pip install --upgrade pip >/dev/null 2>&1
pip install -r requirements_tmp.txt >/dev/null 2>&1
rm requirements_tmp.txt

# 5. 模型就位确认
echo "[5/5] 正在检查模型状态..."
if [ -d "opt/ms-marco-MultiBERT-L-12" ]; then
    echo "✅ 发现本地预存模型 (opt/MultiBERT)。"
else
    echo "[提示] 首次对话时系统将自动下载大模型。"
fi

echo ""
echo "======================================================"
echo "🎉 部署全部完成！"
echo "以后只需在终端运行 ./start.sh 即可启动系统。"
echo "======================================================"