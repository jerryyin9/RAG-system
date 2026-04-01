#!/bin/bash

echo "======================================================"
echo "  RAG 知识库系统 - 正在启动..."
echo "======================================================"

# 1. 确保 Docker 容器在运行
echo "正在确保数据库与爬虫服务在线..."
if ! docker info &> /dev/null; then
    echo "[提示] 检测到 Docker 尚未运行，正在尝试唤醒 Docker Desktop..."
    # 仅限 Mac 系统唤醒应用
    if [ "$(uname -s)" = "Darwin" ]; then
        open -a Docker
        echo "⚠️ 请等待顶部菜单栏的 Docker 图标稳定后，重新运行 ./start.sh"
    else
        echo "请手动启动 Docker 服务。"
    fi
    exit 1
fi

docker-compose up -d

# 2. 激活环境
echo "正在启动 Web UI 界面..."
source venv/bin/activate

# 3. 跨平台自动打开浏览器
if command -v open &> /dev/null; then
    open http://localhost:8501   # Mac 系统的命令
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:8501 # Linux 系统的命令
fi

# 4. 运行 Streamlit
streamlit run app.py