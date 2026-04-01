@echo off
CHCP 65001
echo ======================================================
echo   RAG 知识库系统 - 正在启动...
echo ======================================================

:: 1. 确保 Docker 容器在运行
echo 正在确保数据库服务在线...
docker-compose up -d

:: 2. 激活环境并启动 Streamlit
echo 正在启动 Web UI 界面...
<<<<<<< HEAD
call ..\venv\Scripts\activate
=======
call \venv\Scripts\activate
>>>>>>> d9df69c (fix issue#3 When it is crawling the webpages, the status of log window and control buttons are not correct)
streamlit run app.py

pause