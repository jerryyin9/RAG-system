# RAG-system
如何在电脑上 Deploy部署？
步骤如下：

1. 拷贝文件：将所有文件 app.py, rag_core.py, rag_settings.py, docker-compose.yml, setup.bat, setup.sh 和 start.bat,start.sh 放在同一个项目文件夹。

2. 创建环境并装好所有包: Windows 用户双击 setup.bat运行脚本，Mac/Linux 用户双击setup.sh运行脚本。它会自动创建环境并装好所有包。如果提示需要安装Docker和Python，请提前安装好。

3. 启动应用：Windows 用户双击 start.bat运行脚本，Mac/Linux 用户双击start.sh运行脚本。


注意： Mac 上运行 .sh 的关键一步（非常重要）。在 Mac 上，出于安全机制，新创建的脚本默认是没有执行权限的。如果你直接双击，它可能会用文本编辑器打开。
Mac 用户拿到文件夹后，打开终端 (Terminal)，进行以下两步操作：

第一步：赋予执行权限（只需做一次）
在终端中进入你的项目文件夹，输入以下命令并回车：
Bash
chmod +x setup.sh start.sh

第二步：运行脚本
以后只要在终端里输入下面的命令就可以运行了：
Bash
#第一次运行
./setup.sh
#以后每天启动
./start.sh
