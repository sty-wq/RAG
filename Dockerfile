# 问答服务的镜像。构建：docker build -t rag-notes:dev .
#
# 镜像里只有"代码 + Python 依赖"。下面这些不放进镜像，运行时再从外面接进来：
#   - 密钥 .env.ocr      → 用 -v 挂载成只读文件
#   - 向量库 .rag_data/   → 用 -v 挂载目录，重建镜像也不会丢数据
#   - Ollama             → 跑在宿主机上（能用 GPU），通过环境变量 EMBED_BASE_URL 告诉服务它在哪

# 基础镜像：一个装好了 Python 3.12 的精简版 Linux（Debian）
FROM python:3.12-slim

# 不生成 .pyc 文件；日志直接输出，不在内存里攒着
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 容器里的工作目录，相当于 cd /app，后面的命令都在这里执行
WORKDIR /app

# 先只复制依赖清单并安装。Docker 是一层一层构建的，依赖没变时这一层会直接用缓存，
# 以后只改代码、重新构建，就不用再等漫长的 pip install
COPY requirements.txt requirements-ocr.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 再复制代码（哪些文件不复制，见 .dockerignore）
COPY . .

# 声明服务用 8000 端口。这只是说明，真正对外开放要在 docker run 时用 -p
EXPOSE 8000

# 容器启动时运行的命令。必须监听 0.0.0.0：如果只监听 127.0.0.1，容器外面访问不到
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
