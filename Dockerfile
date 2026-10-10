# graphForge 镜像：纯 NumPy，零重依赖，极简运行环境
FROM python:3.11-slim

LABEL author="晨星" \
      description="graphForge — 纯 NumPy 世界级谱图学习与社区发现系统"

WORKDIR /app

# 仅运行时依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 默认运行端到端演示
CMD ["python", "examples/run_demo.py", "benchmark.json"]
