# 1. Base image 
FROM python:3.13-slim

# 2. Working directory 
WORKDIR /app

# 3. Dependencies 
COPY pyproject.toml uv.lock ./

RUN pip install uv
RUN uv sync --no-dev

# 4. Copy files 
COPY . .

# 5. Expose port 
EXPOSE 8000

# 6. Run command 
CMD ["uv", "run", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]