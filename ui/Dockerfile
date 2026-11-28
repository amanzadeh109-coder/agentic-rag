FROM python:3.12-slim
WORKDIR /ui
RUN pip install --no-cache-dir streamlit==1.* httpx
COPY app.py .
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
