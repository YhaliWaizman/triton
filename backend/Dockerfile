FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . .

CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8080"]
