FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt update -y && apt install awscli -y

COPY . .

CMD ["python", "application.py"]