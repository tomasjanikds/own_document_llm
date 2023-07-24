FROM python:3.9-buster

ARG API_KEY
ENV OPENAI_API_KEY=${API_KEY}

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "main.py"]