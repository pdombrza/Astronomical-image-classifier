FROM python:3.12-slim

WORKDIR /usr
COPY . .
RUN pip install -r requirements.txt
EXPOSE 7680
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "src/main.py"]
