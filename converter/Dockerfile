FROM tensorflow/tensorflow:2.10.0-gpu

RUN pip install deepface

WORKDIR /app

COPY test.py .

CMD ["python3", "--version"]