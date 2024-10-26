FROM  python:3.10-slim-buster

    WORKDIR /

    COPY requirements.txt requirements.txt
    RUN pip3 install -r requirements.txt

    COPY . .

    EXPOSE 8080

    ENTRYPOINT ["python3"]
    CMD ["app.py"]