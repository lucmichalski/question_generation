FROM python:3.7-slim-stretch

RUN apt-get -y update && apt-get -y install gcc

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY . .

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 6011

CMD ["python", "server.py"]
