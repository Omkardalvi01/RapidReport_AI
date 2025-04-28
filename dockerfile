FROM python:3.11-alpine

RUN apk add --no-cache gcc musl-dev libffi-dev

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY .env /
COPY app.py /
COPY content_creator.py /
COPY static /static/
COPY templates /templates/
COPY start.sh /

RUN chmod +x /start.sh

EXPOSE 8000
EXPOSE 5000

CMD ["/start.sh"]