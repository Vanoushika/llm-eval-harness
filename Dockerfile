FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY evalharness ./evalharness
COPY examples ./examples

ENV DATABASE_URL=sqlite:///./evalharness.db
EXPOSE 8000

CMD ["uvicorn", "evalharness.main:app", "--host", "0.0.0.0", "--port", "8000"]
