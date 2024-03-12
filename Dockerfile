FROM python:3.8

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install uvicorn fastapi numpy scikit-learn==1.2.2 joblib pydantic 

CMD uvicorn mlapi:app --reload --port=8000 --host=0.0.0.0

