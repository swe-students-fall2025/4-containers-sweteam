FROM python:3.10

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install dependencies via Pipenv using the lockfile for reproducibility
RUN pip install --upgrade pip && pip install pipenv

COPY Pipfile Pipfile.lock ./
RUN pipenv install --deploy --system --ignore-pipfile

# Copy application source
COPY . .

EXPOSE 5000

# Default entrypoint for the Flask web app
CMD ["python", "app.py"]
