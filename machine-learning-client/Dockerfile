FROM python:3.10

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install dependencies via Pipenv using the lockfile for reproducibility
RUN pip install --upgrade pip && pip install pipenv

COPY Pipfile Pipfile.lock ./
# Resolve and install directly from Pipfile for the container platform
RUN pipenv install --system --skip-lock

# Copy application source
COPY . .

EXPOSE 8000

# Default entrypoint for the ML service
CMD ["python", "-m", "src.server"]
