![Lint-free](https://github.com/swe-students-fall2025/4-containers-sweteam/actions/workflows/lint.yml/badge.svg) ![Web-App-CI](https://github.com/swe-students-fall2025/4-containers-sweteam/actions/workflows/web-app-ci.yml/badge.svg)

# Nutribob: Containerized Drink Analysis App

Build a containerized app that uses machine learning. See [instructions](./instructions.md) for details.

## Project Overview

Nutribob is a small containerized web application that helps users analyze **bubble tea / drink images** end-to-end:

1. The user uploads a drink photo in the web app.  
2. A **machine learning client** classifies the drink type (e.g., *matcha milk tea*, *brown sugar milk tea*).  
3. The system maps the label to a recipe, calls an external **food & nutrition API**, and estimates calories and basic nutrition information.  
4. Results are stored in **MongoDB** and shown back to the user in a clean web UI.

## Team

- [Natalie Han](https://github.com/nateisnataliehan)
- [Jack Chen](https://github.com/a247686991)
- [Jason Li](https://github.com/jsl1114)

## Architecture

The whole system runs in Docker using three main services:

- `web-app`: Flask web front end (image upload + results page)
- `ml-client`: Python machine learning client that loads a Keras/TensorFlow model
- `db`: MongoDB database used to persist analysis results and any related data

## Repository Structure

```text
.
├── machine-learning-client/
│   ├── .pylintrc                    # Pylint configuration for ML client
│   ├── Pipfile                      # ML client dependencies
│   ├── Pipfile.lock

│   ├── models/                      # ML assets
│   │   ├── labels.txt               # Class labels used by the model
│   │   └── nutribob_model.h5        # Trained Keras/TensorFlow model

│   ├── src/                         # Machine learning subsystem source code
│   │   ├── __init__.py
│   │   ├── classifier.py            # Loads model + predicts image label
│   │   ├── demo_pipeline.py         # Local demo run script
│   │   ├── nutrition_api.py         # Calls external nutrition API
│   │   ├── pipeline.py              # End-to-end ML pipeline
│   │   ├── recipe_mapper.py         # Maps labels to textual recipes
│   │   ├── server.py                # Flask API server for ML client
│   │   └── train_nutribob_model.py  # Model training script (not run in production)

│   └── tests/                       # Unit tests for ML client
│       ├── __init__.py
│       └── test_analyze_drink_image.py

├── web-app/
│   ├── .env.example                 # Example environment just for web-app (if used)
│   ├── Pipfile
│   ├── Pipfile.lock
│   ├── readme.txt

│   ├── app.py                       # Main Flask application

│   ├── static/
│   │   ├── style.css
│   │   └── uploads/                 # Uploaded images from local runs
│   │       ├── 273465029009.JPEG
│   │       ├── WechatIMG64.jpg
│   │       └── image.jpg

│   ├── templates/                   # HTML templates
│   │   ├── history.html
│   │   ├── index.html               # Upload page
│   │   ├── login.html               # Optional login page
│   │   └── result.html              # Results and nutrition info

│   └── test_app.py                  # Tests for web application

├── .githooks/
│   └── commit-msg                   # Commit hook used for PR metadata checks

├── .env.example                     # Root-level example environment file (for Docker Compose)
├── Pipfile                          # Root-level Python dependencies (if needed for CI)
├── docker-compose.yml               # Defines the 3-service system: web-app, ml-client, MongoDB
├── .gitignore                       # Git ignore rules
├── LICENSE
├── README.md
```

## Quick Start
The easiest way to run the entire system (web app + ML client + MongoDB) on any platform is with Docker Compose.

### 1. Clone the repository
```bash
git clone https://github.com/swe-students-fall2025/4-containers-sweteam.git
cd 4-containers-sweteam
```

### 2. Create your .env file
At the project root:

```bash
cp .env.example .env
```

Then open .env in your editor and replace the dummy values with real ones (see Environment Variables below).

### 3. Start all services
```bash
docker compose up --build
```

Docker Compose will:

Start MongoDB (db)

Build and start the machine-learning client (ml-client)

Build and start the Flask web app (web-app)

### 4. Use the application
Open: `http://localhost:5000`

From there you can:

- Upload a drink image
- Submit it for analysis
- View the predicted label, mapped recipe, and nutrition summary
- Visit the history page to see previously analyzed drinks

## Environment Variables

Before running the system, create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then fill in the required environment variables as described below.


### **MongoDB Configuration**

These variables control access to the MongoDB instance inside Docker.

| Variable        | Description                                    | Example                                                            |
| --------------- | ---------------------------------------------- | ------------------------------------------------------------------ |
| `MONGO_DB_NAME` | Name of the MongoDB database                   | `nutribob_db`                                                      |
| `MONGO_HOST`    | MongoDB hostname (Docker Compose service name) | `db`                                                               |
| `MONGO_PORT`    | MongoDB port                                   | `27017`                                                            |
| `MONGO_USER`    | MongoDB root username                          | `sweteam`                                                          |
| `MONGO_PASS`    | MongoDB root password                          | `password`                                                         |
| `MONGO_URI`     | Full MongoDB connection URI                    | `mongodb://sweteam:password@db:27017/nutribob_db?authSource=admin` |

### **Nutrition API**

The ML client uses a third-party API to fetch calorie and nutrition estimates for each drink.

| Variable       | Description                | Example                                      |
| -------------- | -------------------------- | -------------------------------------------- |
| `FOOD_API_URL` | Nutrition API endpoint     | `https://api.calorieninjas.com/v1/nutrition` |
| `FOOD_API_KEY` | API key for authentication | `YOUR_API_KEY_HERE`                          |

> You **must** replace the API key with a real one that you get from `https://api.calorieninjas.com/v1/nutrition` for nutrition lookup to work.

### **Flask Web App Settings**

These variables configure the behavior of the Flask web application.

| Variable       | Description                 | Example                   |
| -------------- | --------------------------- | ------------------------- |
| `FLASK_ENV`    | Flask environment mode      | `development`             |
| `FLASK_DEBUG`  | Enable debug mode (1 = on)  | `1`                       |
| `SECRET_KEY`   | Secret key for sessions     | `replace_with_secure_key` |
| `WEB_APP_PORT` | Port exposed by the web app | `5000`                    |

### **Machine Learning Client**

The web app communicates with the ML client through these variables.

| Variable          | Description                  | Example                 |
| ----------------- | ---------------------------- | ----------------------- |
| `ML_SERVICE_URL`  | URL of the ML client service | `http://ml-client:8000` |
| `ML_SERVICE_PORT` | ML service port              | `8000`                  |

> **Important:**
> When running with Docker Compose, the ML service must be accessed using the service name (`ml-client`), *not* `localhost`.

### **`.env.example`**
Create a `.env` file in the project root:

```bash
MONGO_DB_NAME=nutribob_db
MONGO_HOST=db
MONGO_PORT=27017

MONGO_USER=sweteam
MONGO_PASS=password

MONGO_URI=mongodb://sweteam:password@db:27017/nutribob_db?authSource=admin

FOOD_API_URL=https://api.calorieninjas.com/v1/nutrition
FOOD_API_KEY=your_api_key

FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=replace_with_secure_key

WEB_APP_PORT=5000

ML_SERVICE_URL=http://ml-client:8000
ML_SERVICE_PORT=8000
```
