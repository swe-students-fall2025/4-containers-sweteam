![Lint-free](https://github.com/swe-students-fall2025/4-containers-sweteam/actions/workflows/lint.yml/badge.svg) ![Web-App-CI](https://github.com/swe-students-fall2025/4-containers-sweteam/actions/workflows/web-app-ci.yml/badge.svg)

# Nutribob: Containerized Drink Analysis App

Build a containerized app that uses machine learning. See [instructions](./instructions.md) for details.

## Project Overview

Nutribob is a small containerized web application that helps users analyze **bubble tea / drink images** end-to-end:

1. The user uploads a drink photo in the web app.  
2. A **machine learning client** classifies the drink type (e.g., *matcha milk tea*, *brown sugar milk tea*).  
3. The system maps the label to a recipe, calls an external **food & nutrition API**, and estimates calories and basic nutrition information.  
4. Results are stored in **MongoDB** and shown back to the user in a clean web UI.

---

## Team

- [Natalie Han](https://github.com/nateisnataliehan)
- [Jack Chen](https://github.com/a247686991)
- [Jason Li](https://github.com/jsl1114)

---

## Architecture

The whole system runs in Docker using three main services:

- `web-app`: Flask web front end (image upload + results page)
- `ml-client`: Python machine learning client that loads a Keras/TensorFlow model
- `db`: MongoDB database used to persist analysis results and any related data

---

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
