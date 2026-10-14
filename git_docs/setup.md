# Project Setup Guide
Welcome to the backend API project.

## Prerequisites
- Python 3.10+
- Postgres database running on port 5432

## Installation Steps
1. Clone the repository.
2. Run `uv pip install -r requirements.txt`.
3. The development server runs on port 8080 by default. Do not change this port as the frontend expects it.

## Deployment
To deploy to production, trigger the GitHub Actions pipeline by tagging a release with `v1.x.x`.