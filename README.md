# Real-Time Multi-Agent AI-Based Phishing Detection System

## Project Status

| Agent                     | Status         | Accuracy |
| ------------------------- | -------------- | -------- |
| URL Agent                 | ✅ Complete    | 96.22%   |
| Domain Intelligence Agent | 🔨 In Progress | -        |
| NLP Agent                 | ⏳ Pending     | -        |
| Fusion Agent              | ⏳ Pending     | -        |
| Backend API               | ⏳ Pending     | -        |
| Frontend                  | ⏳ Pending     | -        |

## Setup

pip install -r requirements.txt

## Run URL Agent Training

cd training
python train_url_model.py

## Test URL Agent

cd agents/url_agent
python test_url_agent.py

## Architecture

User Input → URL Agent → Domain Agent → NLP Agent → Fusion → Risk Score
