# Real-Time Multi-Agent AI-Based Phishing Detection System

## Project Status
| Agent | Status | Accuracy |
|---|---|---|
| URL Agent | ✅ Complete | 95.78% |
| Domain Intelligence Agent | 🔨 In Progress | - |
| NLP Agent | ⏳ Pending | - |
| Fusion Agent | ⏳ Pending | - |
| Backend API | ⏳ Pending | - |
| Frontend | ⏳ Pending | - |

## Setup
```bash
pip install -r requirements.txt
```

## Run URL Agent Training
```bash
cd training
python train_url_model.py
```

## Architecture
User Input → URL Agent → Domain Agent → NLP Agent → Fusion → Risk Score