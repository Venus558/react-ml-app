# Image Classification Web App (PyTorch + Flask + React)

This project is a full-stack image classification app using a custom PyTorch CNN, Flask API backend, and a React frontend.

> **Live Demo**: [https://brendandidier.com](https://brendandidier.com)

---

## Project Overview

- **Model**: Custom CNN built with PyTorch
- **Frontend**: Built with React + Vite
- **Backend**: Flask REST API exposing `/predict`
- **Hosting**: Deployed on a **Linux VPS using Apache**
- **Test Accuracy**: **98%** on over 4,000 images
- **Training Data**: 40,000+ labeled images from Kaggle
- **Trained on**: Personal machine with **RTX 2090 GPU**

---

## Overfitting Prevention

- Used data augmentation (`RandomResizedCrop`, `HorizontalFlip`, etc.)
- Applied `Dropout` in fully connected layers
- Used `BatchNorm` to stabilize training
- Scheduled learning rate drops at key milestones

### Learning Rate Strategy

- LR halved after:
  - Epoch 10
  - Every 7 epochs up to Epoch 32
  - Every 5 epochs after that
- Implemented directly in the training loop
- This helps with plateauing and wasted training loops

---

## Deployment

- Flask backend and React frontend both hosted on **Apache** via systemd services
- Model inference served live from the API
- React app communicates with backend using simple POST requests

---

## Repo Structure

```
├── classify_api.py               # Flask app for inference
├── evaluation.py                 # Evaluation script for testing predictions
├── requirements.txt              # Python dependencies
├── training_model.py             # Trains and saves the CNN (.pth)
├── saved_model.pth               # Saved trained CNN model
├── user_uploads.py               # Handles image upload processing

├── frontend/
│   └── app/
│       └── my-react-app/
│           ├── src/
│           │   ├── App.tsx
│           │   ├── index.css
│           │   └── ...
│           ├── public/
│           ├── vite.config.ts
│           └── package.json

├── test-data-set/
│   ├── cat/
│   ├── dog/
│   ├── other/
│   └── training_set/             # Public sample images (1 per class)

├── Uploaded_Images/              # Stores user image uploads
│   ├── cat/
│   ├── dog/
│   ├── other/
│   └── user_upload/

├── .gitignore
├── package-lock.json
```

---

### Requirements

- Python 3.8+
- Node.js (latest LTS)
- Vite + React
- CUDA-capable GPU (optional, for retraining the model)

### Step-by-Step

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python training_model.py
```

> Note: The `training_model.py` script expects a large dataset of labeled images for training. You will need to download your own images (e.g., from Kaggle or another source) and organize them into appropriate folders (e.g., `cat/`, `dog/`, `other/`) before running this script.  
>  
> A sample `training_set/` directory is provided under `test-data-set/` with one image per class for testing purposes only.

This will save the CNN as `saved_model.pth`.

Run the Flask backend:

```bash
python classify_api.py
```

Run the React frontend:

```bash
cd frontend/app/my-react-app
npm install
npm run dev
```

This will start the frontend on `http://localhost:5173`, connected to your Flask backend.

---

## Tech Stack

| Area           | Tools                         |
|----------------|-------------------------------|
| Deep Learning  | PyTorch, torchvision          |
| Backend        | Flask, Python                 |
| Frontend       | React, Vite, TypeScript       |
| Deployment     | Apache, systemd, Linux VPS    |
| Hosting        | [brendandidier.com](https://brendandidier.com) |
