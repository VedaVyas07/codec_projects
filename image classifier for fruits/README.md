# 🍎 Fruit Image Classifier

A CNN-based image classifier that identifies **Apple, Banana, Orange, Mango,
and Grapes** from uploaded images, with a Streamlit web interface for
interactive predictions.

## Tech Stack
- Python 3.10+
- TensorFlow / Keras (CNN)
- OpenCV & Pillow (image preprocessing)
- Scikit-learn (evaluation metrics)
- Streamlit (UI)

---

## 1. VS Code Setup

1. Install [Python](https://www.python.org/downloads/) (3.10 or 3.11 recommended).
2. Install [VS Code](https://code.visualstudio.com/) and the **Python extension**
   (Ctrl+Shift+X → search "Python" → install Microsoft's extension).
3. Open the project folder in VS Code: `File > Open Folder... > fruit-classifier`.
4. Open a terminal in VS Code: `` Ctrl+` ``.
5. Create and activate a virtual environment:

   **Windows:**
```bash
   python -m venv venv
   venv\Scripts\activate
```

   **macOS/Linux:**
```bash
   python3 -m venv venv
   source venv/bin/activate
```
6. In VS Code, select the venv interpreter: `Ctrl+Shift+P` → "Python: Select
   Interpreter" → choose `./venv`.
7. Install dependencies:
```bash
   pip install -r requirements.txt
```

---

## 2. Dataset Setup

You need images sorted into 5 folders per class, inside `data/train/` and
`data/test/`: