AI Image Similarity Search

This repository contains a Streamlit app that indexes images using CLIP embeddings
and FAISS for similarity search. It includes utilities to sync image folders,
upload images, edit per-image metadata (caption & keywords), and search by text
or image.

Quick start

1. Create and activate a Python environment (recommended):

   ```cmd
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Run the app:

   ```cmd
   streamlit run app.py
   ```

Notes

- Metadata is stored in `metadata-files/<model>/metadata.json`.
- If you have many images, consider excluding large backups from the repository.
- The repository may include precomputed index files under `metadata-files/`.

If you want me to push this repository to GitHub, provide the repo URL (HTTPS or
SSH) and I will add it as `origin` and attempt to push the `main` branch.