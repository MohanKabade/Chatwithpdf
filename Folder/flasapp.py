from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
import uuid
from chatwithpdf import (
    load_and_chunk_pdfs,
    get_gemini_embeddings,
    store_in_pinecone,
    run_rag_pipeline
)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------- Routes --------------------

@app.route('/')
def home():
    return render_template('chatwithpdf.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    files = request.files.getlist('file')
    pdf_paths = []
    os.makedirs('pdfs', exist_ok=True)
    namespace = str(uuid.uuid4())

    for file in files:
        pdf_path = os.path.join("pdfs", file.filename)
        file.save(pdf_path)
        pdf_paths.append(pdf_path)

    chunks = load_and_chunk_pdfs(pdf_paths)
    texts = [doc.page_content for doc in chunks]
    embeddings = get_gemini_embeddings(texts, max_workers=8)
    store_in_pinecone(chunks, embeddings, namespace=namespace, index_name="llm-chatbot", batch_size=100)

    return jsonify({'message': 'Files uploaded and processed successfully', 'namespace': namespace})

@app.route('/ask_pdf', methods=['POST'])
def ask_pdf():
    data = request.get_json()
    query = data.get('query')
    namespace = data.get('namespace')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    if not namespace:
        return jsonify({'error': 'No namespace provided'}), 400

    response = run_rag_pipeline(query, namespace=namespace, index_name="llm-chatbot")
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)
