# Multimodal-RAG

## Overview
This project is a **Multi-Modal Retrieval-Augmented Generation (RAG)** system designed to help users interact with their PDF documents in a conversational manner. It extracts text, tables, and images from PDFs, processes them using OCR (Optical Character Recognition) for images, and enables a conversational interface powered by OpenAI's GPT-4. The system uses FAISS for efficient vector storage and retrieval, making it a powerful tool for querying and analyzing PDF content.

Whether you're working with research papers, reports, or any other PDF documents, this app allows you to ask questions and get answers based on the content of your files.

## How it works
![diagram-export-3-22-2025-6_32_27-PM](https://github.com/user-attachments/assets/d9d85121-b7f7-4c02-a247-d5cb0affc997)


## Features
- **Multi-Modal Processing**:
  - Extracts text, tables, and images from PDFs.
  - Uses **Tesseract OCR** to extract text from images.
  - Combines all extracted content for a comprehensive RAG pipeline.

- **Conversational Interface**:
  - Allows users to ask questions about the content of the PDFs.
  - Retrieves relevant information using FAISS and generates responses using GPT-4.   

- **Efficient Storage and Retrieval**:
  - Uses **FAISS** for vector storage and similarity search.
  - Splits text into chunks for optimal processing.   

- **Robust Error Handling**:
  - Handles unsupported or corrupted images gracefully.
  - Provides detailed logging for debugging. 

- **User-Friendly Interface**:
  - Built with **Streamlit**, providing an intuitive and interactive web interface.



## Tech Stack

- **Python**: Core programming language.

- **LangChain**: Framework for building conversational retrieval chains.

- **OpenAI API**: Powers the conversational interface using GPT-4.

- **FAISS**: Library for efficient similarity search and vector storage.

- **PyMuPDF (Fitz)**: Extracts images and text from PDFs.

- **Tesseract OCR**: Extracts text from images.

- **Pillow (PIL)**: Image processing library.

- **Streamlit**: Framework for building interactive web applications.

- **pdfplumber**: Extracts text and tables from PDFs.

- **imageio**: Handles unsupported image formats.

- **logging**: Provides detailed logs for debugging.
  

## Example Workflow
- Upload a PDF containing text, tables, and images.

- The app will:

  - Extract text and tables using pdfplumber.

  - Extract images using PyMuPDF.

  - Use Tesseract OCR to extract text from images.

- Combine all extracted content and create a FAISS vector store.

- Ask questions about the PDF content using the conversational interface.


## Logging
- The app provides detailed logging for debugging. Logs are printed to the console and include information about:

  - Extracted images and text.

  - OCR processing.

  - Vector store creation.

  - User queries and responses.


## Steps to Run the MultiModal-RAG Project on Your Local Machine:

- ### Clone the Repository
Open a terminal and run the following command to clone the repository:

```
git clone https://github.com/AmaanSyed110/Multimodal-RAG.git
```
- ### Set Up a Virtual Environment
It is recommended to use a virtual environment for managing dependencies:

```
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```
- ### Install Dependencies
Install the required packages listed in the ```requirements.txt``` file
```
pip install -r requirements.txt
```
- ### Add Your OpenAI API Key
Create a ```.env``` file in the project directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```
- ### Run the Application
Launch the Streamlit app by running the following command:
```
streamlit run app.py
```
- ### Upload PDF Documents
Use the sidebar to upload one or more PDF files.

- ### Process Documents
Click the "Process" button to extract text, tables, and images from the PDFs.

- ### Interact with the Application
Ask questions related to the PDFs, and the app will provide relevant responses based on the document content.

## Contributions
Contributions are welcome! Please fork the repository and create a pull request for any feature enhancements or bug fixes.
