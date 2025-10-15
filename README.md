# PDF Vector Tool 📄

A modern web application that converts PDF documents into vector embeddings using OpenAI's powerful text embedding models. Built with Streamlit for a beautiful, user-friendly interface.

## Features ✨

- **PDF Text Extraction**: Extract text from any PDF document
- **Smart Text Chunking**: Automatically split text into optimal chunks (200 words each)
- **OpenAI Embeddings**: Generate high-quality vector embeddings using `text-embedding-3-small`
- **Modern UI**: Beautiful, responsive web interface built with Streamlit
- **Local Processing**: All processing happens locally on your machine
- **JSON Export**: Download embeddings as structured JSON files
- **Secure**: API keys stored in environment variables

## Prerequisites 📋

- Python 3.8 or higher
- OpenAI API key (get one from [OpenAI Platform](https://platform.openai.com/api-keys))

## Installation 🚀

### Option 1: Quick Setup (Recommended)

1. **Install Python** (if not already installed):
   - Download from [python.org](https://www.python.org/downloads/)
   - Or install from Microsoft Store

2. **Run the setup script**:
   ```bash
   python setup.py
   ```

3. **Add your OpenAI API key**:
   - Edit the `.env` file
   - Replace `your_api_key_here` with your actual OpenAI API key

### Option 2: Manual Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Create `.env` file**:
   ```bash
   # Create .env file with your API key
   echo "OPENAI_API_KEY=your_actual_api_key_here" > .env
   ```

## Usage 🎯

### Start the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### How to Use

1. **Upload PDF**: Use the file uploader in the sidebar to select your PDF
2. **Process**: The app will automatically extract text, chunk it, and generate embeddings
3. **Download**: Click the download button to save embeddings as JSON
4. **View**: Expand the chunks to see the extracted text and corresponding embeddings

### Command Line Usage

You can also use the tool programmatically:

```python
from main import process_pdf

# Process a PDF and get embeddings
json_path = process_pdf("path/to/your/document.pdf")
print(f"Embeddings saved to: {json_path}")
```

## Project Structure 📁

```
pdf_vector_tool/
├── app.py              # Streamlit web application
├── main.py             # Core PDF processing logic
├── utils.py            # Utility functions (text extraction, embeddings)
├── requirements.txt    # Python dependencies
├── setup.py           # Setup script
├── README.md          # This file
├── .env               # Environment variables (create this)
├── data/              # Uploaded PDF files
├── embeddings_json/   # Generated embedding files
└── chroma_db/         # Vector database (if using ChromaDB)
```

## API Key Security 🔐

**Important**: Never commit your API keys to version control!

- API keys are stored in the `.env` file
- The `.env` file is automatically ignored by git
- Always use environment variables for sensitive data

## Dependencies 📦

- **PyMuPDF**: PDF text extraction
- **OpenAI**: Text embeddings generation
- **Streamlit**: Web interface
- **python-dotenv**: Environment variable management
- **ChromaDB**: Vector database (optional)

## Troubleshooting 🔧

### Common Issues

1. **Python not found**: Install Python and add it to your PATH
2. **API key error**: Make sure your OpenAI API key is correct in the `.env` file
3. **Import errors**: Run `pip install -r requirements.txt`
4. **PDF extraction fails**: Ensure the PDF is not password-protected or corrupted

### Getting Help

- Check the `app.log` file for detailed error messages
- Ensure all dependencies are installed correctly
- Verify your OpenAI API key has sufficient credits

## License 📄

This project is open source and available under the MIT License.

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Happy PDF Processing! 🎉** 