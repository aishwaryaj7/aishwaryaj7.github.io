# Projects Portfolio 🚀

Welcome to my collection of **production-grade AI/ML projects**. Each project demonstrates end-to-end implementation from data processing to deployment, showcasing modern development practices and real-world applications.

## 🎯 **Project Philosophy**

All projects in this portfolio are built with:

- **Modern Tech Stack**: Latest versions of frameworks and tools
- **Comprehensive Testing**: Thorough validation and quality assurance
- **Detailed Documentation**: Complete setup guides and usage examples
- **Real-World Applications**: Solving actual business problems

---

## 🤖 **Featured Projects**

<div class="project-grid">
  <div class="project-card">
    <div class="project-header">
      <div class="project-icon">⚡</div>
      <h3 class="project-title">Energy Price Forecasting</h3>
    </div>
    <p class="project-description">
      End-to-end MLOps pipeline for electricity price forecasting across European energy markets. Features multi-horizon predictions, ensemble methods, and comprehensive model comparison with MLflow integration.
    </p>
    <div class="project-tech">
      <span class="tech-tag">XGBoost</span>
      <span class="tech-tag">FastAPI</span>
      <span class="tech-tag">Streamlit</span>
      <span class="tech-tag">GCP</span>
      <span class="tech-tag">ENTSO-E API</span>
      <span class="tech-tag">Time Series</span>
      <span class="tech-tag">MLflow</span>
    </div>

    <div class="project-links">
      <a href="energy-price-forecasting/" class="project-link">
        <i class="fas fa-external-link-alt"></i>
        Explore Project
      </a>
      <a href="https://youtu.be/ZK0IV5H3RXo" class="project-link secondary" target="_blank">
        <i class="fas fa-play"></i>
        Demo Video
      </a>
    </div>
  </div>



  <div class="project-card">
    <div class="project-header">
      <div class="project-icon">🧾</div>
      <h3 class="project-title">DocTract - Document RAG Assistant</h3>
    </div>
    <p class="project-description">
      A privacy-first document processing system that transforms PDFs into conversational interfaces. Built with local AI inference, vector search, and advanced RAG capabilities for intelligent document analysis.
    </p>
    <div class="project-tech">
      <span class="tech-tag">Llama 2</span>
      <span class="tech-tag">HuggingFace</span>
      <span class="tech-tag">PostgreSQL</span>
      <span class="tech-tag">PGVector</span>
      <span class="tech-tag">Streamlit</span>
      <span class="tech-tag">PyMuPDF</span>
      <span class="tech-tag">LlamaIndex</span>
    </div>

    <div class="project-links">
      <a href="doctract/" class="project-link">
        <i class="fas fa-external-link-alt"></i>
        Explore Project
      </a>
      <a href="https://www.youtube.com/watch?v=9pInYAGtVZA" class="project-link secondary" target="_blank">
        <i class="fas fa-play"></i>
        Demo Video
      </a>
    </div>
  </div>

  <div class="project-card">
    <div class="project-header">
      <div class="project-icon">📋</div>
      <h3 class="project-title">Doc Parser - PDF Processing</h3>
    </div>

    <p class="project-description">
      Smart PDF extraction tool that converts documents to structured JSON/Markdown with AI-powered table querying. Features text, image, and metadata extraction with OpenAI integration.
    </p>

    <div class="project-tech">
      <span class="tech-tag">Python</span>
      <span class="tech-tag">Streamlit</span>
      <span class="tech-tag">OpenAI API</span>
      <span class="tech-tag">PyMuPDF</span>
      <span class="tech-tag">JSON</span>
      <span class="tech-tag">Markdown</span>
    </div>

    <div class="project-links">
      <a href="doc-parser/" class="project-link">
        <i class="fas fa-external-link-alt"></i>
        Explore Project
      </a>
      <a href="https://youtu.be/XFfAS7NJs3k" class="project-link secondary" target="_blank">
        <i class="fas fa-play"></i>
        Demo Video
      </a>
    </div>
  </div>

  <div class="project-card">
    <div class="project-header">
      <div class="project-icon">🎤</div>
      <h3 class="project-title">Speech2Text - Audio Transcription</h3>
    </div>

    <p class="project-description">
      Advanced speech-to-text transcriber using HuggingFace transformers and Hubert model. Supports multiple audio/video formats with timestamped segments and downloadable transcripts.
    </p>

    <div class="project-tech">
      <span class="tech-tag">HuggingFace</span>
      <span class="tech-tag">Transformers</span>
      <span class="tech-tag">Streamlit</span>
      <span class="tech-tag">Torchaudio</span>
      <span class="tech-tag">Wav2Vec2</span>
      <span class="tech-tag">Hubert</span>
    </div>

    <div class="project-links">
      <a href="speech2text/" class="project-link">
        <i class="fas fa-external-link-alt"></i>
        Explore Project
      </a>
      <a href="https://youtu.be/eSG_FsoUtRo" class="project-link secondary" target="_blank">
        <i class="fas fa-play"></i>
        Demo Video
      </a>
    </div>
  </div>
</div>

---

## 🛠️ **Technical Highlights**

### **Modern Development Practices**
- **Unified Environment**: Single `.venv` for all projects
- **Dependency Management**: UV package manager with pyproject.toml
- **Code Quality**: Type hints, linting (flake8), formatting (black)
- **Testing**: Pytest test suites with coverage reporting

### **Production Architecture**
- **API Design**: FastAPI with async patterns and automatic documentation
- **Error Handling**: Comprehensive exception handling with HTTP status codes
- **Documentation**: MkDocs with Material theme and interactive examples

### **Cloud-Ready Deployment**
- **Containerization**: Docker for consistent environments
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Health Monitoring**: API health checks and status endpoints
- **Performance**: Optimized for production workloads

---

## 📚 **Project Structure**

Each project follows a consistent, scalable structure:

```
project-name/
├── src/                    # Source code
│   └── package/           # Main package
├── tests/                 # Test suites
├── data/                  # Data files (centralized)
├── docs/                  # Project documentation
├── README.md             # Comprehensive project guide
└── pyproject.toml        # Dependencies and config
```

---


## 🚀 **Getting Started**

### **Prerequisites**
- Python 3.11+
- UV package manager
- Git

### **Quick Setup**
```bash
# Clone the repository
git clone https://github.com/aishwaryaj7/aishwaryaj7.github.io.git
cd aishwaryaj7.github.io

# Install dependencies
uv sync

# Explore projects
cd ai-portfolio/projects/
```

### **Running Projects**
Each project includes detailed setup instructions in its README file. All projects use the unified virtual environment for consistency.



---

## 🤝 **Collaboration**

These projects are designed to showcase AI/ML engineering skills. If you're interested in:

- **Technical discussions** about implementation details
- **Collaboration** on similar projects
- **Hiring** for AI/ML engineering or Data Scientist roles



---

