# 🚀 Aishwarya's AI/ML Portfolio

> **Production-grade AI/ML projects showcasing MLOps, RAG systems, and modern development practices**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Portfolio](https://img.shields.io/badge/Portfolio-Live-green.svg)](https://aishwaryaj7.github.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 **Overview**

This portfolio demonstrates production-ready AI/ML engineering skills through two comprehensive projects:

1. **🤖 MLOps Auto-Retraining Pipeline** - Complete ML deployment with automated retraining on GCP
2. **📄 Multi-Modal RAG Chatbot** - Advanced document processing with pyMuPDF and Weaviate

Each project showcases end-to-end implementation from data processing to production deployment, with comprehensive testing, monitoring, and documentation.

## 🏗️ **Project Structure**

```
aishwaryaj7.github.io/
├── ai-portfolio/
│   └── projects/
│       ├── mlops-auto-retrain-gcp/     # MLOps pipeline project
│       └── rag-chatbot-multimodal/     # RAG chatbot project
├── docs/                               # Portfolio documentation
│   ├── projects/                       # Project documentation
│   └── blog/                          # Technical blog posts
├── mkdocs.yml                         # Documentation site config
└── pyproject.toml                     # Main dependencies
```

## 🚀 **Featured Projects**

### 1. MLOps Auto-Retraining Pipeline
[![MLOps](https://img.shields.io/badge/MLOps-Production-green)](ai-portfolio/projects/mlops-auto-retrain-gcp/)

**Technologies**: MLflow, FastAPI, Docker, GCP, GitHub Actions

- Automated model retraining and deployment
- Comprehensive experiment tracking
- Production-ready API with monitoring
- CI/CD pipeline with automated testing

[**→ Explore Project**](ai-portfolio/projects/mlops-auto-retrain-gcp/)

### 2. Multi-Modal RAG Chatbot
[![RAG](https://img.shields.io/badge/RAG-Advanced-blue)](ai-portfolio/projects/rag-chatbot-multimodal/)

**Technologies**: LangChain, pyMuPDF, Weaviate, FastAPI, OpenTelemetry

- Multi-modal document processing (PDFs, images)
- Advanced RAG pipeline with vector search
- Production-grade API with observability
- Comprehensive testing and error handling

[**→ Explore Project**](ai-portfolio/projects/rag-chatbot-multimodal/)

## 🛠️ **Quick Start**

### Prerequisites
- Python 3.11+
- UV package manager
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/aishwaryaj7/aishwaryaj7.github.io.git
cd aishwaryaj7.github.io

# Install dependencies
uv sync

# Serve the documentation site
uv run mkdocs serve
```

### Running Individual Projects

```bash
# MLOps Project
cd ai-portfolio/projects/mlops-auto-retrain-gcp
python train.py
python serve.py

# RAG Chatbot Project
cd ai-portfolio/projects/rag-chatbot-multimodal
uv run python test_pipeline.py
uv run uvicorn src.rag_chatbot.api.app:app --reload
```

## 📚 **Documentation**

- **[Portfolio Website](https://aishwaryaj7.github.io)** - Complete portfolio with project details
- **[Technical Blog](docs/blog/)** - Deep dives into implementation details
- **[Project Documentation](docs/projects/)** - Detailed project documentation

## 🎓 **Skills Demonstrated**

### **Machine Learning & AI**
- Model development, training, and evaluation
- MLOps practices and experiment tracking
- RAG system architecture and implementation
- Multi-modal document processing

### **Software Engineering**
- Production-grade API development
- Comprehensive testing and error handling
- Clean code architecture and documentation
- CI/CD pipeline implementation

### **Cloud & DevOps**
- Docker containerization
- Cloud deployment readiness (GCP)
- Monitoring and observability
- Automated deployment pipelines

### **Modern Python Development**
- Async programming patterns
- Type hints and code quality tools
- Package management with UV
- Modern development practices

## 📈 **Why This Portfolio Stands Out**

- **🏭 Production-Ready**: Enterprise-grade code with comprehensive error handling
- **🔧 Modern Stack**: Latest versions of all frameworks and tools
- **📊 Real-World Applications**: Solves actual business problems
- **🧪 Thoroughly Tested**: Comprehensive test suites and validation
- **📚 Well-Documented**: Detailed documentation and examples
- **🚀 Scalable Architecture**: Designed for enterprise deployment

## 🤝 **Connect With Me**

- 📧 **Email**: [aishwarya.jauhari@gmail.com](mailto:aishwarya.jauhari@gmail.com)
- 💼 **LinkedIn**: [linkedin.com/in/aishwaryaj7](https://linkedin.com/in/aishwaryaj7)
- 🐙 **GitHub**: [github.com/aishwaryaj7](https://github.com/aishwaryaj7)
- 🌐 **Portfolio**: [aishwaryaj7.github.io](https://aishwaryaj7.github.io)

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ to showcase production-grade AI/ML engineering skills**
