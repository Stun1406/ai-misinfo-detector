# 🚀 AI Misinformation Detector

An AI-powered system for detecting misinformation in text content using advanced machine learning techniques, semantic analysis, and fact-checking databases.

## 🎯 Features

- **Real-time Claim Analysis**: Analyze individual claims for misinformation
- **Batch Processing**: Upload CSV files for bulk analysis
- **Hybrid Retrieval**: Combines vector similarity search with keyword matching
- **Explainable AI**: Provides reasoning and evidence for classifications
- **Web Interface**: User-friendly frontend for testing
- **REST API**: Complete API for integration with other systems
- **Vector Database**: Uses Qdrant for semantic similarity search
- **Fact-checking Database**: PostgreSQL storage for verified sources

## 🏗️ Architecture

### Core Components

1. **Text Preprocessing**: Cleans and prepares text for analysis
2. **Embedding Generation**: Creates semantic embeddings using sentence transformers
3. **Vector Storage**: Stores embeddings in Qdrant for similarity search
4. **Classification Model**: ML-based misinformation detection
5. **Hybrid Retrieval**: Finds relevant fact-checking sources
6. **Web Interface**: React-like frontend for user interaction

### Technology Stack

- **Backend**: FastAPI, Python 3.8+
- **ML/NLP**: Transformers, Sentence-Transformers, PyTorch
- **Databases**: PostgreSQL (metadata), Qdrant (vectors)
- **Frontend**: HTML, CSS, JavaScript (Vanilla)
- **Deployment**: Docker Compose

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose
- 4GB+ RAM (for ML models)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd ai-misinformation-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings (optional for local development)
```

### 3. Start Databases

```bash
# Start PostgreSQL and Qdrant using Docker
docker-compose up -d

# Wait for services to be ready (about 30 seconds)
```

### 4. Initialize Database

```bash
# Seed with sample fact-checking data
python scripts/seed_data.py
```

### 5. Start Application

```bash
# Start the FastAPI server
python scripts/start_app.py
```

The application will be available at:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📊 API Usage

### Analyze Single Claim

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "COVID-19 vaccines contain microchips",
    "source_url": "https://example.com/article"
  }'
```

**Response:**
```json
{
  "claim_id": 1,
  "claim": "COVID-19 vaccines contain microchips",
  "classification": "False",
  "reliability": 15,
  "confidence": 85,
  "evidence": [
    "Multiple large-scale clinical trials have demonstrated that COVID-19 vaccines approved by the FDA are both safe and effective..."
  ],
  "reasoning": "Contains sensational language; No credible evidence supports this claim",
  "source_url": "https://example.com/article",
  "processing_time_ms": 1250,
  "evidence_sources": [
    {
      "title": "COVID-19 Vaccines Are Safe and Effective",
      "source_name": "CDC",
      "source_url": "https://www.cdc.gov/coronavirus/2019-ncov/vaccines/safety/safety-of-vaccines.html",
      "relevance_score": 0.89
    }
  ]
}
```

### Batch Analysis

```bash
curl -X POST "http://localhost:8000/analyze/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "claims": [
      {"text": "The earth is flat"},
      {"text": "Vaccines are safe and effective"}
    ]
  }'
```

### Upload CSV

```bash
curl -X POST "http://localhost:8000/analyze/upload" \
  -F "file=@data/sample_claims.csv"
```

## 🧪 Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=app
```

### Test with Sample Data

```bash
# Use the provided sample CSV
curl -X POST "http://localhost:8000/analyze/upload" \
  -F "file=@data/sample_claims.csv"
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` | localhost | PostgreSQL host |
| `POSTGRES_PORT` | 5432 | PostgreSQL port |
| `POSTGRES_DB` | misinformation_detector | Database name |
| `QDRANT_HOST` | localhost | Qdrant host |
| `QDRANT_PORT` | 6333 | Qdrant port |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |
| `API_PORT` | 8000 | API server port |
| `DEBUG` | True | Debug mode |

### Database Schema

#### Claims Table
- `id`: Primary key
- `text`: Claim text
- `classification`: True/False/Misleading/Unverified
- `reliability_score`: 0-100 score
- `evidence`: JSON array of evidence snippets
- `source_url`: Optional source URL
- `created_at`, `processed_at`: Timestamps

#### Fact Sources Table
- `id`: Primary key
- `title`: Source title
- `content`: Full content text
- `source_name`: Publisher (e.g., "CDC", "Reuters")
- `source_url`: URL to original source
- `topic`: Category (health, politics, science, etc.)
- `reliability_rating`: Source reliability (0.0-1.0)

## 🚀 Production Deployment

### Docker Production Setup

```bash
# Build production image
docker build -t ai-misinformation-detector .

# Run with production settings
docker run -p 8000:8000 \
  -e DEBUG=False \
  -e POSTGRES_HOST=your-postgres-host \
  -e QDRANT_HOST=your-qdrant-host \
  ai-misinformation-detector
```

### Scaling Considerations

- **Model Caching**: Models are loaded once at startup
- **Database Connections**: Connection pooling configured
- **Vector Search**: Qdrant handles concurrent searches efficiently
- **Async Processing**: FastAPI handles concurrent requests

## 🔮 Future Enhancements

### Planned Features

1. **Multi-modal Detection**: Support for images and videos
2. **Fine-tuned Models**: Custom models trained on misinformation datasets
3. **Real-time Monitoring**: Continuous web scraping and analysis
4. **Trust Networks**: Source reliability graphs
5. **Browser Extension**: Real-time webpage fact-checking
6. **Advanced NLP**: Named entity recognition and fact extraction

### Model Improvements

- Fine-tune BERT/RoBERTa on misinformation datasets
- Implement few-shot learning for new misinformation types
- Add uncertainty quantification
- Multi-language support

## 🤝 Contributing

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black app/
isort app/

# Run linting
flake8 app/
```

### Adding New Fact Sources

```python
# Use the API endpoint
import requests

response = requests.post("http://localhost:8000/fact-sources", json={
    "title": "New Fact Check Article",
    "content": "Detailed fact-checking content...",
    "source_name": "Trusted Source",
    "source_url": "https://trusted-source.com/article",
    "topic": "health"
})
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Check if Docker services are running
   docker-compose ps
   
   # Restart databases
   docker-compose restart
   ```

2. **Model Loading Issues**
   ```bash
   # Clear model cache
   rm -rf ~/.cache/huggingface/
   
   # Reinstall transformers
   pip install --upgrade transformers sentence-transformers
   ```

3. **Memory Issues**
   ```bash
   # Reduce batch size in .env
   BATCH_SIZE=16
   
   # Use CPU instead of GPU
   # Models will automatically fall back to CPU
   ```

### Logs

```bash
# View application logs
tail -f logs/app.log

# View database logs
docker-compose logs postgres

# View Qdrant logs
docker-compose logs qdrant
```

## 📊 Performance

### Benchmarks

- **Single Claim**: ~1-3 seconds
- **Batch (10 claims)**: ~8-15 seconds
- **Vector Search**: <100ms
- **Classification**: ~200-500ms per claim

### System Requirements

- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores
- **Storage**: 2GB for models, 1GB for data

---

Built with ❤️ for fighting misinformation through AI
