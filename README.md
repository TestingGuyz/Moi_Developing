# 🌍 Moi: The Self-Learning Vision + Generative AI Assistant

**An AI Assistant with Eyes, Memory, and Autonomous Training**

---

## ✨ What is Moi?

Moi is a comprehensive AI system that combines **vision, generative AI, autonomous agents, and self-learning memory** into a single assistant. It behaves like a human learner — seeing objects, describing them, gathering knowledge, storing it, and recalling it in future interactions.

---

## 🎯 Core Features

### ✅ Vision System
- **BLIP** image-to-text model for image captioning
- **NVIDIA Vision API** (Llama 3.2 90B Vision) for enhanced descriptions
- Real-time **webcam/camera** integration
- Automatic fallback between vision models

### 🎯 Object Detection
- **YOLOv5** for real-time object detection
- Bounding boxes with confidence scores
- Color-coded detections with AI-generated descriptions
- Automatic knowledge storage

### 💾 Knowledge Database (SQLite)
- Persistent storage of detected objects
- Metadata: name, description, color, climate, types, category, user notes
- Learning history tracking
- Training data storage for secret model
- Full CRUD operations + search + export

### 🤖 Agentic AI Systems (3 Autonomous Agents)

1. **Secret Model Trainer (Agent #1)**
   - Continuously trains a local GPT-2 based model
   - Uses chat interactions and synthetic data
   - Improves without user intervention

2. **Info Gathering Agent (Agent #2)**
   - Automatically researches newly detected objects
   - Scrapes web data using Tavily
   - Extracts and stores structured information

3. **Training Supervisor (Agent #3)**
   - Manages training data quality
   - Generates synthetic Q&A pairs
   - Monitors model performance

### 🎨 Image Generation (100% FREE!)
- **Pollinations.ai** - Free, no API key needed ✅
- **A4F (AI for All)** - Free, no API key needed ✅
- **OpenRouter** - Free models available (optional API key)
- **Hugging Face** - Free tier available (optional token)

### 💬 Generative Chat
- **Groq API** with GPT-OSS 120B model
- Think mode with reasoning levels
- Custom behavior instructions
- Temperature control
- Web search integration (Tavily)

### 🧠 Memory & Learning
- Objects stored when detected
- Knowledge accumulates over time
- Chat history used for training
- User can teach the system

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam (for vision mode)
- GPU (optional, recommended)

### Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd Moi_Developing

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env and add your keys (see below)

# 4. Run the application
python app.py
```

Access at: **http://localhost:5000**

---

## 🔑 API Keys Required

### Required (Minimum)
```env
GROQ_API_KEY=xxx          # Get from console.groq.com (FREE)
TAVILY_API_KEY=xxx        # Get from tavily.com (FREE tier available)
```

### Optional (Enhanced Features)
```env
NVIDIA_API_KEY=xxx        # Better vision quality (build.nvidia.com)
HUGGINGFACE_TOKEN=xxx     # Image generation (huggingface.co)
OPENROUTER_API_KEY=xxx    # More image models (openrouter.ai)
```

**Note:** Image generation works WITHOUT any API keys! Pollinations.ai and A4F are completely free.

---

## 📖 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Moi AI Assistant                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Camera → Vision Models → Object Detection → Database  │
│           (BLIP/NVIDIA)    (YOLOv5)         (SQLite)   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Agentic AI Systems (Background)          │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌──────────┐  │  │
│  │  │Secret Model │ │Info Gatherer│ │Supervisor│  │  │
│  │  │  Trainer    │ │   Agent     │ │  Agent   │  │  │
│  │  └─────────────┘ └─────────────┘ └──────────┘  │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  Chat (Groq) + Image Gen (Free APIs) + Web Search      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 💡 Usage Examples

### 1. Start Vision Mode
```bash
# In browser or via API
POST /start-vision

# Analyze what camera sees
POST /analyze-frame
```

**Response:**
```json
{
  "caption": "A desk with a laptop and orange fruit",
  "detections": [
    {
      "class": "laptop",
      "confidence": 0.92,
      "description": "A silver laptop computer...",
      "knowledge": {...}
    },
    {
      "class": "orange", 
      "confidence": 0.87,
      "description": "A round citrus fruit...",
      "knowledge": {...}
    }
  ]
}
```

### 2. Generate Images (FREE!)
```bash
POST /generate-image
{
  "prompt": "A beautiful orange tree with ripe oranges",
  "size": "512x512",
  "backend": "auto"  # Uses free backends: pollinations or a4f
}
```

### 3. Chat with Memory
```bash
POST /chat-with-memory
{
  "message": "Tell me about oranges",
  "history": []
}
```
*Uses stored knowledge from previous detections*

### 4. Teach Moi
```bash
POST /knowledge/object
{
  "name": "orange",
  "description": "A citrus fruit",
  "color": "orange",
  "climate": "tropical/subtropical",
  "user_notes": "Rich in vitamin C"
}
```

### 5. Ask About Objects
```bash
POST /ask-about-object
{
  "object_class": "orange",
  "question": "What's the current price?",
  "location": "India"
}
```

---

## 🛠️ API Endpoints

### Chat & Interaction
- `POST /chat` - Standard chat
- `POST /chat-with-memory` - Chat with knowledge context
- `POST /improve-prompt` - AI prompt improvement

### Vision & Detection
- `POST /start-vision` - Start camera
- `POST /stop-vision` - Stop camera
- `GET /video-feed` - Video stream
- `POST /analyze-frame` - Analyze current frame
- `GET /vision-status` - Vision models status

### Knowledge Database
- `GET /knowledge/objects` - Get all objects
- `GET /knowledge/object/<name>` - Get object info
- `POST /knowledge/object` - Add/update object
- `DELETE /knowledge/object/<name>` - Delete object
- `GET /knowledge/search` - Search objects
- `GET /knowledge/statistics` - Database stats
- `GET /knowledge/export` - Export to JSON

### Image Generation (FREE!)
- `POST /generate-image` - Generate from prompt

### Agentic Systems
- `GET /agentic/status` - Agent status
- `POST /agentic/secret-model/generate` - Use secret model

### Web Search & Objects
- `POST /search-object` - Web search for objects
- `POST /ask-about-object` - Ask questions about objects

---

## 📊 How It Works

### Autonomous Learning Flow

1. **See** → Camera detects object (e.g., "orange")
2. **Describe** → Vision model generates description
3. **Detect** → YOLOv5 draws bounding box
4. **Store** → Object saved to database
5. **Research** → Info Gatherer Agent automatically searches web
6. **Learn** → Structured data extracted and stored
7. **Train** → Secret Model Trainer uses data for training
8. **Remember** → Next time object is seen, info is recalled instantly

### Image Generation Backends

| Backend | API Key? | Quality | Speed |
|---------|----------|---------|-------|
| Pollinations.ai | ❌ No | Good | Fast |
| A4F | ❌ No | Good | Fast |
| OpenRouter | ✅ Optional | Better | Medium |
| Hugging Face | ✅ Optional | Better | Slower |

**Default:** Auto-selects best available free backend

---

## 🎓 Advanced Features

### Secret Model Training
- Runs in background automatically
- Trains on chat interactions
- Generates synthetic Q&A from knowledge
- Query via `/agentic/secret-model/generate`

### Memory System
- All objects tracked with metadata
- Times seen, confidence scores
- User notes and custom properties
- Export for backup/transfer

### Think Mode
```json
{
  "think_mode": true,
  "reasoning_level": "high"  // low, medium, high
}
```
Shows step-by-step reasoning in responses

---

## 🗄️ Database Schema

### Objects Table
```
- name (unique), description, color, climate, types
- category, confidence_score, times_seen
- first_seen, last_seen, user_notes
```

### Object Properties (Key-Value)
```
- object_id, property_key, property_value
- source (user/web_search), added_date
```

### Training Data
```
- input_text, output_text, context
- quality_score, used_for_training, created_date
```

---

## 🔧 Configuration

### Environment Variables
```env
# Required
GROQ_API_KEY=xxx
TAVILY_API_KEY=xxx

# Optional (enhances capabilities)
NVIDIA_API_KEY=xxx
HUGGINGFACE_TOKEN=xxx
OPENROUTER_API_KEY=xxx

# Settings
DEFAULT_TEMPERATURE=0.7
```

### File Locations
- Database: `moi_knowledge.db`
- Exported data: `knowledge_export.json`
- Model cache: `models_cache/`
- Secret model: `models/secret_model/`

---

## 🐛 Troubleshooting

### Camera Issues
```bash
# Check camera permissions
# Try different camera index (0, 1, 2)
# Close other apps using camera
```

### Models Not Loading
```bash
# Check internet connection (first download)
# Verify disk space for cache
# Check HuggingFace token if using gated models
```

### API Errors
```bash
# Verify API keys in .env
# Check rate limits
# Review console logs
```

### Performance
```bash
# Use GPU if available (CUDA)
# Reduce camera resolution
# Disable unused agents
# Lower training frequency
```

---

## 🌟 What Makes Moi Special?

1. **Completely Free to Start** - No paid APIs required for core features
2. **Self-Learning** - Gets smarter over time automatically
3. **Vision + Memory** - Remembers what it sees
4. **Autonomous Agents** - Works in background to improve itself
5. **User Teaching** - You can correct and teach it
6. **Full Stack** - Vision, chat, image gen, all in one

---

## 📈 Future Enhancements

- [ ] Voice interaction (STT/TTS)
- [ ] Multi-camera support
- [ ] Cloud sync
- [ ] Mobile app
- [ ] Advanced fine-tuning
- [ ] Multi-language
- [ ] Video analysis

---

## 📝 Credits

**Models & APIs:**
- Vision: Salesforce BLIP, NVIDIA Llama Vision
- Detection: Ultralytics YOLOv5
- Chat: Groq GPT-OSS 120B
- Search: Tavily
- Image: Pollinations.ai, A4F, OpenRouter, Hugging Face

**Frameworks:**
- Flask, PyTorch, Transformers, OpenCV

---

## 📧 Support

For issues, create an issue in the repository.

---

**Made with ❤️ for the AI community**

**Start learning with Moi today! 🚀**