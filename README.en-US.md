# ChatMaster - Intelligent English Tutor System

## Project Introduction

ChatMaster is a scene-based multi-turn voice dialogue system. This project aims to help users practice daily English speaking and conversation skills painlessly, improving their abilities progressively from simple to difficult levels.

## Core Features

- **Text Dialogue Mode**: Provides a smooth text-based conversation experience.
- **Oral Communication Mode**: Supports voice input and output to simulate real-life conversation scenarios.
- **Scene-Based Speaking Practice**: Provides 10 default scenes and supports uploading custom scenes.
- **Multi-turn Dialogue Management**: Maintains conversation context to provide a coherent interaction experience.
- **Customizable Difficulty Levels**: Adjusts conversation complexity based on the user's proficiency level.
- **Question Recommendation System**: Intelligently recommends relevant questions based on current conversation content.
- **Advanced RAG Retrieval**: Combines vector and keyword hybrid retrieval technologies for more accurate knowledge matching.
- **Intelligent Document Parsing**: Supports parsing of various document formats and scene extraction.

## Technical Architecture

### Frontend Interface

- Interactive Web application built with the Streamlit framework.
- Supports both text and voice interaction modes.
- Scene selection and custom upload functionality.
- Dialogue history management and session saving.

### Backend Services

- **Automatic Speech Recognition (ASR)**: Speech-to-text functionality based on FunASR.
- **Text Generation**: Dialogue generation based on Large Language Models (GPT-3.5/GPT-4).
- **Text-to-Speech (TTS)**: Text-to-speech functionality based on ChatTTS.
- **Retrieval-Augmented Generation (RAG)**:
  - Advanced vector retrieval (FAISS).
  - Hybrid retrieval strategy (BM25 + Vector Similarity).
  - Intelligent document parsing and chunking.
  - Session state management.
  - Retrieval result re-ranking.

## Core Technical Implementation

### 1. Dialogue Data Processing

- **Data Sources**: English movies and TV shows from YouTube.
- **Processing Workflow**:
  - Download videos using `youtube-dl`.
  - Speech recognition using Tencent speech packages.
  - Rule-based determination using Regular Expressions + GPT-4o.
  - Low-quality text filtering (de-duplication, special symbol removal, error correction).
  - Sensitive word and stop-word filtering.
  - BM25 keyword matching (delete if match > 70%).
  - Vector retrieval (delete if similarity > 0.8).
  - Unified JSON format output.
  - Attached scene and difficulty information.

### 2. Oral Understanding Optimization

- **Colloquial Question Processing**:
  - Fine-tuned T5-base model to handle omitted words, informal fillers, and irregular word order.
  - CrossEntropyLoss loss function, AdamW optimizer.
  - BLEU score $\ge 90$.
- **Coreference Resolution**:
  - Implemented based on `bert-base-uncased`.
  - Parameters: 110M, `max_sequence_len=512`.
- **Intent Recognition**:
  - Three major intent categories: Scene Dialogue, Daily Chit-chat, Learning-related Questions.
  - Rule-based recognition + RoBERTa-base classification model.
  - Accuracy $> 96\%$.
- **Named Entity Recognition (NER)**:
  - BERT+CRF model using BIO tagging strategy.
  - Entity recognition F1 score $> 94\%$.

### 3. Enhanced RAG Retrieval System

- **Document Processing**:
  - Multi-format document support (PDF/Word/Excel/TXT).
  - Semantic-based intelligent chunking algorithm.
  - Document metadata extraction.
- **Advanced Retrieval Engine**:
  - Hybrid retrieval strategy (Vector similarity + Keyword matching).
  - Automatic weight adjustment mechanism.
  - Context-aware retrieval.
  - ElasticSearch full-text indexing.
- **Knowledge Base Management**:
  - Multi-knowledge base support.
  - Incremental update mechanism.
  - Automatic scene extraction.
  - Bi-directional indexing (Keywords + Vectors).
- **Session Enhancement Features**:
  - Intelligent session naming.
  - Recommended question generation.
  - Dialogue history analysis.
  - Retrieval evidence tracking.

### 4. Dialogue Management

- **Split Dialogue Management**:
  - Dialogue Policy (DP).
  - Dialogue State Tracking (DST).
- **External Memory Mechanism**:
  - Implementation of long-term conversation memory using MemGPT, AttentionStore, MemLLM, etc.
- **Context Compression and Summarization**:
  - Prompt Engineering.
  - Conversational Memory.
  - Topic-based context focusing.

## Installation and Usage

### Environment Requirements

- Python 3.8+
- OpenAI API key (for GPT model calls)
- ElasticSearch 7.x (for advanced RAG retrieval, optional)

### Installation Steps

1. Clone this repository

```bash
git clone https://github.com/yourusername/ChatMaster.git
cd ChatMaster
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Configure API keys
   Create a `.env` file and add the following:

```
OPENAI_API_KEY=your_api_key_here
DASHSCOPE_API_KEY=your_dashscope_key_here  # Optional, for Chinese models
```

4. Prepare knowledge base directories

```bash
mkdir -p dataset/knowledge_base
mkdir -p dataset/uploads
```

5. Start the application

```bash
cd frontend
streamlit run app.py
```

## User Guide

1. Select the dialogue model (GPT-3.5/GPT-4) in the sidebar.
2. Select the dialogue language (English by default).
3. Select the communication mode (Text Dialogue/Oral Communication).
4. Select a preset scene or upload a custom document to create a scene.
5. Start practicing English by chatting with the AI assistant.
6. View the system's recommended related questions; click to ask quickly.
7. Use session management features to save and restore dialogue history.

## Advanced Features

- **Document Parsing**: Upload PDF, Word, Excel, etc., and the system automatically extracts scenes and knowledge points.
- **Scene Customization**: Filter scenes based on industry, difficulty, etc.
- **Voice Customization**: Select different accents and speeds for speech synthesis.
- **Custom Themes**: Professional English training for specific industries or scenarios.
- **Progress Tracking**: Record user practice history and improvement.

## 💕 Acknowledgments

- [InternLM](https://github.com/InternLM/InternLM)
- [xtuner](https://github.com/InternLM/xtuner)
- [LMDeploy](https://github.com/InternLM/LMDeploy)
- [lagent](https://github.com/InternLM/lagent)

Thanks to the Shanghai AI Laboratory for providing valuable technical guidance and powerful computing support for my project.

## 🎫 Open Source License

This project is licensed under the [Apache License 2.0](https://github.com/PeterH0323/Streamer-Sales/LICENSE). Please also comply with the licenses of the models and datasets used.

## 🔗 Citation

If this project has been helpful to you, please cite it using the following format:

```bibtex
@misc{ChatMaster,
    title={ChatMaster},
    author={wjh2001},
    url={https://github.com/wjh2001/ChatMaster},
    year={2024}
}
```
