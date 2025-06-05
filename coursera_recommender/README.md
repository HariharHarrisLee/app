# 🎓 Personalized Course Recommender System

Using NLP, Deep Learning, and Streamlit UI on the Coursera Dataset 2021

## 🚀 Project Overview

This project builds an end-to-end course recommender system for Coursera using:
- **Deep Learning**: Neural Collaborative Filtering
- **NLP**: BERT embeddings and TF-IDF for content similarity
- **Streamlit**: Interactive web interface
- **Hybrid Models**: Combining collaborative and content-based filtering

## 📦 Dataset

Source: Coursera Courses Dataset 2021
Contains course titles, descriptions, skills, difficulty levels, ratings, and providers.

## 🛠️ Installation & Setup

1. **Clone and navigate to project:**
```bash
cd coursera_recommender
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data:**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 🎯 Usage

### Streamlit App
```bash
streamlit run streamlit_app/app.py
```

### Jupyter Notebook
```bash
jupyter notebook notebooks/recommender_notebook.ipynb
```

## 📁 Project Structure

```
coursera_recommender/
├── data/
│   └── coursera_courses.csv
├── notebooks/
│   └── recommender_notebook.ipynb
├── models/
│   └── model_weights.h5
├── streamlit_app/
│   ├── app.py
│   └── utils.py
├── slides/
│   └── presentation.pdf
├── requirements.txt
└── README.md
```

## 🧠 Models Implemented

1. **Content-Based Filtering**: BERT embeddings + cosine similarity
2. **Neural Collaborative Filtering**: Deep learning for user-course interactions
3. **Hybrid Model**: Combines both approaches for better recommendations

## 📊 Features

- **Interactive Query Interface**: Search by skills or interests
- **Personalized Recommendations**: Top-N course suggestions
- **Visualization**: Similarity scores and course analytics
- **Multi-Model Support**: Toggle between different recommendation engines

## 🎯 Key Metrics

- Precision@K, Recall@K
- NDCG for ranking quality
- RMSE for rating prediction
- Coverage and diversity metrics

## 🚀 Next Steps

- Fine-tune BERT models on course domain
- Add more sophisticated user modeling
- Implement real-time learning capabilities
- Add course content analysis
