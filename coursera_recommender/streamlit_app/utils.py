"""
Utility functions for the Course Recommender System.

This module contains data preprocessing, model initialization, and recommendation functions
for the Coursera course recommender system using NLP and deep learning.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
import streamlit as st
warnings.filterwarnings('ignore')

# Raw URL to the Coursera dataset CSV on GitHub (raw file link)
RAW_CSV_URL = 'https://raw.githubusercontent.com/HariharHarrisLee/app/main/coursera_recommender/streamlit_app/Coursera.csv'

# Download required NLTK data
for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)


def safe_re_sub(pattern: str, repl: str, string: str) -> str:
    """
    A safe wrapper around re.sub to catch regex errors.
    
    Args:
        pattern (str): Regex pattern.
        repl (str): Replacement string.
        string (str): Input string to process.
    
    Returns:
        str: Result after applying regex substitution or original string if error.
    """
    if not pattern:
        print("Warning: Empty regex pattern provided. Returning original string.")
        return string
    
    try:
        return re.sub(pattern, repl, string)
    except re.error as e:
        print(f"Regex error with pattern '{pattern}': {e}")
        # Fallback: remove non-alphabetic characters manually
        return ''.join([c for c in string if c.isalpha() or c.isspace()])


class CourseRecommender:
    """
    A comprehensive course recommendation system using multiple approaches.
    
    This class implements content-based filtering using TF-IDF and BERT embeddings,
    with capabilities for hybrid recommendations and neural collaborative filtering.
    """
    
    def __init__(self, data_path: str = RAW_CSV_URL):
        """
        Initialize the CourseRecommender with dataset.
        
        Args:
            data_path (str): Path to the CSV file containing course data.
        """
        self.data_path = data_path
        self.df = None
        self.bert_model = None
        self.tfidf_vectorizer = None
        self.bert_embeddings = None
        self.tfidf_matrix = None
        self.processed_descriptions = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """
        Load and preprocess the course dataset.
        
        Returns:
            pd.DataFrame: Cleaned and preprocessed course dataframe.
        """
        try:
            # Load data
            self.df = pd.read_csv(self.data_path)
            print(f"Loaded {len(self.df)} courses from dataset")
            
            # Clean and preprocess
            self.df = self._clean_data()
            self.df = self._encode_categorical_features()
            self.processed_descriptions = self._preprocess_text_data()
            
            print("Data preprocessing completed successfully")
            return self.df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def _clean_data(self) -> pd.DataFrame:
        """
        Clean the raw dataset by handling missing values and standardizing formats.
        
        Returns:
            pd.DataFrame: Cleaned dataframe.
        """
        # Drop rows missing essential text fields
        self.df = self.df.dropna(subset=['Course Name', 'Course Description'])
        
        # Clean and standardize text fields
        text_columns = ['Course Name', 'Course Description', 'Skills']
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()
        
        # Map difficulty levels to numeric
        if 'Difficulty Level' in self.df.columns:
            difficulty_mapping = {
                'Beginner': 1,
                'Intermediate': 2, 
                'Advanced': 3
            }
            self.df['Difficulty_Numeric'] = self.df['Difficulty Level'].map(difficulty_mapping)
        
        # Convert ratings to numeric and fill missing with median
        if 'Course Rating' in self.df.columns:
            self.df['Course Rating'] = pd.to_numeric(self.df['Course Rating'], errors='coerce')
            self.df['Course Rating'].fillna(self.df['Course Rating'].median(), inplace=True)
        
        return self.df
    
    def _encode_categorical_features(self) -> pd.DataFrame:
        """
        Encode categorical features for machine learning models.
        
        Returns:
            pd.DataFrame: Dataframe with encoded features.
        """
        # One-hot encode University/Provider
        if 'University' in self.df.columns:
            university_dummies = pd.get_dummies(self.df['University'], prefix='uni')
            self.df = pd.concat([self.df, university_dummies], axis=1)
        
        # Multi-hot encode Skills
        if 'Skills' in self.df.columns:
            all_skills = set()
            for skills_str in self.df['Skills']:
                if pd.notna(skills_str):
                    skills = [skill.strip() for skill in str(skills_str).split(',')]
                    all_skills.update(skills)
            
            # Create binary features for top skills (limit to top 20 sorted alphabetically)
            top_skills = sorted(list(all_skills))[:20]
            for skill in top_skills:
                self.df[f'skill_{skill.replace(" ", "_").lower()}'] = self.df['Skills'].str.contains(
                    re.escape(skill), case=False, na=False
                ).astype(int)
        
        return self.df
    
    def _preprocess_text_data(self) -> List[str]:
        """
        Preprocess text data for NLP models using tokenization, lemmatization, etc.
        
        Returns:
            List[str]: List of preprocessed course descriptions.
        """
        processed_texts = []
        
        for _, row in self.df.iterrows():
            # Combine title, description, and skills
            text_parts = []
            
            if pd.notna(row['Course Name']):
                text_parts.append(str(row['Course Name']))
            
            if pd.notna(row['Course Description']):
                text_parts.append(str(row['Course Description']))
                
            if pd.notna(row['Skills']):
                # Add skills with higher weight by repeating them
                skills = str(row['Skills']).replace(',', ' ')
                text_parts.extend([skills] * 2)
            
            combined_text = ' '.join(text_parts)
            processed_text = self._clean_text(combined_text)
            processed_texts.append(processed_text)
        
        return processed_texts
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess individual text strings.
        
        Args:
            text (str): Raw text to be cleaned.
            
        Returns:
            str: Cleaned and preprocessed text.
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits safely
        pattern = r'[^a-zA-Z\s]'
        text = safe_re_sub(pattern, '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    @st.cache_resource
    def initialize_models(_self) -> None:
        """
        Initialize and configure the ML models (BERT, TF-IDF).
        """
        print("Initializing models...")
        
        # Initialize BERT model
        try:
            print("Loading BERT model...")
            _self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("BERT model loaded successfully")
        except Exception as e:
            print(f"Error loading BERT model: {str(e)}")
            _self.bert_model = None
        
        # Initialize TF-IDF vectorizer
        _self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        print("Models initialized successfully")
    
    @st.cache_data
    def generate_embeddings(_self) -> None:
        """
        Generate embeddings for all courses using BERT and TF-IDF.
        """
        print("Generating embeddings...")
        
        if _self.processed_descriptions is None:
            raise ValueError("Processed descriptions not available. Please preprocess data first.")
        
        # Generate TF-IDF embeddings
        print("Generating TF-IDF embeddings...")
        _self.tfidf_matrix = _self.tfidf_vectorizer.fit_transform(_self.processed_descriptions)
        print(f"TF-IDF matrix shape: {_self.tfidf_matrix.shape}")
        
        # Generate BERT embeddings
        if _self.bert_model is not None:
            print("Generating BERT embeddings...")
            _self.bert_embeddings = _self.bert_model.encode(
                _self.processed_descriptions,
                show_progress_bar=True,
                batch_size=32,
                convert_to_numpy=True
            )
            print(f"BERT embeddings shape: {_self.bert_embeddings.shape}")
        else:
            _self.bert_embeddings = None
        
        print("Embeddings generated successfully")
    
    def get_content_recommendations(
        self, 
        query: str, 
        method: str = 'bert',
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get course recommendations based on content similarity.
        
        Args:
            query (str): User query describing interests or skills.
            method (str): Embedding method to use ('bert' or 'tfidf').
            top_n (int): Number of recommendations to return.
            
        Returns:
            List[Dict]: List of recommended courses with metadata.
        """
        try:
            # Preprocess query
            processed_query = self._clean_text(query)
            
            if method == 'bert':
                if self.bert_model is None or self.bert_embeddings is None:
                    raise ValueError("BERT model or embeddings not initialized")
                # Use BERT embeddings
                query_embedding = self.bert_model.encode([processed_query], convert_to_numpy=True)
                similarities = cosine_similarity(query_embedding, self.bert_embeddings)[0]
                
            elif method == 'tfidf':
                if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
                    raise ValueError("TF-IDF vectorizer or matrix not initialized")
                # Use TF-IDF embeddings
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
                
            else:
                raise ValueError(f"Invalid method: {method}")
            
            # Get top recommendations
            top_indices = np.argsort(similarities)[::-1][:top_n]
            
            recommendations = []
            for idx in top_indices:
                course_data = self.df.iloc[idx]
                recommendations.append({
                    'course_name': course_data.get('Course Name', 'Unknown'),
                    'university': course_data.get('University', 'Unknown'),
                    'difficulty': course_data.get('Difficulty Level', 'Unknown'),
                    'rating': float(course_data.get('Course Rating', 0)),
                    'description': course_data.get('Course Description', ''),
                    'skills': course_data.get('Skills', ''),
                    'url': course_data.get('Course URL', '#'),
                    'similarity_score': float(similarities[idx])
                })
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return []
    
    def get_skill_based_recommendations(
        self, 
        target_skills: List[str], 
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations based on specific skills.
        
        Args:
            target_skills (List[str]): List of target skills.
            top_n (int): Number of recommendations to return.
            
        Returns:
            List[Dict]: List of recommended courses.
        """
        if self.df is None:
            print("Dataframe not loaded. Please load data first.")
            return []
        
        skill_scores = np.zeros(len(self.df))
        
        for skill in target_skills:
            skill_column = f'skill_{skill.replace(" ", "_").lower()}'
            if skill_column in self.df.columns:
                skill_scores += self.df[skill_column].values
            else:
                # Fallback to text search in 'Skills' column
                matches = self.df['Skills'].str.contains(re.escape(skill), case=False, na=False)
                skill_scores += matches.astype(int).values
        
        # Get top courses
        top_indices = np.argsort(skill_scores)[::-1][:top_n]
        
        recommendations = []
        for idx in top_indices:
            if skill_scores[idx] > 0:  # Only include courses with matching skills
                course_data = self.df.iloc[idx]
                recommendations.append({
                    'course_name': course_data.get('Course Name', 'Unknown'),
                    'university': course_data.get('University', 'Unknown'),
                    'difficulty': course_data.get('Difficulty Level', 'Unknown'),
                    'rating': float(course_data.get('Course Rating', 0)),
                    'description': course_data.get('Course Description', ''),
                    'skills': course_data.get('Skills', ''),
                    'url': course_data.get('Course URL', '#'),
                    'skill_match_score': float(skill_scores[idx])
                })
        
        return recommendations
    
    def get_course_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the course dataset.
        
        Returns:
            Dict: Dictionary containing dataset statistics.
        """
        if self.df is None:
            print("Dataframe not loaded. Please load data first.")
            return {}
        
        stats = {
            'total_courses': len(self.df),
            'universities': self.df['University'].nunique() if 'University' in self.df.columns else 0,
            'difficulty_distribution': self.df['Difficulty Level'].value_counts().to_dict() if 'Difficulty Level' in self.df.columns else {},
            'avg_rating': float(self.df['Course Rating'].mean()) if 'Course Rating' in self.df.columns else 0,
            'top_skills': self._get_top_skills() if 'Skills' in self.df.columns else [],
            'top_universities': self.df['University'].value_counts().head(5).to_dict() if 'University' in self.df.columns else {}
        }
        return stats
    
    def _get_top_skills(self) -> List[str]:
        """
        Extract and return the most common skills from the dataset.
        
        Returns:
            List[str]: List of top skills.
        """
        all_skills = []
        for skills_str in self.df['Skills']:
            if pd.notna(skills_str):
                skills = [skill.strip() for skill in str(skills_str).split(',')]
                all_skills.extend(skills)
        
        from collections import Counter
        skill_counts = Counter(all_skills)
        return [skill for skill, count in skill_counts.most_common(10)]


@st.cache_resource
def load_recommender(data_path: str = RAW_CSV_URL) -> CourseRecommender:
    """
    Initialize and return a CourseRecommender instance.
    
    Args:
        data_path (str): Path to the course dataset.
        
    Returns:
        CourseRecommender: Initialized recommender system or None if failed.
    """
    recommender = CourseRecommender(data_path)
    
    # Load and preprocess data
    if recommender.load_and_preprocess_data() is None:
        return None
    
    # Initialize models
    recommender.initialize_models()
    
    # Generate embeddings
    recommender.generate_embeddings()
    
    return recommender
