
import sys
import os
import unittest
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add the streamlit_app directory to the path
sys.path.append('/app/coursera_recommender/streamlit_app')

# Import the CourseRecommender class
from utils import CourseRecommender, load_recommender

class TestCourseRecommender(unittest.TestCase):
    """Test cases for the CourseRecommender class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.data_path = '/app/coursera_recommender/data/coursera_courses.csv'
        self.recommender = CourseRecommender(self.data_path)
        
    def test_data_loading(self):
        """Test data loading and preprocessing."""
        df = self.recommender.load_and_preprocess_data()
        self.assertIsNotNone(df, "Data loading failed")
        self.assertGreater(len(df), 0, "Dataframe is empty")
        print(f"✅ Successfully loaded {len(df)} courses")
        
    def test_model_initialization(self):
        """Test model initialization."""
        self.recommender.load_and_preprocess_data()
        self.recommender.initialize_models()
        self.assertIsNotNone(self.recommender.bert_model, "BERT model initialization failed")
        self.assertIsNotNone(self.recommender.tfidf_vectorizer, "TF-IDF vectorizer initialization failed")
        print("✅ Models initialized successfully")
        
    def test_embedding_generation(self):
        """Test embedding generation."""
        self.recommender.load_and_preprocess_data()
        self.recommender.initialize_models()
        self.recommender.generate_embeddings()
        self.assertIsNotNone(self.recommender.tfidf_matrix, "TF-IDF matrix generation failed")
        self.assertIsNotNone(self.recommender.bert_embeddings, "BERT embeddings generation failed")
        print(f"✅ Generated embeddings - TF-IDF shape: {self.recommender.tfidf_matrix.shape}, BERT shape: {self.recommender.bert_embeddings.shape}")
        
    def test_bert_recommendations(self):
        """Test BERT-based recommendations."""
        # Load the recommender
        recommender = load_recommender(self.data_path)
        self.assertIsNotNone(recommender, "Recommender initialization failed")
        
        # Test queries
        test_queries = [
            "machine learning and artificial intelligence",
            "web development with JavaScript",
            "data science and visualization"
        ]
        
        for query in test_queries:
            recommendations = recommender.get_content_recommendations(
                query=query,
                method='bert',
                top_n=3
            )
            self.assertGreater(len(recommendations), 0, f"No recommendations for query: {query}")
            print(f"✅ BERT recommendations for '{query}':")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec['course_name']} (Score: {rec['similarity_score']:.3f})")
        
    def test_tfidf_recommendations(self):
        """Test TF-IDF recommendations."""
        # Load the recommender
        recommender = load_recommender(self.data_path)
        
        # Test queries
        test_queries = [
            "python programming",
            "finance and economics",
            "cybersecurity"
        ]
        
        for query in test_queries:
            recommendations = recommender.get_content_recommendations(
                query=query,
                method='tfidf',
                top_n=3
            )
            self.assertGreater(len(recommendations), 0, f"No recommendations for query: {query}")
            print(f"✅ TF-IDF recommendations for '{query}':")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec['course_name']} (Score: {rec['similarity_score']:.3f})")
                
    def test_skill_based_recommendations(self):
        """Test skill-based recommendations."""
        # Load the recommender
        recommender = load_recommender(self.data_path)
        
        # Test skills
        test_skills = [
            ["Python", "Data Science"],
            ["Web Development", "JavaScript"],
            ["Machine Learning", "Deep Learning"]
        ]
        
        for skills in test_skills:
            recommendations = recommender.get_skill_based_recommendations(
                target_skills=skills,
                top_n=3
            )
            self.assertGreater(len(recommendations), 0, f"No recommendations for skills: {skills}")
            print(f"✅ Skill-based recommendations for {skills}:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec['course_name']} (Score: {rec.get('skill_match_score', 0):.1f})")
                
    def test_edge_cases(self):
        """Test edge cases like empty queries and special characters."""
        # Load the recommender
        recommender = load_recommender(self.data_path)
        
        # Empty query
        empty_recommendations = recommender.get_content_recommendations(
            query="",
            method='bert',
            top_n=3
        )
        print(f"Empty query test: {'Passed' if len(empty_recommendations) > 0 else 'Failed'}")
        
        # Special characters
        special_recommendations = recommender.get_content_recommendations(
            query="!@#$%^&*() machine learning",
            method='bert',
            top_n=3
        )
        self.assertGreater(len(special_recommendations), 0, "Failed to handle special characters")
        print("✅ Special characters handled correctly")
        
    def test_course_statistics(self):
        """Test course statistics generation."""
        # Load the recommender
        recommender = load_recommender(self.data_path)
        
        # Get statistics
        stats = recommender.get_course_statistics()
        self.assertIsNotNone(stats, "Failed to generate course statistics")
        self.assertIn('total_courses', stats, "Missing total_courses in statistics")
        self.assertIn('top_skills', stats, "Missing top_skills in statistics")
        
        print("✅ Course statistics:")
        print(f"  Total courses: {stats['total_courses']}")
        print(f"  Universities: {stats['universities']}")
        print(f"  Average rating: {stats['avg_rating']:.1f}")
        print(f"  Top skills: {', '.join(stats['top_skills'][:5])}")

if __name__ == '__main__':
    print("🧪 Running Course Recommender System Tests 🧪")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("✅ All tests completed")
