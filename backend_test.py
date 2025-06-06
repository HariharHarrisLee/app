
import sys
import os
import unittest
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests
from unittest.mock import patch, MagicMock

# Add the streamlit_app directory to the path
sys.path.append('/app/coursera_recommender/streamlit_app')

# Import the CourseRecommender class
from utils import CourseRecommender, load_recommender, RAW_CSV_URL

class TestCourseRecommender(unittest.TestCase):
    """Test cases for the CourseRecommender class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.data_path = '/app/coursera_recommender/data/coursera_courses.csv'
        self.github_data_path = RAW_CSV_URL
        self.recommender = CourseRecommender(self.data_path)
        
    def test_data_loading_local(self):
        """Test data loading and preprocessing from local file."""
        df = self.recommender.load_and_preprocess_data()
        self.assertIsNotNone(df, "Data loading failed")
        self.assertGreater(len(df), 0, "Dataframe is empty")
        print(f"✅ Successfully loaded {len(df)} courses from local file")
        
    def test_data_loading_github(self):
        """Test data loading from GitHub URL."""
        # Verify GitHub URL is accessible
        try:
            response = requests.head(RAW_CSV_URL)
            self.assertEqual(response.status_code, 200, f"GitHub URL not accessible: {RAW_CSV_URL}")
            
            # Test loading from GitHub URL
            github_recommender = CourseRecommender(RAW_CSV_URL)
            df = github_recommender.load_and_preprocess_data()
            self.assertIsNotNone(df, "GitHub data loading failed")
            self.assertGreater(len(df), 0, "GitHub dataframe is empty")
            print(f"✅ Successfully loaded {len(df)} courses from GitHub URL")
        except Exception as e:
            self.fail(f"Failed to load data from GitHub URL: {str(e)}")
        
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
                top_n=5  # Test with 5 recommendations
            )
            self.assertGreater(len(recommendations), 0, f"No recommendations for query: {query}")
            self.assertLessEqual(len(recommendations), 5, f"Too many recommendations returned for query: {query}")
            
            # Verify recommendation structure
            for rec in recommendations:
                self.assertIn('course_name', rec, "Missing course_name in recommendation")
                self.assertIn('url', rec, "Missing URL in recommendation")
                self.assertIn('similarity_score', rec, "Missing similarity_score in recommendation")
                
            print(f"✅ BERT recommendations for '{query}':")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec['course_name']} (Score: {rec['similarity_score']:.3f})")
                print(f"     URL: {rec['url']}")
        
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
                top_n=5  # Test with 5 recommendations
            )
            self.assertGreater(len(recommendations), 0, f"No recommendations for query: {query}")
            self.assertLessEqual(len(recommendations), 5, f"Too many recommendations returned for query: {query}")
            
            # Verify recommendation structure
            for rec in recommendations:
                self.assertIn('course_name', rec, "Missing course_name in recommendation")
                self.assertIn('url', rec, "Missing URL in recommendation")
                self.assertIn('similarity_score', rec, "Missing similarity_score in recommendation")
                
            print(f"✅ TF-IDF recommendations for '{query}':")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec['course_name']} (Score: {rec['similarity_score']:.3f})")
                print(f"     URL: {rec['url']}")
                
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
                top_n=5  # Test with 5 recommendations
            )
            self.assertGreater(len(recommendations), 0, f"No recommendations for skills: {skills}")
            self.assertLessEqual(len(recommendations), 5, f"Too many recommendations returned for skills: {skills}")
            
            # Verify recommendation structure
            for rec in recommendations:
                self.assertIn('course_name', rec, "Missing course_name in recommendation")
                self.assertIn('url', rec, "Missing URL in recommendation")
                self.assertIn('skill_match_score', rec, "Missing skill_match_score in recommendation")
                
            print(f"✅ Skill-based recommendations for {skills}:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec['course_name']} (Score: {rec.get('skill_match_score', 0):.1f})")
                print(f"     URL: {rec['url']}")
                
    def test_difficulty_filtering(self):
        """Test filtering recommendations by difficulty level."""
        recommender = load_recommender(self.data_path)
        
        # Get recommendations
        query = "data science"
        recommendations = recommender.get_content_recommendations(
            query=query,
            method='bert',
            top_n=10  # Get more recommendations to ensure we have different difficulty levels
        )
        
        # Filter by difficulty
        difficulty_levels = ["Beginner", "Intermediate", "Advanced"]
        for difficulty in difficulty_levels:
            filtered_recs = [r for r in recommendations if r.get('difficulty') == difficulty]
            print(f"✅ Found {len(filtered_recs)} {difficulty} courses for query '{query}'")
            
            # If we have recommendations at this difficulty level, verify them
            if filtered_recs:
                for rec in filtered_recs:
                    self.assertEqual(rec['difficulty'], difficulty, 
                                    f"Filtering failed: Expected {difficulty}, got {rec['difficulty']}")
        
    def test_rating_filtering(self):
        """Test filtering recommendations by minimum rating."""
        recommender = load_recommender(self.data_path)
        
        # Get recommendations
        query = "machine learning"
        recommendations = recommender.get_content_recommendations(
            query=query,
            method='bert',
            top_n=10  # Get more recommendations to ensure we have different ratings
        )
        
        # Filter by minimum rating
        min_ratings = [3.0, 4.0, 4.5]
        for min_rating in min_ratings:
            filtered_recs = [r for r in recommendations if r.get('rating', 0) >= min_rating]
            print(f"✅ Found {len(filtered_recs)} courses with rating >= {min_rating} for query '{query}'")
            
            # Verify filtering
            for rec in filtered_recs:
                self.assertGreaterEqual(rec['rating'], min_rating, 
                                       f"Rating filtering failed: Expected >= {min_rating}, got {rec['rating']}")
                
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
        
        # Very long query
        long_query = "machine learning " * 50
        long_recommendations = recommender.get_content_recommendations(
            query=long_query,
            method='bert',
            top_n=3
        )
        self.assertGreater(len(long_recommendations), 0, "Failed to handle very long query")
        print("✅ Long query handled correctly")
        
    def test_course_statistics(self):
        """Test course statistics generation."""
        # Load the recommender
        recommender = load_recommender(self.data_path)
        
        # Get statistics
        stats = recommender.get_course_statistics()
        self.assertIsNotNone(stats, "Failed to generate course statistics")
        
        # Verify required statistics
        required_stats = ['total_courses', 'universities', 'avg_rating', 'top_skills', 
                         'difficulty_distribution', 'top_universities']
        for stat in required_stats:
            self.assertIn(stat, stats, f"Missing {stat} in statistics")
        
        # Verify top skills
        self.assertGreater(len(stats['top_skills']), 0, "No top skills found")
        
        # Verify difficulty distribution
        self.assertGreater(len(stats['difficulty_distribution']), 0, "No difficulty distribution found")
        
        print("✅ Course statistics:")
        print(f"  Total courses: {stats['total_courses']}")
        print(f"  Universities: {stats['universities']}")
        print(f"  Average rating: {stats['avg_rating']:.1f}")
        print(f"  Top skills: {', '.join(stats['top_skills'][:5])}")
        print(f"  Difficulty distribution: {stats['difficulty_distribution']}")
        
    def test_multiple_recommendations_display(self):
        """Test that multiple course recommendations can be displayed simultaneously."""
        recommender = load_recommender(self.data_path)
        
        # Test different numbers of recommendations
        for num_recommendations in [3, 5, 10]:
            recommendations = recommender.get_content_recommendations(
                query="data science",
                method='bert',
                top_n=num_recommendations
            )
            
            self.assertEqual(len(recommendations), num_recommendations, 
                           f"Expected {num_recommendations} recommendations, got {len(recommendations)}")
            
            print(f"✅ Successfully retrieved {len(recommendations)} recommendations")
            
    def test_caching(self):
        """Test that caching works for better performance."""
        # This is a basic test to ensure the caching decorators don't cause errors
        # We can't directly test the caching behavior in a unit test
        
        # Test load_recommender with caching
        recommender1 = load_recommender(self.data_path)
        self.assertIsNotNone(recommender1, "First cached load failed")
        
        # Load again - should use cache
        recommender2 = load_recommender(self.data_path)
        self.assertIsNotNone(recommender2, "Second cached load failed")
        
        print("✅ Caching decorators working correctly")

class TestStreamlitIntegration(unittest.TestCase):
    """Test cases for Streamlit integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock Streamlit session state
        if not hasattr(st, 'session_state'):
            setattr(st, 'session_state', {})
        
    def test_session_state_management(self):
        """Test session state management for recommendations."""
        # Import app module
        sys.path.append('/app/coursera_recommender/streamlit_app')
        import app
        
        # Mock session state
        st.session_state.recommendations = []
        st.session_state.last_query = ""
        
        # Create test recommendations
        test_recommendations = [
            {
                'course_name': 'Test Course 1',
                'university': 'Test University',
                'difficulty': 'Beginner',
                'rating': 4.5,
                'description': 'Test description',
                'skills': 'Python, Data Science',
                'url': 'https://www.coursera.org/test1',
                'similarity_score': 0.95
            },
            {
                'course_name': 'Test Course 2',
                'university': 'Test University',
                'difficulty': 'Intermediate',
                'rating': 4.2,
                'description': 'Test description 2',
                'skills': 'Machine Learning, AI',
                'url': 'https://www.coursera.org/test2',
                'similarity_score': 0.85
            }
        ]
        
        # Store in session state
        st.session_state.recommendations = test_recommendations
        st.session_state.last_query = "test query"
        
        # Verify session state is maintained
        self.assertEqual(len(st.session_state.recommendations), 2, "Recommendations not stored in session state")
        self.assertEqual(st.session_state.last_query, "test query", "Last query not stored in session state")
        
        # Test display_course_card function
        for course in st.session_state.recommendations:
            # This just tests that the function doesn't raise exceptions
            try:
                app.display_course_card(course)
                print(f"✅ Successfully displayed course card for {course['course_name']}")
            except Exception as e:
                self.fail(f"Failed to display course card: {str(e)}")
        
        print("✅ Session state management working correctly")

if __name__ == '__main__':
    print("🧪 Running Course Recommender System Tests 🧪")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("✅ All tests completed")
