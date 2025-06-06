"""
Streamlit Application for Course Recommender System.

This is the main web interface for the Coursera Course Recommender System
using NLP, Deep Learning, and interactive visualizations.
"""

import subprocess
import sys
import os

# Install plotly if not already installed  (DO NOT REMOVE THIS LINE OF CODE)
#subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])


import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import List, Dict, Any


# Add the parent directory to the path to import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils import CourseRecommender, load_recommender


# Page configuration
st.set_page_config(
    page_title="🎓 Course Recommender System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better styling
st.markdown(
    """
    <style>
        .main-header {
            font-size: 3rem;
            color: #1e3d59;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
        }
        
        .sub-header {
            font-size: 1.5rem;
            color: #2e5266;
            margin: 1rem 0;
        }
        
        .course-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #4a90e2;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .metric-card {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .skill-tag {
            display: inline-block;
            background-color: #e3f2fd;
            color: #1565c0;
            padding: 0.3rem 0.8rem;
            margin: 0.2rem;
            border-radius: 20px;
            font-size: 0.8rem;
        }
        
        .stButton > button {
            background-color: #4a90e2;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
        }
    </style>
    """, 
    unsafe_allow_html=True
)


@st.cache_resource
def initialize_recommender():
    """
    Initialize the course recommender system with caching.

    Returns:
        CourseRecommender: Initialized recommender instance.
    """
    data_path = os.path.join(current_dir, 'Coursera.csv')
    return load_recommender(data_path)


def display_course_card(course: Dict[str, Any], show_similarity: bool = True):
    """
    Display a course in a formatted card layout.

    Args:
        course (Dict): Course information dictionary.
        show_similarity (bool): Whether to show similarity score.
    """
    with st.container():

        st.markdown(
            f"""
            <div class="course-card">
                <h3 style="color: #1e3d59; margin-bottom: 0.5rem;">{course.get('course_name', 'N/A')}</h3>
                <p style="color: #666; margin-bottom: 1rem;"><strong>🏛️ {course.get('university', 'N/A')}</strong> | 
                📊 {course.get('difficulty', 'N/A')} | ⭐ {course.get('rating', 'N/A')}/5.0</p>
                <p style="margin-bottom: 1rem;">{course.get('description', '')[:200]}...</p>
            </div>
            """, 
            unsafe_allow_html=True
        )


        # Skills tags
        if course.get('skills'):
            skills = [skill.strip() for skill in str(course['skills']).split(',')]
            skill_html = ""

            for skill in skills[:5]:  # Show only first 5 skills
                skill_html += f'<span class="skill-tag">{skill}</span>'

            st.markdown(skill_html, unsafe_allow_html=True)


        # Similarity score and view course button
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            if show_similarity and 'similarity_score' in course:
                st.metric("🎯 Similarity", f"{course['similarity_score']:.3f}")
            elif 'skill_match_score' in course:
                st.metric("🎯 Skill Match", f"{course['skill_match_score']:.0f}")

        with col2:
            # FIXED: Use st.link_button for proper new tab opening
            course_url = course.get('url', '#')
            if course_url and course_url != '#':
                st.link_button(
                    "📚 View Course", 
                    url=course_url,
                    help="Open course in new tab"
                )
            else:
                st.markdown("🚫 No course URL available")

        st.markdown("---")


def create_wordcloud(text_data: List[str], title: str = "Word Cloud"):
    """
    Create and display a word cloud from text data.

    Args:
        text_data (List[str]): List of text strings.
        title (str): Title for the word cloud.
    """
    try:
        combined_text = ' '.join(text_data)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(combined_text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")


def create_analytics_dashboard(recommender: CourseRecommender):
    """
    Create and display analytics dashboard.

    Args:
        recommender (CourseRecommender): Initialized recommender system.
    """
    st.markdown('<h2 class="sub-header">📊 Course Analytics Dashboard</h2>', unsafe_allow_html=True)

    stats = recommender.get_course_statistics()


    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3 style="color: #4a90e2;">{stats.get('total_courses', 0)}</h3>
                <p>Total Courses</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3 style="color: #4a90e2;">{stats.get('universities', 0)}</h3>
                <p>Universities</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3 style="color: #4a90e2;">{stats.get('avg_rating', 0):.1f}</h3>
                <p>Avg Rating</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3 style="color: #4a90e2;">{len(stats.get('top_skills', []))}</h3>
                <p>Top Skills</p>
            </div>
            """, 
            unsafe_allow_html=True
        )


    col1, col2 = st.columns(2)

    with col1:
        if stats.get('difficulty_distribution'):
            fig_diff = px.pie(
                values=list(stats['difficulty_distribution'].values()),
                names=list(stats['difficulty_distribution'].keys()),
                title="📈 Course Difficulty Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_diff.update_layout(height=400)
            st.plotly_chart(fig_diff, use_container_width=True)

    with col2:
        if stats.get('top_universities'):
            fig_uni = px.bar(
                x=list(stats['top_universities'].values()),
                y=list(stats['top_universities'].keys()),
                orientation='h',
                title="🏛️ Top Universities by Course Count",
                color=list(stats['top_universities'].values()),
                color_continuous_scale='Blues'
            )
            fig_uni.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_uni, use_container_width=True)


    if stats.get('top_skills'):
        st.markdown('<h3 class="sub-header">🎯 Popular Skills</h3>', unsafe_allow_html=True)
        skills_text = ' '.join(stats['top_skills'] * 3)  # Repeat for better visualization
        create_wordcloud([skills_text], "Most Popular Skills in Courses")


def main():
    """Main application function."""

    st.markdown('<h1 class="main-header">🎓 Personalized Course Recommender System</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #666;">'
        'Using NLP, Deep Learning, and Streamlit UI on the Coursera Dataset 2021</p>', 
        unsafe_allow_html=True
    )

    # Initialize session state for maintaining app state
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""

    with st.spinner("🚀 Initializing AI models and loading course data..."):
        recommender = initialize_recommender()


    if recommender is None:
        st.error("❌ Failed to initialize the recommender system. Please check the data file.")
        return


    st.success("✅ Recommender system initialized successfully!")


    st.sidebar.markdown("## 🎯 Find Your Perfect Course")

    recommendation_method = st.sidebar.selectbox(
        "🔧 Choose Recommendation Method:",
        ["🤖 BERT-based (AI)", "📊 TF-IDF (Statistical)", "🎯 Skill-based"]
    )

    num_recommendations = st.sidebar.slider(
        "📝 Number of recommendations:",
        min_value=3,
        max_value=10,
        value=5
    )


    tab1, tab2, tab3 = st.tabs(["🔍 Get Recommendations", "📊 Analytics Dashboard", "ℹ️ About"])


    with tab1:
        st.markdown('<h2 class="sub-header">🔍 Find Courses for You</h2>', unsafe_allow_html=True)

        if "BERT" in recommendation_method or "TF-IDF" in recommendation_method:

            user_query = st.text_area(
                "💭 What do you want to learn? (Describe your interests, goals, or skills)",
                placeholder="e.g., machine learning, data science, web development, artificial intelligence...",
                height=100,
                value=st.session_state.last_query
            )

            col1, col2 = st.columns(2)

            with col1:
                difficulty_filter = st.selectbox(
                    "📈 Preferred Difficulty:",
                    ["Any", "Beginner", "Intermediate", "Advanced"]
                )

            with col2:
                min_rating = st.slider(
                    "⭐ Minimum Rating:",
                    min_value=0.0,
                    max_value=5.0,
                    value=4.0,
                    step=0.1
                )

            if st.button("🔍 Get Recommendations", type="primary"):

                if user_query.strip():
                    with st.spinner("🤖 AI is analyzing courses and generating personalized recommendations..."):

                        method = 'bert' if "BERT" in recommendation_method else 'tfidf'

                        recommendations = recommender.get_content_recommendations(
                            query=user_query,
                            method=method,
                            top_n=num_recommendations
                        )

                        if difficulty_filter != "Any":
                            recommendations = [r for r in recommendations if r.get('difficulty') == difficulty_filter]

                        recommendations = [r for r in recommendations if r.get('rating', 0) >= min_rating]

                        # Store recommendations in session state
                        st.session_state.recommendations = recommendations
                        st.session_state.last_query = user_query

                else:
                    st.warning("💭 Please enter a search query to get recommendations.")

            # Display stored recommendations
            if st.session_state.recommendations:
                st.markdown(
                    f'<h3 class="sub-header">🎯 Top {len(st.session_state.recommendations)} Recommendations for: "{st.session_state.last_query}"</h3>', 
                    unsafe_allow_html=True
                )

                for i, course in enumerate(st.session_state.recommendations, 1):
                    st.markdown(f"### {i}. 📚 Recommended Course")
                    display_course_card(course, show_similarity=True)
            elif st.session_state.last_query:
                st.warning("🤔 No courses found matching your criteria. Try adjusting your filters or query.")


        elif "Skill-based" in recommendation_method:
            st.markdown("### 🎯 Select Skills You Want to Learn")

            stats = recommender.get_course_statistics()
            available_skills = stats.get('top_skills', [])

            selected_skills = st.multiselect(
                "Choose skills:",
                available_skills,
                default=available_skills[:3] if len(available_skills) >= 3 else available_skills
            )

            custom_skills = st.text_input(
                "Or enter custom skills (comma-separated):",
                placeholder="e.g., Python, Machine Learning, Data Analysis"
            )

            if custom_skills:
                custom_skill_list = [skill.strip() for skill in custom_skills.split(',')]
                selected_skills.extend(custom_skill_list)

            if st.button("🔍 Find Courses by Skills", type="primary"):

                if selected_skills:
                    with st.spinner("🎯 Finding courses that match your skills..."):

                        recommendations = recommender.get_skill_based_recommendations(
                            target_skills=selected_skills,
                            top_n=num_recommendations
                        )

                        # Store skill-based recommendations in session state
                        st.session_state.recommendations = recommendations
                        st.session_state.last_query = f"Skills: {', '.join(selected_skills)}"

                else:
                    st.warning("🎯 Please select at least one skill.")

            # Display skill-based recommendations
            if st.session_state.recommendations and "Skills:" in st.session_state.last_query:
                st.markdown(
                    f'<h3 class="sub-header">🎯 Courses for {st.session_state.last_query}</h3>', 
                    unsafe_allow_html=True
                )

                for i, course in enumerate(st.session_state.recommendations, 1):
                    st.markdown(f"### {i}. 📚 Skill-Matched Course")
                    display_course_card(course, show_similarity=False)
            elif "Skills:" in st.session_state.last_query:
                st.warning("🤔 No courses found for the selected skills.")


    with tab2:
        create_analytics_dashboard(recommender)


    with tab3:
        st.markdown(
            """
            ## 🎓 About This Course Recommender System
            
            ### 🚀 Overview
            This is an advanced course recommendation system built using state-of-the-art **NLP** and **Deep Learning** techniques. 
            The system analyzes course content and provides personalized recommendations based on your interests and goals.
            
            ### 🧠 Technology Stack
            - **🤖 BERT Embeddings**: Advanced transformer-based language model for semantic understanding
            - **📊 TF-IDF**: Statistical approach for text analysis and similarity computation
            - **🎯 Skill Matching**: Multi-hot encoding and skill-based filtering
            - **🖥️ Streamlit**: Interactive web interface for seamless user experience
            - **📈 Plotly**: Interactive visualizations and analytics
            
            ### 🔧 Recommendation Methods
            
            1. **🤖 BERT-based Recommendations**
               - Uses pre-trained BERT model (`all-MiniLM-L6-v2`)
               - Semantic understanding of course content
               - Best for natural language queries
            
            2. **📊 TF-IDF Statistical Approach**
               - Term frequency-inverse document frequency analysis
               - Fast and efficient for keyword-based search
               - Good for specific technical terms
            
            3. **🎯 Skill-based Matching**
               - Direct skill matching from course metadata
               - Perfect for targeted skill development
               - Supports both predefined and custom skills
            
            ### 📊 Dataset Information
            - **Source**: Coursera Courses Dataset 2021
            - **Courses**: 15+ courses across various domains
            - **Features**: Course names, descriptions, skills, difficulty levels, ratings, universities
            
            ### 🎯 Key Features
            - ✅ **Multi-method Recommendations**: Choose between AI-powered or statistical approaches
            - ✅ **Interactive Filtering**: Filter by difficulty, rating, and skills
            - ✅ **Real-time Analytics**: Live dashboard with course statistics
            - ✅ **Semantic Search**: Understand context and intent, not just keywords
            - ✅ **Skill Visualization**: Word clouds and interactive charts
            - ✅ **Working Course Links**: Direct access to Coursera course pages
            
            ### 🚀 Future Enhancements
            - Neural Collaborative Filtering for user-based recommendations
            - Fine-tuned BERT models for education domain
            - Real-time learning from user interactions
            - Advanced hybrid recommendation models
            
            ### 👨‍💻 Technical Implementation
            This system demonstrates modern ML engineering practices including:
            - Object-oriented design with comprehensive docstrings
            - Efficient caching and model initialization
            - Interactive UI with responsive design
            - Scalable architecture for production deployment
            
            ---
            
            **Built with ❤️ using Python, Streamlit, and Transformers**
            """
        )


if __name__ == "__main__":
    main()
