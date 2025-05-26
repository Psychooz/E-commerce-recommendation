import streamlit as st
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.sparse import csr_matrix

# Add script directory to Python path
sys.path.append(os.path.dirname(__file__))

# Import the recommendation engine
from spark_recommendation import RecommendationEngine

def load_recommendation_engine(csv_path):
    """
    Load recommendation engine with error handling
    
    Args:
        csv_path (str): Path to reviews CSV
    
    Returns:
        RecommendationEngine or None
    """
    try:
        return RecommendationEngine(csv_path)
    except Exception as e:
        st.error(f"Failed to load recommendation engine: {e}")
        return None

def evaluate_model(engine, test_size=0.2):
    """Evaluate ALS model performance with improved implementation"""
    try:
        if engine.als_model is None:
            st.warning("ALS model not available for evaluation")
            return None
        
        # Create a sample of users for evaluation (for performance)
        eval_users = np.random.choice(
            list(engine.user_to_idx.keys()),
            size=min(500, len(engine.user_to_idx)),  # Evaluate on max 500 users
            replace=False
        )
        
        precisions = []
        recalls = []
        f1_scores = []
        test_samples = 0
        
        for user_id in eval_users:
            # Get user's actual liked items (ratings >= 4)
            user_actual = engine.df[
                (engine.df['userId'] == user_id) & 
                (engine.df['score'] >= 4)
            ]['productId'].unique()
            
            if len(user_actual) == 0:
                continue  # Skip users with no positive ratings
                
            # Get user's predicted liked items
            try:
                user_idx = engine.user_to_idx[user_id]
                rec_indices, _ = engine.als_model.recommend(
                    user_idx, 
                    engine.interaction_matrix, 
                    N=20,  # Get top 20 predictions
                    filter_already_liked_items=False
                )
                user_predicted = [engine.idx_to_product[idx] for idx in rec_indices]
            except Exception as e:
                continue
                
            # Create binary vectors for metrics calculation
            all_items = set(engine.product_to_idx.keys())
            y_true = [1 if item in user_actual else 0 for item in all_items]
            y_pred = [1 if item in user_predicted else 0 for item in all_items]
            
            # Calculate metrics for this user
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
            test_samples += 1
        
        if test_samples == 0:
            return None
            
        return {
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1_score': np.mean(f1_scores),
            'test_samples': test_samples
        }
        
    except Exception as e:
        st.error(f"Evaluation error: {str(e)}")
        return None

def show_model_stats(engine):
    """Display ALS model statistics with better formatting"""
    st.header("🧠 Model Statistics")
    
    if engine.als_model is None:
        st.warning("ALS model not available or failed to train")
        return
    
    # Model configuration
    st.subheader("Model Configuration")
    config_cols = st.columns(4)
    config_cols[0].metric("Latent Factors", engine.als_model.factors)
    config_cols[1].metric("Iterations", engine.als_model.iterations)
    config_cols[2].metric("Regularization", f"{engine.als_model.regularization:.3f}")
    config_cols[3].metric("Training Loss", f"{getattr(engine.als_model, 'loss', 0):.4f}")
    
    # Evaluation metrics
    st.subheader("Model Performance")
    with st.spinner("Evaluating model (this may take a minute)..."):
        eval_results = evaluate_model(engine)
    
    if eval_results:
        st.success(f"Evaluated on {eval_results['test_samples']} users")
        
        # Main metrics
        metric_cols = st.columns(3)
        metric_cols[0].metric("Precision", f"{eval_results['precision']:.2%}",
                            help="Percentage of recommended items that are relevant")
        metric_cols[1].metric("Recall", f"{eval_results['recall']:.2%}",
                            help="Percentage of relevant items that are recommended")
        metric_cols[2].metric("F1 Score", f"{eval_results['f1_score']:.2%}",
                            help="Harmonic mean of precision and recall")
        
        # Explanation
        with st.expander("What do these metrics mean?"):
            st.markdown("""
            - **Precision**: Measures how many of the recommended items were actually relevant
            - **Recall**: Measures how many of the relevant items were recommended
            - **F1 Score**: Balanced measure combining both precision and recall
            
            These metrics are calculated by comparing recommendations against actual user ratings ≥4 stars.
            """)
    else:
        st.warning("Could not evaluate model performance - not enough data")

def show_user_info(engine, user_id):
    """Display information about a specific user"""
    if user_id not in engine.user_to_idx:
        st.warning(f"User {user_id} not found in database")
        return
    
    user_reviews = engine.df[engine.df['userId'] == user_id]
    if len(user_reviews) == 0:
        st.warning("No reviews found for this user")
        return
    
    st.subheader(f"👤 User Profile: {user_id}")
    
    # Basic stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", len(user_reviews))
    col2.metric("Average Rating", f"{user_reviews['score'].mean():.1f}")
    col3.metric("First Review", pd.to_datetime(user_reviews['timestamp'].min(), unit='s').strftime('%Y-%m-%d'))
    
    # Recent purchases
    st.subheader("Recently Purchased Products")
    recent_purchases = user_reviews.sort_values('timestamp', ascending=False).head(5)
    st.dataframe(recent_purchases[['title', 'price', 'score']])
    
    # Rating distribution
    st.subheader("Rating Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='score', data=user_reviews, ax=ax)
    ax.set_title("User's Rating Distribution")
    st.pyplot(fig)

def show_statistics(engine):
    """Display various statistics about the data"""
    st.header("📊 Data Statistics and Insights")
    
    # Basic statistics
    st.subheader("Basic Dataset Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", len(engine.df))
    col2.metric("Unique Users", len(engine.user_ids))
    col3.metric("Unique Products", len(engine.product_ids))
    
    # Create a copy of the dataframe for visualization
    viz_df = engine.df.copy()
    
    # Convert score to numeric if it's not already
    viz_df['score'] = pd.to_numeric(viz_df['score'], errors='coerce')
    
    # Rating distribution
    st.subheader("Rating Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='score', data=viz_df.dropna(subset=['score']), ax=ax)
    ax.set_title("Distribution of Product Ratings")
    st.pyplot(fig)
    
    # Price distribution
    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(viz_df['price'].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title("Distribution of Product Prices")
    ax.set_xlabel("Price")
    st.pyplot(fig)
    
    # Top rated products
    st.subheader("Top Rated Products (with most reviews)")
    top_products = engine.product_stats.sort_values(
        ['avg_score', 'review_count'], 
        ascending=[False, False]
    ).head(10)
    st.dataframe(top_products[['title', 'price', 'avg_score', 'review_count']])
    
    # Most active users
    st.subheader("Most Active Users")
    user_activity = engine.df['userId'].value_counts().reset_index()
    user_activity.columns = ['userId', 'review_count']
    st.dataframe(user_activity.head(10))
    
    # Price vs. Rating analysis
    st.subheader("Price vs. Rating Analysis")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x='price', 
        y='avg_score', 
        size='review_count',
        sizes=(20, 200),
        alpha=0.6,
        data=engine.product_stats,
        ax=ax
    )
    ax.set_title("Price vs. Average Rating")
    st.pyplot(fig)

def main():
    # Page configuration
    st.set_page_config(
        page_title="Product Recommendation System", 
        page_icon="🛍️", 
        layout="wide"
    )
    
    # Title and description
    st.title("🎯 Advanced Product Recommendation System")
    st.markdown("""
    ### Discover Personalized Product Recommendations
    
    Choose between:
    - User-based Recommendations
    - Similar Product Recommendations
    - Data Statistics
    - Model Performance
    """)
    
    # Load recommendation engine
    csv_path = "data/reviews.csv"  # Adjust path as needed
    engine = load_recommendation_engine(csv_path)
    
    if engine is None:
        st.stop()
    
    # Sidebar configuration
    st.sidebar.header("Recommendation Options")
    rec_type = st.sidebar.radio(
        "Select Recommendation Type",
        ["User Recommendations", "Similar Products", "Data Statistics", "Model Performance"],
        index=0
    )
    
    # Recommendation count slider
    rec_count = st.sidebar.slider(
        "Number of Recommendations", 
        min_value=1, 
        max_value=20, 
        value=5,
        help="Adjust the number of recommendations to display"
    )
    
    # Example IDs
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Example IDs:\n"
        "- User ID: A31KXTOQNTWUVM\n"
        "- Product ID: B000EENAE0"
    )
    
    # User Recommendations Section
    if rec_type == "User Recommendations":
        st.header("👤 Personalized Recommendations")
        
        user_id = st.text_input(
            "Enter User ID", 
            value="A31KXTOQNTWUVM",
            help="Input a specific user ID"
        )
        
        if st.button("Get Recommendations", key="user_rec"):
            with st.spinner("Generating recommendations..."):
                try:
                    # Show user info first
                    show_user_info(engine, user_id)
                    
                    st.subheader("Recommended Products")
                    recommendations = engine.recommend_for_user(user_id, n_recommendations=rec_count)
                    
                    if len(recommendations) > 0:
                        # Display recommendations with more info
                        st.dataframe(recommendations[[
                            'title', 
                            'price', 
                            'avg_score',
                            'review_count',
                            'recommendation_score'
                        ]].sort_values('recommendation_score', ascending=False))
                        
                        # Visualize recommendations
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(
                            x='recommendation_score',
                            y='title',
                            data=recommendations.sort_values('recommendation_score', ascending=True),
                            ax=ax
                        )
                        ax.set_title("Recommendation Scores")
                        ax.set_xlabel("Confidence Score")
                        ax.set_ylabel("Product Title")
                        st.pyplot(fig)
                    else:
                        st.warning("No recommendations found.")
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")
    
    # Similar Products Section
    elif rec_type == "Similar Products":
        st.header("🔍 Find Similar Products")
        
        product_id = st.text_input(
            "Enter Product ID", 
            value="B000EENAE0",
            help="Input a product ID to find similar products"
        )
        
        if st.button("Find Similar Products", key="product_rec"):
            with st.spinner("Searching for similar products..."):
                try:
                    similar_products = engine.recommend_similar_products(product_id, n_recommendations=rec_count)
                    
                    if len(similar_products) > 0:
                        st.subheader(f"Products Similar to {product_id}")
                        st.dataframe(similar_products[['title', 'price', 'similarity_score']])
                    else:
                        st.warning("No similar products found.")
                except Exception as e:
                    st.error(f"Error finding similar products: {e}")
    
    # Statistics Section
    elif rec_type == "Data Statistics":
        show_statistics(engine)
    
    # Model Performance Section
    else:
        show_model_stats(engine)

    # Footer
    st.markdown("---")
    st.markdown("*Made By Ziad Boukhalkhal - Khalil Hamdaoui*")

if __name__ == "__main__":
    main()