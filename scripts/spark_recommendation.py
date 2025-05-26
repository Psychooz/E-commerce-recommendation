import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
import logging
import re
import os
import warnings

# Suppress numpy warnings about divide by zero in correlation calculations
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configure OpenBLAS to use just 1 thread to avoid performance issues
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class RecommendationEngine:
    def __init__(self, csv_path):
        """
        Initialize recommendation engine with robust data loading
        
        Args:
            csv_path (str): Path to the reviews CSV file
        """
        try:
            # Read CSV with flexible parsing
            self.df = self._load_csv(csv_path)
            
            # Clean and prepare data
            self._prepare_data()
            
            # Initialize ALS model
            self._prepare_als_model()
            
            logger.info(f"Loaded {len(self.df)} reviews")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def _load_csv(self, csv_path):
        """
        Load CSV with robust parsing
        
        Args:
            csv_path (str): Path to CSV file
        
        Returns:
            pandas.DataFrame: Cleaned DataFrame
        """
        # Custom converters to handle problematic columns
        def safe_float_convert(val):
            """
            Safely convert value to float
            
            Args:
                val (str): Input value
            
            Returns:
                float or np.nan: Converted value
            """
            if pd.isna(val) or val == 'unknown':
                return np.nan
            
            # Remove any non-numeric characters except decimal point
            try:
                # Remove currency symbols, commas, etc.
                clean_val = re.sub(r'[^\d.]', '', str(val))
                return float(clean_val) if clean_val else np.nan
            except (ValueError, TypeError):
                return np.nan

        def safe_str_convert(val):
            """
            Safely convert to string, handling NaN
            
            Args:
                val (any): Input value
            
            Returns:
                str: Converted value or empty string
            """
            if pd.isna(val):
                return ''
            return str(val).strip()

        # Read CSV with custom converters
        df = pd.read_csv(
            csv_path, 
            dtype={
                'productId': str, 
                'userId': str
            },
            converters={
                'price': safe_float_convert,
                'score': safe_float_convert,
                'title': safe_str_convert
            }
        )
        
        return df

    def _prepare_data(self):
        """
        Clean and prepare data for recommendations
        """
        # Remove rows with missing critical information
        self.df.dropna(subset=['productId', 'userId', 'score'], inplace=True)
        
        # Filter out invalid scores and prices
        self.df = self.df[
            (self.df['score'] >= 1) & 
            (self.df['score'] <= 5) & 
            (self.df['price'] > 0)
        ]
        
        # Remove duplicate reviews
        self.df.drop_duplicates(subset=['productId', 'userId'], inplace=True)
        
        # Create product-level aggregations
        self.product_stats = self.df.groupby('productId').agg({
            'title': 'first',  # First unique title
            'price': 'mean',   # Average price
            'score': ['count', 'mean']  # Review count and average score
        }).reset_index()
        
        self.product_stats.columns = ['productId', 'title', 'price', 'review_count', 'avg_score']
        
        # Create user and item mapping for ALS
        self.user_ids = self.df['userId'].unique()
        self.product_ids = self.df['productId'].unique()
        
        # Create mappings for ALS
        self.user_to_idx = {user: idx for idx, user in enumerate(self.user_ids)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        
        self.product_to_idx = {product: idx for idx, product in enumerate(self.product_ids)}
        self.idx_to_product = {idx: product for product, idx in self.product_to_idx.items()}

    # Updated ALS model configuration in spark_recommendation.py
    def _prepare_als_model(self):
        try:
            # Create user-item interaction matrix
            user_indices = [self.user_to_idx[user] for user in self.df['userId']]
            product_indices = [self.product_to_idx[product] for product in self.df['productId']]
            
            # Enhanced confidence scoring
            confidence_scores = 1 + 5 * self.df['score'].values  # Turns 1-5 ratings into 6-30 range
            
            # Create sparse matrix
            shape = (len(self.user_ids), len(self.product_ids))
            interaction_matrix = csr_matrix(
                (confidence_scores, (user_indices, product_indices)), 
                shape=shape
            )
            
            # Improved ALS model configuration
            self.als_model = AlternatingLeastSquares(
                factors=100,           # Increased from 50
                regularization=0.1,    # Increased regularization
                iterations=30,         # More iterations
                alpha=10,             # Confidence scaling factor
                use_native=True,
                use_cg=True,
                calculate_training_loss=True,
                random_state=42
            )
            
            # Fit the model
            self.als_model.fit(interaction_matrix * 2)  # Boost confidence further
            
            # Store the interaction matrix
            self.interaction_matrix = interaction_matrix
            
            # Calculate item-item similarities manually
            logger.info("Calculating item-item similarities...")
            item_factors = self.als_model.item_factors
            self.item_similarity = cosine_similarity(item_factors)
            
            logger.info("ALS model trained successfully with similarity matrix")
        
        except Exception as e:
            logger.error(f"Error training ALS model: {e}")
            self.als_model = None

    def recommend_for_user(self, user_id, n_recommendations=5):
        """
        Generate recommendations for a specific user using ALS
        
        Args:
            user_id (str): User ID
            n_recommendations (int): Number of recommendations
        
        Returns:
            DataFrame: Recommended products
        """
        try:
            # Validate user_id
            if not isinstance(user_id, str):
                user_id = str(user_id)
            
            # Get products user has already reviewed
            user_reviewed_products = set(
                self.df[self.df['userId'] == user_id]['productId']
            )
            
            # Check if user exists in our mapping
            if user_id not in self.user_to_idx:
                logger.warning(f"User {user_id} not found. Returning top products.")
                return self._get_top_products(n_recommendations)
            
            # Get user index
            user_idx = self.user_to_idx[user_id]
            
            # Get ALS recommendations if model is available
            if self.als_model is not None:
                try:
                    # Get recommendations from ALS model
                    rec_product_indices, rec_scores = self.als_model.recommend(
                        user_idx, 
                        self.interaction_matrix, 
                        N=n_recommendations + len(user_reviewed_products),  # Get extra to account for filtering
                        filter_already_liked_items=True
                    )
                    
                    # Convert back to product IDs
                    rec_product_ids = [self.idx_to_product[idx] for idx in rec_product_indices]
                    
                    # Filter out products user has already reviewed (extra safety check)
                    filtered_products = []
                    filtered_scores = []
                    for product_id, score in zip(rec_product_ids, rec_scores):
                        if product_id not in user_reviewed_products:
                            filtered_products.append(product_id)
                            filtered_scores.append(score)
                    
                    # If we don't have enough recommendations after filtering
                    if len(filtered_products) < n_recommendations:
                        additional_products = self._get_top_products(
                            n_recommendations - len(filtered_products)
                        )
                        remaining_product_ids = list(additional_products['productId'])
                        filtered_products.extend(remaining_product_ids)
                        # Add placeholder scores
                        filtered_scores.extend([0.5] * len(remaining_product_ids))
                    
                    # Get product information for recommended products
                    recommended_products = self.product_stats[
                        self.product_stats['productId'].isin(filtered_products[:n_recommendations])
                    ].copy()
                    
                    # Add recommendation score
                    recommendation_scores = {
                        pid: score for pid, score in zip(filtered_products, filtered_scores)
                    }
                    recommended_products['recommendation_score'] = recommended_products['productId'].map(
                        recommendation_scores
                    )
                    
                    return recommended_products.sort_values('recommendation_score', ascending=False)
                
                except Exception as e:
                    logger.error(f"ALS recommendation error for user {user_id}: {e}")
                    # Fallback to previous method
                    return self._collaborative_filtering_recommendations(user_id, n_recommendations)
            else:
                # Fallback to previous method if ALS model failed
                return self._collaborative_filtering_recommendations(user_id, n_recommendations)
        
        except Exception as e:
            logger.error(f"Recommendation error for user {user_id}: {e}")
            # Fallback to collaborative filtering or top products
            try:
                return self._collaborative_filtering_recommendations(user_id, n_recommendations)
            except Exception as e2:
                logger.error(f"Fallback recommendation error: {e2}")
                return self._get_top_products(n_recommendations)

    def _collaborative_filtering_recommendations(self, user_id, n_recommendations=5):
        """
        Fallback recommendation method using simple collaborative filtering
        
        Args:
            user_id (str): User ID
            n_recommendations (int): Number of recommendations
        
        Returns:
            DataFrame: Recommended products
        """
        # Get products user has already reviewed
        user_reviewed_products = set(
            self.df[self.df['userId'] == user_id]['productId']
        )
        
        # Get user's reviews
        user_ratings = self.df[self.df['userId'] == user_id]
        
        # If user not found, return top-rated products
        if len(user_ratings) == 0:
            logger.warning(f"User {user_id} not found. Returning top products.")
            return self._get_top_products(n_recommendations)
        
        # Find similar users based on rating patterns
        similar_users_ratings = self.df[
            ~self.df['productId'].isin(user_reviewed_products)
        ]
        
        # Score recommendations
        def compute_similarity(group):
            """
            Compute recommendation score for a product
            """
            try:
                # Compare ratings distribution
                return np.corrcoef(
                    group['score'], 
                    user_ratings['score']
                )[0,1]
            except Exception:
                return 0
        
        recommendation_scores = similar_users_ratings.groupby('productId').apply(compute_similarity)
        
        # Get top recommendations
        top_recommendations = recommendation_scores.nlargest(n_recommendations)
        
        # Merge with product stats
        recommended_products = self.product_stats[
            self.product_stats['productId'].isin(top_recommendations.index)
        ].copy()
        
        # Add recommendation score
        recommended_products['recommendation_score'] = top_recommendations.values
        
        return recommended_products.sort_values('recommendation_score', ascending=False)

    def recommend_similar_products(self, product_id, n_recommendations=5):
        """
        Find similar products based on title
        
        Args:
            product_id (str): Product ID to find similar products for
            n_recommendations (int): Number of recommendations
        
        Returns:
            DataFrame: Similar products
        """
        try:
            # Validate product_id
            if not isinstance(product_id, str):
                product_id = str(product_id)
            
            # Get current product details
            current_product = self.product_stats[
                self.product_stats['productId'] == product_id
            ]
            
            if len(current_product) == 0:
                logger.warning(f"Product {product_id} not found")
                return self._get_top_products(n_recommendations)
            
            # Prepare titles for similarity
            titles = self.product_stats['title'].fillna('')
            
            # Create TF-IDF vectorizer
            tfidf = TfidfVectorizer(stop_words='english')
            
            # Compute TF-IDF matrix
            tfidf_matrix = tfidf.fit_transform(titles)
            
            # Find similar products using cosine similarity
            current_product_title = current_product['title'].values[0]
            current_product_tfidf = tfidf.transform([current_product_title])
            cosine_similarities = cosine_similarity(current_product_tfidf, tfidf_matrix)[0]
            
            # Get top similar product indices
            similar_indices = cosine_similarities.argsort()[::-1][1:n_recommendations+1]
            
            # Return similar products
            similar_products = self.product_stats.iloc[similar_indices].copy()
            similar_products['similarity_score'] = cosine_similarities[similar_indices]
            
            return similar_products.sort_values('similarity_score', ascending=False)
        
        except Exception as e:
            logger.error(f"Similar product error for {product_id}: {e}")
            return self._get_top_products(n_recommendations)

    def _get_top_products(self, n_products=5):
        """
        Get top-rated products
        
        Args:
            n_products (int): Number of top products to return
        
        Returns:
            DataFrame: Top-rated products
        """
        return (
            self.product_stats
            .sort_values(['avg_score', 'review_count'], ascending=[False, False])
            .head(n_products)
        )

def main():
    # Example usage
    try:
        # Path to your CSV file
        csv_path = 'data/reviews.csv'
        
        # Initialize recommendation engine
        engine = RecommendationEngine(csv_path)
        
        # Example user recommendations
        print("\n=== User Recommendations (ALS) ===")
        user_id = "A31KXTOQNTWUVM"
        user_recs = engine.recommend_for_user(user_id)
        print(user_recs[['title', 'price', 'recommendation_score']])
        
        # Example similar products
        print("\n=== Similar Products ===")
        product_id = "B000EENAE0"
        similar_products = engine.recommend_similar_products(product_id)
        print(similar_products[['title', 'price', 'similarity_score']])
    
    except Exception as e:
        logger.error(f"Main execution error: {e}")

if __name__ == "__main__":
    main()