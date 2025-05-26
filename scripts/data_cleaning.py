from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, when, monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, LongType
import logging
import os

def create_spark_session():
    """Create a Spark session with optimized configurations"""
    return SparkSession.builder \
        .appName("ReviewDataCleaning") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

def define_schema():
    """Define a strict schema for the input CSV"""
    return StructType([
        StructField("productId", StringType(), False),
        StructField("title", StringType(), True),
        StructField("price", StringType(), True),
        StructField("userId", StringType(), False),
        StructField("helpfulness", StringType(), True),
        StructField("score", StringType(), False),
        StructField("timestamp", LongType(), False)
    ])

def clean_reviews(input_path, output_path):
    """
    Clean and process review data
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to save processed data
    """
    # Create logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DataCleaning")
    
    # Create Spark session
    spark = create_spark_session()
    
    try:
        # Read CSV with predefined schema
        df = spark.read.csv(
            input_path, 
            schema=define_schema(), 
            header=True, 
            mode='DROPMALFORMED'
        )
        
        # Basic data cleaning
        df = df.dropna(subset=['productId', 'userId', 'score'])
        
        # Clean and convert score
        df = df.withColumn('score_clean', 
            regexp_replace(col('score'), r'[^\d.]', '').cast(FloatType())
        )
        
        # Clean and convert price
        df = df.withColumn('price_clean', 
            regexp_replace(col('price'), r'[^\d.]', '').cast(FloatType())
        )
        
        # Filter valid scores and prices
        df = df.filter(
            (col('score_clean') >= 1) & 
            (col('score_clean') <= 5) &
            (col('price_clean') > 0)
        )
        
        # Create numeric ID mappings
        user_mapping = df.select('userId').distinct() \
            .withColumn('userNumericId', monotonically_increasing_id().cast(IntegerType()))
        
        product_mapping = df.select('productId').distinct() \
            .withColumn('productNumericId', monotonically_increasing_id().cast(IntegerType()))
        
        # Join mappings
        df = df.join(user_mapping, 'userId') \
               .join(product_mapping, 'productId')
        
        # Select and rename final columns
        clean_df = df.select(
            col('productNumericId').alias('productId'),
            col('userNumericId').alias('userId'),
            col('score_clean').alias('score'),
            col('title'),
            col('price_clean').alias('price'),
            col('timestamp')
        )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save processed data
        clean_df.write.mode('overwrite').csv(output_path, header=True)
        
        # Save mappings
        user_mapping.write.mode('overwrite').csv(output_path + "_user_mapping", header=True)
        product_mapping.write.mode('overwrite').csv(output_path + "_product_mapping", header=True)
        
        logger.info(f"Data cleaned and saved to {output_path}")
        return clean_df
    
    except Exception as e:
        logger.error(f"Error in data cleaning: {e}")
        raise

if __name__ == "__main__":
    input_path = "data/reviews.csv"
    output_path = "processed_data/cleaned_reviews.csv"
    clean_reviews(input_path, output_path)