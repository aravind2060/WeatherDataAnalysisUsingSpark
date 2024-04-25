import os
import shutil
import argparse
import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, trim, concat_ws, regexp_replace

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_spark_session(app_name, master_node_url):
    """Initialize and return a Spark session with the specified application name and master URL."""
    logging.info("Creating Spark session with app name: %s and master URL: %s", app_name, master_node_url)
    return SparkSession.builder \
        .appName(app_name) \
        .master(master_node_url) \
        .getOrCreate()
        
def stopSparkSession(spark):
    logging.info("Stopping the Spark session.")
    spark.stop()

def read_data(spark, filepath):
    """Load data from a CSV file, infer schema based on the data, and select the first four columns with specified headers."""
    logging.info("Reading data from: %s", filepath) 
    df = spark.read.csv(filepath, inferSchema=True, header=False, encoding="utf-8")
    return df.select(df.columns[:4]).toDF("stationID", "date", "element", "value")

def filter_and_transform(df):
    """Filter data to include only records from the US, Canada, or Mexico and specific climate elements."""
    logging.info("Filtering data.")
    return df.filter(
        (col("stationID").startswith("US") | col("stationID").startswith("CA") | col("stationID").startswith("MX")) &
        col("element").isin(["TMAX", "TMIN", "TAVG", "PRCP"])
    )

def format_output(df):
    """Create a single string column by concatenating relevant fields, separated by commas."""
    logging.info("Formatting output.")
    formatted_df = df.withColumn("output", trim(concat_ws(",", col("stationID"), col("date"), col("element"), col("value"))))
    formatted_df = formatted_df.withColumn("output", regexp_replace("output", "[^\x20-\x7E]", ""))
    return formatted_df.select("output")

def save_output(df: DataFrame, output_path: str, output_file: str):
    """Save the DataFrame to the specified path as a single CSV file without quotes.
    
    Args:
    df: DataFrame to be saved.
    output_path: The directory path where the final CSV will be saved.
    output_file: The final output file name.
    """
    logging.info("Saving output to %s as %s.", output_path, output_file)
    temp_path = os.path.join(output_path, "temp")
    
    # Ensure the temp directory is clean before writing
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    
    # Write DataFrame to a temporary path
    df.coalesce(1).write.option("quote", "").csv(temp_path, header=False)
    
    # Find the part file that Spark outputs
    part_files = [f for f in os.listdir(temp_path) if f.startswith('part')]
    if part_files:
        # There should be only one part file due to coalesce(1)
        os.rename(os.path.join(temp_path, part_files[0]), os.path.join(output_path, output_file))
    
    # Clean up the temporary directory
    shutil.rmtree(temp_path)

def process_files(spark, input_dir, output_filepath):
    """Process each file in the input directory and save it with the same name in the output directory."""
    # List all CSV files in the input directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    for file in files:
        input_filepath = os.path.join(input_dir, file)
        logging.info("Processing file: %s", file)
        
        # Read and process each file
        df = read_data(spark, input_filepath)
        filtered_df = filter_and_transform(df)
        final_df = format_output(filtered_df)
        
        # Save the processed data to the output directory with the same filename
        save_output(final_df, output_filepath, file)

def main(args):
    spark = create_spark_session(args.app_name, args.master_node_url)
    process_files(spark, args.input_filepath, args.output_filepath)
    stopSparkSession(spark)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process weather data.")
    parser.add_argument("--master_node_url", default="local[*]", help="URL of the master node.")
    parser.add_argument("--app_name", default="GHCN Step 1 Filtering only required data", help="Name of the Spark application.")
    parser.add_argument("--input_filepath", default="/workspaces/WeatherDataAnalysisUsingSpark/data/input/", help="Input directory path.")
    parser.add_argument("--output_filepath", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output/", help="Output directory path.")
    args = parser.parse_args()
    main(args)
