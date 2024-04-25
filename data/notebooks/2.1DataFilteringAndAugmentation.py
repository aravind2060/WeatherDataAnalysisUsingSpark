import logging
import argparse
import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, upper, regexp_replace
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_spark_session(app_name, master_node_url):    
    """Create a SparkSession instance with the specified application name."""
    logging.info("Spark session created successfully.")
    return SparkSession.builder \
        .appName(app_name) \
        .master(master_node_url) \
        .getOrCreate()

def stopSparkSession(spark):
    """Stop the Spark session."""
    spark.stop()
    logging.info("Spark session stopped.")

def load_data(spark, input_path, schema):
    """Load data from the given path using the provided schema, ensuring StationID is trimmed and consistent."""
    logging.info(f"Loading data from: {input_path}")
    data = spark.read.csv(input_path, header=False, schema=schema)
    data = data.withColumn("StationID", upper(trim(col("StationID"))))
    data = clean_data(data)
    logging.info("Data loaded and formatted correctly.")
    return data

def clean_data(data):
    """Remove non-printable characters and other unwanted artifacts from data."""
    logging.info("Cleaning data by removing non-printable characters.")
    for column in data.columns:
        data = data.withColumn(column, regexp_replace(col(column), "[^\x20-\x7E]", ""))
    return data

def load_geographic_metadata(spark, metadata_path):
    """Load the geographic metadata from the specified path and process into a dictionary for quick lookup."""
    logging.info(f"Loading geographic metadata from: {metadata_path}")
    schema = "StationID STRING, StateName STRING, LocationName STRING, Country STRING"
    metadata = spark.read.csv(metadata_path, sep=",", header=False, schema=schema)
    metadata = metadata.withColumn("StationID", upper(trim(col("StationID"))))
    metadata_dict = metadata.rdd.map(lambda row: (row[0], (row[1], row[2], row[3]))).collectAsMap()
    logging.info("Geographic metadata loaded and processed into a dictionary with %d entries.", len(metadata_dict))
    return metadata_dict

def augment_data(spark, data, metadata):
    """Augment the input data with geographic metadata using a broadcast variable for efficiency."""
    logging.info("Augmenting data with geographic metadata using broadcast variables.")
    metadata_broadcast = spark.sparkContext.broadcast(metadata)
    augmented_rdd = data.rdd.map(lambda row: (
        row["StationID"],
        row["Date"],
        row["ElementType"],
        row["Value"],
        *metadata_broadcast.value.get(row["StationID"], ("N/A", "N/A", "N/A"))
    ))
    schema = StructType([
        StructField("StationID", StringType(), True),
        StructField("Date", StringType(), True),
        StructField("ElementType", StringType(), True),
        StructField("Value", StringType(), True),
        StructField("StateName", StringType(), True),
        StructField("LocationName", StringType(), True),
        StructField("Country", StringType(), True)
    ])
    augmented_data = spark.createDataFrame(augmented_rdd, schema)
    logging.info("Data augmentation completed.")
    return augmented_data

def save_output(df, output_path, output_file):
    """Save the augmented data to the specified path as a single CSV file without quotes."""
    full_path = os.path.join(output_path, output_file)
    temp_path = os.path.join(output_path, "temp")
    logging.info(f"Preparing to save output to: {full_path}")

    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)

    df.coalesce(1).write.option("quote", "").csv(temp_path, header=False)
    part_files = [f for f in os.listdir(temp_path) if f.startswith('part')]
    if part_files:
        os.rename(os.path.join(temp_path, part_files[0]), full_path)

    shutil.rmtree(temp_path)
    logging.info(f"Output successfully saved to: {full_path}")

def process_files(spark, input_dir, output_filepath, metadata_filepath):
    """Process each file in the input directory and save it with the same name in the output directory."""
    metadata = load_geographic_metadata(spark, metadata_filepath)
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    for file in files:
        logging.info(f"Processing file: {file}")
        input_filepath = os.path.join(input_dir, file)
        data_schema = "StationID STRING, Date STRING, ElementType STRING, Value STRING"
        
        data = load_data(spark, input_filepath, data_schema)
        augmented_data = augment_data(spark, data, metadata)
        save_output(augmented_data, output_filepath, file)

def main():
    """Main function to orchestrate the data processing using Spark."""
    args = parse_arguments()
    spark = create_spark_session(args.app_name, args.master_node_url)    
    process_files(spark, args.input_filepath, args.output_filepath, args.metadata_filepath)
    stopSparkSession(spark)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process weather data.")
    parser.add_argument("--master_node_url", default="local[*]", help="URL of the master node.")
    parser.add_argument("--app_name", default="DataFilteration And Augmentation", help="Name of the Spark application.")
    parser.add_argument("--input_filepath", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output1/", help="Input directory path.")
    parser.add_argument("--output_filepath", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output2/", help="Output directory path.")
    parser.add_argument("--metadata_filepath", default="/workspaces/WeatherDataAnalysisUsingSpark/data/input/metadatafiles/", help="Metadata file path.")
    return parser.parse_args()

if __name__ == "__main__":
    main()
