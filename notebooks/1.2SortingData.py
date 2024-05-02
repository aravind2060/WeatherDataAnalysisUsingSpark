import logging
import shutil
import os
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_spark_session(app_name, master_node_url):
    """Create a SparkSession instance for running Spark operations."""
    logging.info("Creating Spark session.")
    return SparkSession.builder \
        .appName(app_name) \
        .master(master_node_url) \
        .getOrCreate()

def load_data(spark, file_path,schema):
    """Load data from a CSV file into a DataFrame."""
    logging.info(f"Loading data from: {file_path}")
    return spark.read.csv(file_path, header=False, schema=schema)

def sort_data(df):
    """Sort data based on stationID, date, and elementType."""
    logging.info("Sorting data by stationID, date, and elementType.")
    sorted_df = df.sort(["StationID", "Date", "ElementType"])
    return sorted_df

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

def process_files(spark, input_dir, output_filepath):
    """Process each file in the input directory and save it with the same name in the output directory."""
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    for file in files:
        logging.info(f"Processing file: {file}")
        input_filepath = os.path.join(input_dir, file)
        
        data_schema = "StationID STRING, Date STRING, ElementType STRING, Value STRING, StateName STRING, LocationName STRING,Country STRING"
        
        data_df = load_data(spark, input_filepath,data_schema)
        
        sorted_df = sort_data(data_df)
        
        save_output(sorted_df, output_filepath,file)
        

def main(app_name,master_node_url,input_path, output_path):
    """Main function to orchestrate the sorting operation."""
    spark = create_spark_session(app_name, master_node_url)    
    process_files(spark,input_path,output_path);
    spark.stop()
    logging.info("Spark job completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort weather data based on composite keys.")
    parser.add_argument("--master_node_url", default="local[*]", help="URL of the master node.")
    parser.add_argument("--app_name", default="DataSorting", help="Name of the Spark application.")
    parser.add_argument("--input_filepath", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output2/", help="Input directory path.")
    parser.add_argument("--output_filepath", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output3/", help="Output directory path.")
    args = parser.parse_args()
    main(args.app_name,args.master_node_url,args.input_filepath, args.output_filepath)
