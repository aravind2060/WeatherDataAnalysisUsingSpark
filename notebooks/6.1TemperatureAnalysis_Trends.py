import logging
import argparse
import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import col, trim, upper, regexp_replace, when, substring, length, coalesce, lit,first,max,concat_ws,split,avg,collect_list,concat,last,mean, stddev, abs,to_date,sum as sql_sum,date_format
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.window import Window

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

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

def temperature_analysis(df):
    """Perform temperature analysis by calculating average, minimum, and maximum temperatures."""
    logging.info("Performing temperature analysis.")

    # Convert 'Date' column to date format and extract month-year for grouping
    df = df.withColumn("MonthYear", date_format(to_date(col("Date"), "yyyyMMdd"), "yyyy-MM"))

    # Calculate average, minimum, and maximum temperatures for each country/state-month combination
    temperature_df = df.groupBy("Country", "StateName", "MonthYear").agg(
        avg("TMAX").alias("AvgMaxTemp"),
        avg("TMIN").alias("AvgMinTemp"),
        avg("TAVG").alias("AvgTemp"),
        max("TMAX").alias("MaxTemp"),
        min("TMIN").alias("MinTemp")
    )

    return temperature_df

    

def save_output(df, output_path, output_file,headerr=True):
    """Save the augmented data to the specified path as a single CSV file without quotes."""
    full_path = os.path.join(output_path, output_file)
    temp_path = os.path.join(output_path, "temp")

    logging.info(f"Preparing to save output to: {full_path}")
    df.show(5)

    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)

    df.coalesce(1).write.option("quote", "").csv(temp_path, header=headerr)
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
        """Example""" """CA001012475,20240209,50.0,67.0,59.0,,BRITISH COLUMBIA,DISCOVERY ISLAND,CANADA"""
        data_schema = "StationID STRING, Date STRING, TMIN Float, TMAX Float,TAVG Float,PRCP Float, StateName STRING, LocationName STRING,Country STRING"

        df = load_data(spark, input_filepath, data_schema)
        df.show(5)
        temp_df = temperature_analysis(df)
        logging.info("Temperature analysis completed.")
        temp_df.show(5)
        
        save_output(temp_df, output_filepath, file)

def main():
    """Main function to orchestrate the data processing using Spark."""
    args = parse_arguments()
    spark = create_spark_session(args.app_name, args.master_node_url)
    process_files(spark, args.input_filepath, args.output_filepath);
    stopSparkSession(spark)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process weather data.")
    parser.add_argument("--master_node_url", default="local[*]", help="URL of the master node.")
    parser.add_argument("--app_name", default="Temperature Analysis", help="Name of the Spark application.")
    parser.add_argument("--input_filepath", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output8/", help="Input directory path.")
    parser.add_argument("--output_filepath", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output10/", help="Output directory path.")
    return parser.parse_args()

if __name__ == "__main__":
    main()