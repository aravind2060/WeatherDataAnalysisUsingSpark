import logging
import argparse
import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, upper, regexp_replace, when, substring, length, coalesce, lit,first,max,concat_ws,split,avg,collect_list,concat,last,mean, stddev, abs
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

def precipitation_outliers(input_df):
    """
    Identify and adjust precipitation outliers using the standard deviation method.
    Outliers are defined as values that are more than 3 standard deviations from the mean.
    Adjusted values will be set to the mean of non-outlier values.
    """
    logging.info("Identifying and adjusting precipitation outliers.")

    # Calculate mean and standard deviation of PRCP within each group
    stats_df = input_df.groupBy("StationID", "Date").agg(
        mean("PRCP").alias("mean_PRCP"),
        stddev("PRCP").alias("stddev_PRCP")
    )

    # Join back to original data to access these statistics
    adjusted_df = input_df.join(stats_df, ["StationID", "Date"])

    # Define outliers as values more than 3 standard deviations from the mean
    adjusted_df = adjusted_df.withColumn(
        "PRCP",
        when(
            abs(col("PRCP") - col("mean_PRCP")) > 3 * col("stddev_PRCP"),
            col("mean_PRCP")
        ).otherwise(col("PRCP"))
    )

    # Drop temporary columns and return the cleaned DataFrame
    adjusted_df = adjusted_df.drop("mean_PRCP", "stddev_PRCP")

    return adjusted_df




def save_output(df, output_path, output_file):
    """Save the augmented data to the specified path as a single CSV file without quotes."""
    full_path = os.path.join(output_path, output_file)
    temp_path = os.path.join(output_path, "temp")

    logging.info(f"Preparing to save output to: {full_path}")
    df.show(5)

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
        data_schema = "StationID STRING, Date STRING, TMIN Float, TMAX Float,TAVG Float,PRCP Float, StateName STRING, LocationName STRING,Country STRING"

        df = load_data(spark, input_filepath, data_schema)
        df.show(5)
        dff = precipitation_outliers(df);
        save_output(dff, output_filepath, file)

def main():
    """Main function to orchestrate the data processing using Spark."""
    args = parse_arguments()
    spark = create_spark_session(args.app_name, args.master_node_url)
    process_files(spark, args.input_filepath, args.output_filepath)
    stopSparkSession(spark)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process weather data.")
    parser.add_argument("--master_node_url", default="local[*]", help="URL of the master node.")
    parser.add_argument("--app_name", default="Precipitation Outliers", help="Name of the Spark application.")
    parser.add_argument("--input_filepath", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output6/", help="Input directory path.")
    parser.add_argument("--output_filepath", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output7/", help="Output directory path.")
    return parser.parse_args()

if __name__ == "__main__":
    main()