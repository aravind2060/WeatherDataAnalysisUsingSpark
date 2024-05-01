import logging
import argparse
import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, upper, regexp_replace, when, substring, length, coalesce, lit,first,max,concat_ws,split,avg,collect_list,concat,last,mean, stddev, abs,lag,datediff,sum as sql_sum,to_date,date_format
from pyspark.sql.types import StructType, StructField, StringType, IntegerType,DateType
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

def detect_and_exclude_gaps(df):
    """
    Detect and exclude data where there are gaps of 3 or more consecutive days.
    """
    logging.info("Detecting and excluding data for gaps of 3 or more consecutive days.")

    # Ensure the 'Date' column is in the proper date format
    df = df.withColumn("Date", to_date(col("Date"), "yyyyMMdd"))

    # Define a window partitioned by 'StationID' and ordered by 'Date'
    window_spec = Window.partitionBy("StationID").orderBy("Date")

    # Calculate the difference in days between current and previous date
    df = df.withColumn("PrevDate", lag("Date", 1).over(window_spec))
    df = df.withColumn("DayDiff", datediff(col("Date"), col("PrevDate")))

    # Identify gap starts (where difference >= 3 days)
    df = df.withColumn("IsGapStart", (col("DayDiff") >= 3).cast("integer"))

    # Create a cumulative sum to identify all days that are part of a gap sequence
    df = df.withColumn("GapGroup", sql_sum("IsGapStart").over(window_spec.rowsBetween(Window.unboundedPreceding, 0)))

    # Exclude entries that are part of a gap sequence
    filtered_df = df.filter(col("GapGroup") == 0)

    # Reformat the 'Date' column to remove hyphens
    filtered_df = filtered_df.withColumn("Date", date_format(col("Date"), "yyyyMMdd"))

    # Select specific columns in the desired order
    filtered_df = filtered_df.select("StationID", "Date", "ElementType", "Value", "StateName", "LocationName", "Country")
    
    return filtered_df

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
        df = load_data(spark,input_filepath,data_schema);
        df.show(5);
        filtered_df = detect_and_exclude_gaps(df);
        filtered_df.show(5);
        save_output(filtered_df, output_filepath, file)


def main(app_name,master_node_url,input_path, output_path):
    """Main function to orchestrate the sorting operation."""
    spark = create_spark_session(app_name, master_node_url)    
    process_files(spark,input_path,output_path);
    spark.stop()
    logging.info("Spark job completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discarding Data which have gaps.")
    parser.add_argument("--master_node_url", default="local[*]", help="URL of the master node.")
    parser.add_argument("--app_name", default="Data Discarding with gaps", help="Name of the Spark application.")
    parser.add_argument("--input_filepath", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output3/", help="Input directory path.")
    parser.add_argument("--output_filepath", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output4/", help="Output directory path.")
    args = parser.parse_args()
    main(args.app_name,args.master_node_url,args.input_filepath, args.output_filepath)
