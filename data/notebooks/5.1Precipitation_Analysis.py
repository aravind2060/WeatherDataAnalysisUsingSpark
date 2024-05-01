import logging
import argparse
import os
import shutil
from pyspark.sql import SparkSession
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

def filter_data_by_location_and_time(df, location, start_date, end_date):
    """Filter data by geographic location and date range."""
    logging.info("Filtering data by location and time.")
    return df.filter(
        (col('StationID') == location) &
        (col('Date') >= start_date) &
        (col('Date') <= end_date)
    )

def aggregate_precipitation(df, time_unit):
    """
    Aggregate precipitation data based on the given time unit ('daily', 'monthly', 'yearly').
    """
    logging.info("Aggregating precipitation data by %s.", time_unit)

    if time_unit == 'monthly':
        df = df.withColumn('Month', date_format(to_date('Date', 'yyyyMMdd'), 'yyyy-MM'))
        group_col = 'Month'
    elif time_unit == 'daily':
        df = df.withColumn('Date', to_date('Date', 'yyyyMMdd'))
        group_col = 'Date'
    elif time_unit == 'yearly':
        df = df.withColumn('Year', date_format(to_date('Date', 'yyyyMMdd'), 'yyyy'))
        group_col = 'Year'
    else:
        raise ValueError("Unsupported time unit. Use 'daily', 'monthly', or 'yearly'.")

    # Perform aggregation
    aggregated_df = df.groupBy(group_col).agg(
        sql_sum('PRCP').alias('TotalPrecipitation'),
        avg('PRCP').alias('AveragePrecipitation')
    )
    return aggregated_df


def identify_anomalies(df, threshold, time_unit):
    """
    Identify heavy rainfall or drought based on thresholds.
    The `time_unit` parameter determines the column used for sorting.
    """
    logging.info("Identifying precipitation anomalies based on %s data.", time_unit)

    # Determine the appropriate column for sorting based on the aggregation level
    # sort_column = 'Date' if time_unit == 'daily' else time_unit.capitalize()
    
    if time_unit == 'daily':
        sort_column = 'Date';
    elif time_unit == 'monthly':
        sort_column = 'Month';
    elif time_unit == 'yearly':
        sort_column = 'Year';

    sorted_df = df.orderBy(sort_column)  # Sorting by the appropriate column

    # Identifying anomalies where precipitation is above or below the specified thresholds
    return sorted_df.withColumn(
        'Anomaly',
        when(col('TotalPrecipitation') >= threshold, 'Heavy Rainfall')
        .when(col('TotalPrecipitation') < threshold, 'Drought')
        .otherwise('Normal')
    )
    

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

def process_files(spark, input_dir, output_filepath,location,start_date,end_date,time_unit,threshold):
    """Process each file in the input directory and save it with the same name in the output directory."""
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    for file in files:
        logging.info(f"Processing file: {file}")
        input_filepath = os.path.join(input_dir, file)
        """Example""" """CA001012475,20240209,50.0,67.0,59.0,,BRITISH COLUMBIA,DISCOVERY ISLAND,CANADA"""
        data_schema = "StationID STRING, Date STRING, TMIN Float, TMAX Float,TAVG Float,PRCP Float, StateName STRING, LocationName STRING,Country STRING"

        df = load_data(spark, input_filepath, data_schema)
        df.show(5)
        filtered_df = filter_data_by_location_and_time(df, location, start_date, end_date)
        logging.info(f"Filtered by station id : {location}, start_date: {start_date} , end_date: {end_date} ");
        filtered_df.show(5);
        aggregated_df = aggregate_precipitation(filtered_df, time_unit);
        logging.info(f"Aggregated Precipitation based on time_unit: {time_unit}");
        aggregated_df.show(5);
        anomalies_df = identify_anomalies(aggregated_df, threshold,time_unit);
        logging.info(f"Identify Anomalies with threshold: {threshold}");
        # anomalies_df.show(5);
        save_output(anomalies_df, output_filepath, file)

def main():
    """Main function to orchestrate the data processing using Spark."""
    args = parse_arguments()
    spark = create_spark_session(args.app_name, args.master_node_url)
    process_files(spark, args.input_filepath, args.output_filepath,args.location,args.start_date,args.end_date,args.time_unit,args.threshold);
    stopSparkSession(spark)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process weather data.")
    parser.add_argument("--master_node_url", default="local[*]", help="URL of the master node.")
    parser.add_argument("--app_name", default="Precipitation Outliers", help="Name of the Spark application.")
    parser.add_argument("--input_filepath", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output8/", help="Input directory path.")
    parser.add_argument("--output_filepath", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output9/", help="Output directory path.")
    parser.add_argument("--location",default="CA001015630", help="Geographic location (station ID).")
    parser.add_argument("--start_date",default="20240101" , help="Start date for the period of interest (yyyy-MM-dd).")
    parser.add_argument("--end_date",default="20240331" , help="End date for the period of interest (yyyy-MM-dd).")
    parser.add_argument("--time_unit", choices=['daily', 'monthly', 'yearly'], default='monthly', help="Time unit for aggregation.")
    parser.add_argument("--threshold",default=100 ,type=float, help="Threshold value for detecting rainfall anomalies.")
    return parser.parse_args()

if __name__ == "__main__":
    main()