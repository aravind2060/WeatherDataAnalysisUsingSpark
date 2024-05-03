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

def filter_data_by_stationId_stateName_and_time(df,stationId ,stateName, start_date, end_date):
    """Filter data by geographic location and date range."""
    logging.info("Filtering data by location and time.")
    return df.filter(
        (col('StationID') == stationId) &
        (col('StateName') == stateName) &
        (col('Date') >= start_date) &
        (col('Date') <= end_date)
    )

def aggregate_temperature(df, time_unit):
    """
    Aggregate Temperature data based on the given time unit ('daily', 'monthly', 'yearly') and include StationID.
    """
    logging.info("Aggregating Temperature data by time_unit %s.", time_unit)

    if time_unit == 'monthly':
        df = df.withColumn('Month', date_format(to_date('Date', 'yyyyMMdd'), 'yyyy-MM'))
        group_col = ['StationID', 'Month']
    elif time_unit == 'daily':
        df = df.withColumn('Date', to_date('Date', 'yyyyMMdd'))
        group_col = ['StationID', 'Date']
    elif time_unit == 'yearly':
        df = df.withColumn('Year', date_format(to_date('Date', 'yyyyMMdd'), 'yyyy'))
        group_col = ['StationID', 'Year']
    else:
        raise ValueError("Unsupported time unit. Use 'daily', 'monthly', or 'yearly'.")

    # Perform aggregation
    aggregated_df = df.groupBy(group_col).agg(
        avg('TMIN').alias('TMINAVG'),
        avg("TMAX").alias("TMAXAVG"),
        avg("TAVG").alias("TAvgAvg")
    )
    
    sort_column = group_col[1]  # Sorting by the time unit column

    sorted_df = aggregated_df.orderBy(sort_column)  # Sorting by the appropriate column
    
    return sorted_df

def compare_temperatures(location1_df, location2_df, time_unit):
    """
    Compare temperatures between two locations and calculate the differences,
    filling missing temperature values with zero.
    """
    logging.info("Comparing temperatures between two locations.")

    # Identify the correct column for joining based on time_unit
    if time_unit == 'monthly':
        join_col = 'Month'
    elif time_unit == 'daily':
        join_col = 'Date'
    elif time_unit == 'yearly':
        join_col = 'Year'
    else:
        raise ValueError("Unsupported time unit. Use 'daily', 'monthly', or 'yearly'.")

    # Join the two DataFrames and calculate the temperature differences
    comparison_df = location1_df.join(location2_df, on=join_col, how="full_outer") \
        .select(
            join_col,
            coalesce(location1_df["TMINAVG"], lit(0)).alias("Location1_TMINAVG"),
            coalesce(location2_df["TMINAVG"], lit(0)).alias("Location2_TMINAVG"),
            coalesce(location1_df["TMAXAVG"], lit(0)).alias("Location1_TMAXAVG"),
            coalesce(location2_df["TMAXAVG"], lit(0)).alias("Location2_TMAXAVG"),
            coalesce(location1_df["TAvgAvg"], lit(0)).alias("Location1_TAvgAvg"),
            coalesce(location2_df["TAvgAvg"], lit(0)).alias("Location2_TAvgAvg")
        ) \
        .withColumn("TMINAVG_Diff", col("Location1_TMINAVG") - col("Location2_TMINAVG")) \
        .withColumn("TMAXAVG_Diff", col("Location1_TMAXAVG") - col("Location2_TMAXAVG")) \
        .withColumn("TAvgAvg_Diff", col("Location1_TAvgAvg") - col("Location2_TAvgAvg"))

    return comparison_df



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

def process_files(spark, input_dir, output_filepath,stationId1,stateName1,stationId2,stateName2,start_date,end_date,time_unit):
    """Process each file in the input directory and save it with the same name in the output directory."""
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    parent_df = [];
    for file in files:
        logging.info(f"Processing file: {file}")
        input_filepath = os.path.join(input_dir, file)
        """Example""" """CA001012475,20240209,50.0,67.0,59.0,,BRITISH COLUMBIA,DISCOVERY ISLAND,CANADA"""
        data_schema = "StationID STRING, Date STRING, TMIN Float, TMAX Float,TAVG Float,PRCP Float, StateName STRING, LocationName STRING,Country STRING"

        df = load_data(spark, input_filepath, data_schema)
        df.show(5)
        parent_df.append(df);
    
    # Merge all DataFrames into a single DataFrame
    merged_df = parent_df[0]
    for df in parent_df[1:]:
        merged_df = merged_df.union(df)
    
    filtered_df1 = filter_data_by_stationId_stateName_and_time(merged_df,stationId1,stateName1, start_date, end_date)
    logging.info(f"Filtered by station id {stationId1} and then statename: {stateName1}, start_date: {start_date} , end_date: {end_date} ");
    filtered_df1.show(5);
    aggregated_df1 = aggregate_temperature(filtered_df1, time_unit);
    logging.info(f"Aggregated Temperature based on time_unit: {time_unit}");
    aggregated_df1.show(5);
    
    
    
    filtered_df2 = filter_data_by_stationId_stateName_and_time(merged_df,stationId2,stateName2, start_date, end_date)
    logging.info(f"Filtered by station id {stationId2} and then statename: {stateName2}, start_date: {start_date} , end_date: {end_date} ");
    filtered_df2.show(5);
    aggregated_df2 = aggregate_temperature(filtered_df2, time_unit);
    logging.info(f"Aggregated Temperature based on time_unit: {time_unit}");
    aggregated_df2.show(5);
    
    logging.info(f"Finished Analysis of both ");
    save_output(aggregated_df1.union(aggregated_df2), output_filepath, file);
    
    logging.info(f"Data Comparing among two stations is started: ");
    compare_df = compare_temperatures(aggregated_df1,aggregated_df2,time_unit);  
    
    save_output(compare_df, output_filepath, "comparsion.csv")

def main():
    """Main function to orchestrate the data processing using Spark."""
    args = parse_arguments()
    spark = create_spark_session(args.app_name, args.master_node_url)
    process_files(spark, args.input_filepath, args.output_filepath,args.stationId1,args.stateName1,args.stationId2,args.stateName2,args.start_date,args.end_date,args.time_unit);
    stopSparkSession(spark)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process weather data.")
    parser.add_argument("--master_node_url", default="local[*]", help="URL of the master node.")
    parser.add_argument("--app_name", default="Temperature Analysis", help="Name of the Spark application.")
    parser.add_argument("--input_filepath", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output8/", help="Input directory path.")
    parser.add_argument("--output_filepath", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output13/", help="Output directory path.")
    parser.add_argument("--stationId1",default="USC00415409", help="Geographic location (station ID).")
    parser.add_argument("--stateName1",default="TEXAS", help="Geographic location (station ID).")
    parser.add_argument("--stationId2",default="USC00415429", help="Geographic location (station ID).")
    parser.add_argument("--stateName2",default="TEXAS", help="Geographic location (station ID).")
    parser.add_argument("--start_date",default="20230101" , help="Start date for the period of interest (yyyy-MM-dd).")
    parser.add_argument("--end_date",default="20240331" , help="End date for the period of interest (yyyy-MM-dd).")
    parser.add_argument("--time_unit", choices=['daily', 'monthly', 'yearly'], default='monthly', help="Time unit for aggregation.")
    return parser.parse_args()

if __name__ == "__main__":
    main()