import logging
import argparse
import os
import shutil
from pyspark.sql import SparkSession,Row
from pyspark.sql.functions import *
from pyspark.sql.functions import col, trim, upper, regexp_replace, when, substring, length, coalesce, lit,first,max,concat_ws,split,avg,collect_list,concat,last,mean, stddev, abs,to_date,sum as sql_sum,date_format
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

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

def load_data(spark, input_path, schema , headerr=True):
    """Load data from the given path using the provided schema, ensuring StationID is trimmed and consistent."""
    logging.info(f"Loading data from: {input_path}")
    data = spark.read.csv(input_path, header=headerr, schema=schema)
    data = clean_data(data)
    data = data.withColumn("AverageTemperature" , col("AvgTemp").cast(DoubleType()))
    logging.info("Data loaded and formatted correctly.")
    return data

def clean_data(data):
    """Remove non-printable characters and other unwanted artifacts from data."""
    logging.info("Cleaning data by removing non-printable characters.")
    for column in data.columns:
        data = data.withColumn(column, regexp_replace(col(column), "[^\x20-\x7E]", ""))
    return data

def identify_long_term_trends(spark, df, feature_col):
    """
    Identify long-term trends using linear regression on yearly data.

    Args:
    spark (SparkSession): Spark session object.
    df (DataFrame): Input Spark DataFrame with at least 'MonthYear' and 'feature_col'.
    feature_col (str): Column name of the feature to analyze.

    Returns:
    DataFrame: DataFrame with trend information for each state or None if an error occurs.
    """
    try:
        # Ensure there is enough data variability
        year_count = df.select(substring(col("MonthYear"), 1, 4).alias("Year")).distinct().count();
        logging.info(f" yyear count:  {year_count}");
        if year_count < 2:
            logging.error("Not enough data variability in 'Year' for trend analysis.")
            return None

        # Convert 'MonthYear' to 'Year' and cast to integer
        df = df.withColumn("Year", year(to_date(col("MonthYear"), "yyyy-MM")))

        # Assemble features for linear regression
        assembler = VectorAssembler(inputCols=["Year"], outputCol="features")
        df = assembler.transform(df)

        # Fit linear regression model
        lr = LinearRegression(featuresCol="features", labelCol=feature_col)
        model = lr.fit(df)

        # Get the slope of the regression line as the trend
        trend = float(model.coefficients[0])

        # Create a DataFrame with trend results
        trend_df = spark.createDataFrame(
            [Row(StateName=row['StateName'], Trend=trend) for row in df.select("StateName").distinct().collect()],
            ["StateName", "Trend"]
        )

        return trend_df

    except Exception as e:
        logging.error(f"Failed to identify trends: {str(e)}")
        return None



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
        """Example""" """CANADA,ALBERTA,2024-02,-5.492969396195202,-93.43672456575682,-49.62737799834574,99.0,-1.0"""
        data_schema = "Country STRING,StateName STRING,MonthYear STRING,AvgMaxTemp Double,AvgMinTemp Double,AvgTemp Double,MaxTemp Double,MinTemp Double"

        df = load_data(spark, input_filepath, data_schema)
        df.show(5)
        temp_df = identify_long_term_trends(spark,df,"AverageTemperature");
        if temp_df is not None:
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
    parser.add_argument("--input_filepath", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output10/", help="Input directory path.")
    parser.add_argument("--output_filepath", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output12/", help="Output directory path.")
    return parser.parse_args()

if __name__ == "__main__":
    main()