from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, count, max as spark_max, min as spark_min, avg, year, row_number
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import logging
import os
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def initialize_spark_session(app_name="Heatwave and Cold Spell Analysis"):
    """ Initialize and return a Spark session """
    logging.info("Initializing Spark session")
    return SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()

def load_data(spark, file_path):
    """ Load data from CSV file into DataFrame with predefined schema """
    logging.info(f"Loading data from: {file_path}")
    schema = StructType([
        StructField("StationID", StringType(), True),
        StructField("Date", StringType(), True),
        StructField("TMIN", FloatType(), True),
        StructField("TMAX", FloatType(), True),
        StructField("TAVG", FloatType(), True),
        StructField("PRCP", FloatType(), True),
        StructField("StateName", StringType(), True),
        StructField("LocationName", StringType(), True),
        StructField("Country", StringType(), True)
    ])
    data = spark.read.option("header", "false").schema(schema).csv(file_path)
    data.createOrReplaceTempView("data")
    logging.info("Data loaded successfully")
    return data

def analyze_heatwaves(spark,data):
    """ Analyze heatwaves from the data """
    logging.info("Analyzing heatwaves")
    heatwave_df = spark.sql("""
        SELECT
            StationID AS station_id,
            YEAR(to_date(Date, 'yyyyMMdd')) AS year,
            MIN(to_date(Date, 'yyyyMMdd')) AS start_date,
            MAX(to_date(Date, 'yyyyMMdd')) AS end_date,
            COUNT(*) AS duration,
            MAX(TMAX) AS max_temperature
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (PARTITION BY StationID, YEAR(to_date(Date, 'yyyyMMdd')) ORDER BY to_date(Date, 'yyyyMMdd')) -
                ROW_NUMBER() OVER (PARTITION BY StationID, YEAR(to_date(Date, 'yyyyMMdd')) ORDER BY to_date(Date, 'yyyyMMdd')) AS grp
            FROM data
            WHERE TMAX > 250
        ) temp_groups
        GROUP BY StationID, YEAR(to_date(Date, 'yyyyMMdd')), grp
        HAVING COUNT(*) >= 3  
    """)
    heatwave_df.show()
    logging.info("Heatwaves analyzed")
    return heatwave_df

def analyze_cold_spells(spark,data):
    """ Analyze cold spells from the data """
    logging.info("Analyzing cold spells")
    cold_spell_df = spark.sql("""
        SELECT
            StationID AS station_id,
            YEAR(to_date(Date, 'yyyyMMdd')) AS year,
            MIN(to_date(Date, 'yyyyMMdd')) AS start_date,
            MAX(to_date(Date, 'yyyyMMdd')) AS end_date,
            COUNT(*) AS duration,
            MIN(TMIN) AS min_temperature
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (PARTITION BY StationID, YEAR(to_date(Date, 'yyyyMMdd')) ORDER BY to_date(Date, 'yyyyMMdd')) -
                ROW_NUMBER() OVER (PARTITION BY StationID, YEAR(to_date(Date, 'yyyyMMdd')) ORDER BY to_date(Date, 'yyyyMMdd')) AS grp
            FROM data
            WHERE TMIN < -200
        ) temp_groups
        GROUP BY StationID, YEAR(to_date(Date, 'yyyyMMdd')), grp
        HAVING COUNT(*) >= 3
    """)
    cold_spell_df.show()
    logging.info("Cold spells analyzed")
    return cold_spell_df

def calculate_heatwave_metrics(heatwave_df):
    """ Calculate metrics from the heatwave data """
    logging.info("Calculating heatwave metrics")
    heatwave_frequency = heatwave_df.count()
    heatwave_intensity = heatwave_df.selectExpr("avg(duration) as avg_duration", "max(max_temperature) as max_temperature").head()
    logging.info("Heatwave metrics calculated")
    return heatwave_frequency, heatwave_intensity

def calculate_cold_spell_metrics(cold_spell_df):
    """ Calculate metrics from the cold spell data """
    logging.info("Calculating cold spell metrics")
    cold_spell_frequency = cold_spell_df.count()
    cold_spell_intensity = cold_spell_df.selectExpr("avg(duration) as avg_duration", "min(min_temperature) as min_temperature").head()
    logging.info("Cold spell metrics calculated")
    return cold_spell_frequency, cold_spell_intensity

def aggregate_results(heatwave_df, cold_spell_df):
    """ Aggregate results from both heatwave and cold spell analyses """
    logging.info("Aggregating results")
    aggregated_heatwaves = heatwave_df.groupBy("station_id").agg(
        count("*").alias("heatwave_frequency"),
        avg("duration").alias("avg_heatwave_duration"),
        spark_max("max_temperature").alias("max_heatwave_temperature")
    )
    aggregated_cold_spells = cold_spell_df.groupBy("station_id").agg(
        count("*").alias("cold_spell_frequency"),
        avg("duration").alias("avg_cold_spell_duration"),
        spark_min("min_temperature").alias("min_cold_spell_temperature")
    )
    aggregated_heatwaves.show()
    aggregated_cold_spells.show()
    logging.info("Results aggregated")
    return aggregated_heatwaves, aggregated_cold_spells

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

def save_results_to_csv(output_path,heatwave_df, cold_spell_df, aggregated_heatwaves, aggregated_cold_spells):
    """ Save the analysis results to CSV files """
    logging.info("Saving results to CSV")
    save_output(heatwave_df,output_path,"heatwave_results.csv");
    save_output(cold_spell_df,output_path,"cold_spell_results.csv.csv");
    save_output(aggregated_heatwaves,output_path,"aggregated_heatwaves.csv")
    save_output(aggregated_cold_spells,output_path,"aggregated_cold_spells.csv");
    logging.info("Results saved to CSV")

def print_results(heatwave_frequency, heatwave_intensity, cold_spell_frequency, cold_spell_intensity):
    """ Print the results of the analyses """
    logging.info("Printing results")
    print("Heatwave Frequency: " + str(heatwave_frequency))
    print("Heatwave Intensity (Avg Duration, Max Temperature): " + str(heatwave_intensity))
    print("Cold Spell Frequency: " + str(cold_spell_frequency))
    print("Cold Spell Intensity (Avg Duration, Min Temperature): " + str(cold_spell_intensity))
    logging.info("Results printed")

def main():
    """ Main function to orchestrate the workflow """
    spark = initialize_spark_session()
    data = load_data(spark, "/workspaces/WeatherDataAnalysisUsingSpark/data/output8/2024.csv")
    heatwave_df = analyze_heatwaves(spark,data)
    heatwave_frequency, heatwave_intensity = calculate_heatwave_metrics(heatwave_df)
    cold_spell_df = analyze_cold_spells(spark,data)
    cold_spell_frequency, cold_spell_intensity = calculate_cold_spell_metrics(cold_spell_df)
    aggregated_heatwaves, aggregated_cold_spells = aggregate_results(heatwave_df, cold_spell_df)
    print_results(heatwave_frequency, heatwave_intensity, cold_spell_frequency, cold_spell_intensity)
    save_results_to_csv("/workspaces/WeatherDataAnalysisUsingSpark/data/output15/",heatwave_df, cold_spell_df, aggregated_heatwaves, aggregated_cold_spells)
    spark.stop()

if __name__ == "__main__":
    main()
