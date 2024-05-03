from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, count, max as spark_max, min as spark_min, avg, year, row_number
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, FloatType

def initialize_spark_session(app_name="Heatwave and Cold Spell Analysis"):
    return SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()

def load_data(spark, file_path):
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
    return data

def analyze_heatwaves(data):
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
    heatwave_df.createOrReplaceTempView("heatwave_df")
    heatwave_df.show()
    return heatwave_df

def analyze_cold_spells(data):
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
    cold_spell_df.createOrReplaceTempView("cold_spell_df")
    cold_spell_df.show()
    return cold_spell_df

def calculate_heatwave_metrics(heatwave_df):
    heatwave_frequency = heatwave_df.count()
    heatwave_intensity = heatwave_df.selectExpr("avg(duration) as avg_duration", "max(max_temperature) as max_temperature").head()
    return heatwave_frequency, heatwave_intensity

def calculate_cold_spell_metrics(cold_spell_df):
    cold_spell_frequency = cold_spell_df.count()
    cold_spell_intensity = cold_spell_df.selectExpr("avg(duration) as avg_duration", "min(min_temperature) as min_temperature").head()
    return cold_spell_frequency, cold_spell_intensity

def aggregate_results(heatwave_df, cold_spell_df):
    aggregated_heatwaves = heatwave_df.groupBy("station_id").agg(
        count("*").alias("heatwave_frequency"),
        avg("duration").alias("avg_heatwave_duration"),
        spark_max("max_temperature").alias("max_heatwave_temperature")
    )
    aggregated_heatwaves.show()
    aggregated_cold_spells = cold_spell_df.groupBy("station_id").agg(
        count("*").alias("cold_spell_frequency"),
        avg("duration").alias("avg_cold_spell_duration"),
        spark_min("min_temperature").alias("min_cold_spell_temperature")
    )
    aggregated_cold_spells.show()
    return aggregated_heatwaves, aggregated_cold_spells

def save_results_to_csv(heatwave_df, cold_spell_df, aggregated_heatwaves, aggregated_cold_spells):
    heatwave_df.toPandas().to_csv("/workspaces/WeatherDataAnalysisUsingSpark/data/output15/heatwave_results.csv", index=False)
    cold_spell_df.toPandas().to_csv("/workspaces/WeatherDataAnalysisUsingSpark/data/output15/cold_spell_results.csv", index=False)
    aggregated_heatwaves.toPandas().to_csv("/workspaces/WeatherDataAnalysisUsingSpark/data/output15/aggregated_heatwaves.csv", index=False)
    aggregated_cold_spells.toPandas().to_csv("/workspaces/WeatherDataAnalysisUsingSpark/data/output15/aggregated_cold_spells.csv", index=False)


def print_results(heatwave_frequency, heatwave_intensity, cold_spell_frequency, cold_spell_intensity):
    print("Heatwave Frequency: " + str(heatwave_frequency))
    print("Heatwave Intensity (Avg Duration, Max Temperature): " + str(heatwave_intensity))
    print("Cold Spell Frequency: " + str(cold_spell_frequency))
    print("Cold Spell Intensity (Avg Duration, Min Temperature): " + str(cold_spell_intensity))

# Initialize Spark session
spark = initialize_spark_session()

# Load data
data = load_data(spark, "/workspaces/WeatherDataAnalysisUsingSpark/data/output8/2024.csv")

# Analyze heatwaves
heatwave_df = analyze_heatwaves(data)
heatwave_frequency, heatwave_intensity = calculate_heatwave_metrics(heatwave_df)

# Analyze cold spells
cold_spell_df = analyze_cold_spells(data)
cold_spell_frequency, cold_spell_intensity = calculate_cold_spell_metrics(cold_spell_df)

# Aggregate results
aggregated_heatwaves, aggregated_cold_spells = aggregate_results(heatwave_df, cold_spell_df)

# Print and display results
print_results(heatwave_frequency, heatwave_intensity, cold_spell_frequency, cold_spell_intensity)

# Save results to CSV
save_results_to_csv(heatwave_df, cold_spell_df, aggregated_heatwaves, aggregated_cold_spells)

# Stop Spark session
spark.stop()
