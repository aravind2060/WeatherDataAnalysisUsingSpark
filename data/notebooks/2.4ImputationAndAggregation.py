import logging
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, coalesce

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def create_spark_session():
    """Create a Spark session."""
    logging.info("Creating Spark session")
    return SparkSession.builder.appName("TemperatureImputation").getOrCreate()

def load_input_data(spark, input_path):
    """Load input data with specified schema."""
    logging.info(f"Loading input data from {input_path}")
    df = spark.read.option("header", "false").csv(input_path)
    df = df.select(
        col("_c0").alias("StationID"),
        col("_c1").alias("Date"),
        col("_c2").alias("ElementType"),
        col("_c3").cast("integer").alias("Value"),
        col("_c4").alias("StateName"),
        col("_c5").alias("LocationName"),
        col("_c6").alias("Country")
    )
    return df

def impute_temperature(df):
    """Impute missing temperatures based on available data."""
    logging.info("Imputing missing temperature data")
    df.show(5);
    df = df.withColumn("ImputedValue", when(
        (col("ElementType").isin(["TMIN", "TMAX", "TAVG"]) & (col("Value") == 0)) |
        col("Value").isNull(),
        when(col("ElementType") == "TMIN", (2 * col("TAVG") - col("TMAX")).cast("integer"))
         .when(col("ElementType") == "TMAX", (2 * col("TAVG") - col("TMIN")).cast("integer"))
         .when(col("ElementType") == "TAVG", ((col("TMIN") + col("TMAX")) / 2).cast("integer"))
    ).otherwise(col("Value")))
    return df.select("StationID", "Date", "ElementType", "ImputedValue", "StateName", "LocationName", "Country")

def save_output(df, output_path):
    """Save the output data."""
    logging.info(f"Saving output data to {output_path}")
    df.show(5);
    df.coalesce(1).write.option("quote", "").csv(output_path, header=False)

def main(input_path, output_path):
    spark = create_spark_session()
    df = load_input_data(spark, input_path)
    imputed_df = impute_temperature(df)
    save_output(imputed_df, output_path)
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output4/2024.csv", help="Input file path")
    parser.add_argument("--output_path", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output5/", help="Output file path")
    args = parser.parse_args()
    main(args.input_path, args.output_path)