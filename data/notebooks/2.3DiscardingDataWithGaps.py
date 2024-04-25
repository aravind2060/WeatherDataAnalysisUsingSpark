import logging
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, lag, collect_list, explode
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, DateType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_spark_session():
    """Create a SparkSession."""
    logging.info("Creating SparkSession")
    return SparkSession.builder.appName("ElementGapAnalysis").getOrCreate()

def load_input_data(spark, input_path):
    """Load the input data and define the schema."""
    logging.info(f"Loading input data from {input_path}")
    schema = StructType([
        StructField("StationID", StringType(), True),
        StructField("Date", StringType(), True),
        StructField("ElementType", StringType(), True),
        StructField("Value", StringType(), True),
        StructField("StateName", StringType(), True),
        StructField("LocationName", StringType(), True),
        StructField("Country", StringType(), True)
    ])

    df = spark.read.csv(input_path, schema=schema, header=False)
    df = df.withColumn("DateCol", to_date(col("Date"), "yyyyMMdd"))
    return df

def detect_gaps(df):
    """Detect and process gaps greater than 3 days."""
    logging.info("Detecting gaps in data")
    window_spec = Window.partitionBy("StationID", "ElementType").orderBy("DateCol")
    df = df.withColumn("PrevDate", lag("DateCol").over(window_spec))
    df = df.withColumn("Gap", (col("DateCol").cast("long") - col("PrevDate").cast("long")) / 86400 > 3)
    return df.filter(col("Gap") == False)

def save_output(df, output_path):
    """Save the output data."""
    logging.info(f"Saving output data to {output_path}")
    df.write.csv(output_path, mode="overwrite", header=True)

def main(input_path, output_path):
    spark = create_spark_session()
    df = load_input_data(spark, input_path)
    gap_df = detect_gaps(df)
    save_output(gap_df, output_path)
    spark.stop()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output3/", help="Input file path")
    parser.add_argument("--output_path", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output4/", help="Output file path")
    args = parser.parse_args()
    main(args.input_path, args.output_path)
