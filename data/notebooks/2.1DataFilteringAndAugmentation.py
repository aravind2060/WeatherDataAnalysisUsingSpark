import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_spark_session():
    """
    Create a SparkSession instance.
    """
    spark = SparkSession.builder \
        .appName("DataAugmentation") \
        .getOrCreate()
    return spark

def load_data(spark, input_path):
    """
    Load the input data from the specified path.
    """
    logging.info(f"Loading input data from {input_path}")
    schema = "StationID STRING, Date INT, ElementType STRING, Value INT"
    data = spark.read.csv(input_path, header=False, schema=schema)
    return data

def load_geographic_metadata(spark, metadata_path):
    """
    Load the geographic metadata from the specified path.
    """
    logging.info(f"Loading geographic metadata from {metadata_path}")
    schema = "StationID STRING, StateName STRING, LocationName STRING, Country STRING"
    metadata = spark.read.csv(metadata_path, sep=",", header=False, schema=schema)
    return metadata

def augment_data(data, metadata):
    """
    Augment the input data with geographic metadata.
    """
    logging.info("Augmenting data with geographic metadata")
    augmented_data = data.join(metadata, on="StationID", how="left") \
        .select(
            col("StationID").alias("station_id"),
            col("Date"),
            col("ElementType"),
            col("Value"),
            col("StateName"),
            col("LocationName"),
            col("Country")
        )
    return augmented_data

def save_output(augmented_data, output_path):
    """
    Save the augmented data to the specified output path.
    """
    logging.info(f"Saving output to {output_path}")
    augmented_data.coalesce(1).write.option("quote", "").csv(output_path, header=False)

def main():
    """
    Main function to run the data augmentation process.
    """
    spark = create_spark_session()
    input_path = "/workspaces/WeatherDataAnalysisUsingSpark/data/output/2024.csv"
    metadata_path = "/workspaces/WeatherDataAnalysisUsingSpark/data/input/metadatafiles/ghcnd_stations_updated.csv"
    output_path = "/workspaces/WeatherDataAnalysisUsingSpark/data/output/new/"

    data = load_data(spark, input_path)
    metadata = load_geographic_metadata(spark, metadata_path)
    augmented_data = augment_data(data, metadata)
    save_output(augmented_data, output_path)

    spark.stop()

if __name__ == "__main__":
    main()