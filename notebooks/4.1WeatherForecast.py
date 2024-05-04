import logging
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, dayofyear, lag, regexp_replace, upper, trim
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def create_spark_session(app_name: str) -> SparkSession:
    """Initialize and return a Spark session with the specified application name."""
    logging.info(f"Creating Spark session with app name: {app_name}")
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_data(spark: SparkSession, input_path: str, schema: StructType, has_header: bool = False) -> DataFrame:
    """Load data from the given path using the provided schema, ensuring StationID is trimmed and consistent."""
    logging.info(f"Loading data from: {input_path}")
    data = spark.read.csv(input_path, header=has_header, schema=schema)
    data = clean_data(data)
    return data

def clean_data(data: DataFrame) -> DataFrame:
    """Remove non-printable characters and other unwanted artifacts from data."""
    logging.info("Cleaning data by removing non-printable characters.")
    for column in data.columns:
        data = data.withColumn(column, regexp_replace(col(column), "[^\x20-\x7E]", ""))
    return data

def prepare_features(df: DataFrame) -> DataFrame:
    """Prepare features for machine learning model."""
    df = df.withColumn("Date", to_date(col("Date"), "yyyyMMdd"))
    df = df.withColumn("day_of_year", dayofyear(col("Date")))
    df = df.withColumn("TMAX", col("TMAX").cast("float"))
    assembler = VectorAssembler(inputCols=["TMAX", "day_of_year"], outputCol="features")
    return assembler.transform(df)

def train_and_evaluate(df: DataFrame) -> dict:
    """Train and evaluate machine learning models."""
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    models = {'Linear Regression': LinearRegression(featuresCol="features", labelCol="TMAX"),
              'Decision Tree': DecisionTreeRegressor(featuresCol="features", labelCol="TMAX"),
              'Random Forest': RandomForestRegressor(featuresCol="features", labelCol="TMAX")}
    evaluator = RegressionEvaluator(labelCol="TMAX", predictionCol="prediction", metricName="rmse")
    results = {}

    for name, model in models.items():
        trained_model = model.fit(train)
        predictions = trained_model.transform(test)
        rmse = evaluator.evaluate(predictions)
        results[name] = {'model': trained_model, 'rmse': rmse}
        logging.info(f"{name} RMSE: {rmse}")

    return results

def save_results(results: dict, output_path: str):
    """Save the RMSE results to a CSV file."""
    with open(output_path, "w") as file:
        for model, info in results.items():
            file.write(f"{model},{info['rmse']}\n")
    logging.info(f"Results saved to {output_path}")

def main(filepath: str, output_path: str):
    spark = create_spark_session("Weather Forecast Analysis")
    data_schema = StructType([
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
    df = load_data(spark, filepath, data_schema)
    df = df.select("StationID", "Date", "TMAX", "PRCP").dropna()
    df = prepare_features(df)
    results = train_and_evaluate(df)
    save_results(results, output_path)
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather Forecast Analysis")
    parser.add_argument("--filepath", required=False,default="/workspaces/WeatherDataAnalysisUsingSpark/data/output8/", help="Path to the CSV file containing weather data.")
    parser.add_argument("--outputpath", default="/workspaces/WeatherDataAnalysisUsingSpark/data/output17/", help="Path to save the output CSV file.")
    args = parser.parse_args()
    main(args.filepath, args.outputpath)
