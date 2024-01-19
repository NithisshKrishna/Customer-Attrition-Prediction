# spark_data_pull.py
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("DataPullApp").getOrCreate()

# Read data into Spark DataFrame
data_path = "path_to_your_data.csv"  # Update this with the actual path
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Show the DataFrame (for demonstration)
df.show()
