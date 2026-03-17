from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline

# Create Spark session with Hadoop configs for local mode
spark = SparkSession.builder \
    .appName("ThreatenedSpeciesML") \
    .config("spark.hadoop.fs.defaultFS", "file:///") \
    .config("spark.hadoop.hadoop.security.authentication", "simple") \
    .config("spark.sql.warehouse.dir", "file:///C:/tmp/warehouse") \
    .getOrCreate()

# Read CSV using file:/// URI
df = spark.read.option("header", "true").option("inferSchema", "true") \
         .csv("file:///C:/Users/kohwa/OneDrive/Desktop/Exam/Big Data Analytics/threatened-species.csv")

# Rest of your code (unchanged)...
df = df.withColumn("is_threatened", when(col("category").isin("CR","EN","VU"), 1).otherwise(0))
df = df.fillna({"population": "unknown"})

categorical_cols = ["kingdom_name", "phylum_name", "class_name", "order_name", 
                    "family_name", "genus_name", "population"]

indexers = [StringIndexer(inputCol=col, outputCol=col+"_idx", handleInvalid="keep") for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=col+"_idx", outputCol=col+"_vec") for col in categorical_cols]
assembler = VectorAssembler(inputCols=[col+"_vec" for col in categorical_cols], outputCol="features")
rf = RandomForestClassifier(labelCol="is_threatened", featuresCol="features", numTrees=100)

pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

train, test = df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train)
predictions = model.transform(test)

accuracy = MulticlassClassificationEvaluator(labelCol="is_threatened", metricName="accuracy").evaluate(predictions)
auc = BinaryClassificationEvaluator(labelCol="is_threatened", rawPredictionCol="rawPrediction").evaluate(predictions)

print(f"Test Accuracy = {accuracy:.4f}")
print(f"Test AUC = {auc:.4f}")