from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when
import sys

# Python 2 compatibility (HDP usually uses Python 2)
if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding('utf-8')

# 1. Spark Session with Hive
spark = SparkSession.builder \
    .appName("IUCN_Conservation_Prediction") \
    .master("yarn") \
    .config("spark.sql.catalogImplementation", "hive") \
    .config("hive.metastore.uris", "thrift://sandbox-hdp.hortonworks.com:9083") \
    .enableHiveSupport() \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Spark version:", spark.version)

# 2. Load Hive table
df = spark.table("default.threatened_species")

# Include scientific_name (you use it later)
df = df.select(
    "scientific_name",
    "kingdom_name", "phylum_name", "class_name",
    "order_name", "family_name", "genus_name",
    "category"
).na.drop(subset=["category"])

# 3. Columns
categorical_cols = [
    "kingdom_name", "phylum_name", "class_name",
    "order_name", "family_name", "genus_name"
]

# 4. Indexers
indexers = [
    StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep")
    for c in categorical_cols
]

# 5. OneHotEncoder (⚠️ SINGLE COLUMN ONLY for Spark 2.2)
encoders = [
    OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_vec")
    for c in categorical_cols
]

# 6. Label indexer
label_indexer = StringIndexer(
    inputCol="category",
    outputCol="label",
    handleInvalid="keep"
)

# 7. Assemble features
assembler = VectorAssembler(
    inputCols=[c+"_vec" for c in categorical_cols],
    outputCol="features"
)

# 8. Model
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=50,
    maxDepth=10,
    seed=42
)

# 9. Pipeline (FIXED)
pipeline = Pipeline(
    stages=indexers + encoders + [label_indexer, assembler, rf]
)

# 10. Train/Test Split
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# 11. Train
model = pipeline.fit(train_data)

# 12. Predict
predictions = model.transform(test_data)

# 13. Convert prediction → category
label_model = None
for stage in model.stages:
    if hasattr(stage, "getOutputCol") and stage.getOutputCol() == "label":
        label_model = stage
        break

labels = label_model.labels

pred_col = when(col("prediction") == 0.0, labels[0])
for i in range(1, len(labels)):
    pred_col = pred_col.when(col("prediction") == float(i), labels[i])

predictions = predictions.withColumn(
    "predicted_category",
    pred_col.otherwise("UNKNOWN")
)

# 14. Show results
predictions.select(
    "scientific_name", "category", "predicted_category"
).show(20, False)

# 15. Evaluation
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction"
)

accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
f1 = evaluator.setMetricName("f1").evaluate(predictions)
precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)

print("\n=== MODEL METRICS ===")
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)

# 16. Save model
model.save("hdfs:///user/maria_dev/sga6/iucn_rf_model")

spark.stop()
