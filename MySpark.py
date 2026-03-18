# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when
import sys

# Python 2 compatibility
if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding('utf-8')

# 1. Initialize SparkSession with Hive support
spark = SparkSession.builder \
    .appName("IUCN_Conservation_Prediction") \
    .master("yarn") \
    .config("spark.sql.catalogImplementation", "hive") \
    .config("hive.metastore.uris", "thrift://sandbox-hdp.hortonworks.com:9083") \
    .enableHiveSupport() \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Spark version: {}".format(spark.version))

# 2. Load data from Hive table
print("Loading data from Hive table default.threatened_species...")
df = spark.table("default.threatened_species")

# Select required columns
df = df.select(
    "scientific_name",
    "kingdom_name", "phylum_name", "class_name",
    "order_name", "family_name", "genus_name",
    "category"
).na.drop(subset=["category"])

# Inspect data
print("Schema:")
df.printSchema()

print("First 5 rows:")
df.show(5, truncate=False)

print("Distribution of target variable (category):")
df.groupBy("category").count().orderBy("category").show()

# 3. Data preprocessing
categorical_cols = [
    "kingdom_name", "phylum_name", "class_name",
    "order_name", "family_name", "genus_name"
]

# StringIndexer for categorical columns
indexers = [
    StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
    for c in categorical_cols
]

# OneHotEncoder (Spark 2 compatible - single column each)
encoders = [
    OneHotEncoder(inputCol=c + "_idx", outputCol=c + "_vec")
    for c in categorical_cols
]

# Label indexer
label_indexer = StringIndexer(
    inputCol="category",
    outputCol="label",
    handleInvalid="keep"
)

# Assemble features
assembler = VectorAssembler(
    inputCols=[c + "_vec" for c in categorical_cols],
    outputCol="features"
)

# 4. Train-test split
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
print("Training records: {}".format(train_data.count()))
print("Test records: {}".format(test_data.count()))

# 5. Random Forest model
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=50,
    maxDepth=10,
    seed=42
)

# 6. Pipeline
pipeline = Pipeline(
    stages=indexers + encoders + [label_indexer, assembler, rf]
)

# 7. Train model
print("Training Random Forest model...")
model = pipeline.fit(train_data)

# 8. Predictions
predictions = model.transform(test_data)

# 9. Convert prediction to category label
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

print("Sample predictions:")
predictions.select(
    "scientific_name", "category", "predicted_category"
).show(20, False)

# 10. Evaluation
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction"
)

accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
f1 = evaluator.setMetricName("f1").evaluate(predictions)
precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)

print("\n=== MODEL METRICS ===")
print("Accuracy: {:.4f}".format(accuracy))
print("F1 Score: {:.4f}".format(f1))
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))

# 11. Feature importance
rf_model = model.stages[-1]
importances = rf_model.featureImportances

print("\nTop feature importance indices:")
for i in range(min(10, len(importances.indices))):
    print("Index {}: {:.4f}".format(
        importances.indices[i],
        importances.values[i]
    ))

# 12. Save model
model.save("hdfs:///user/maria_dev/sga6/iucn_rf_model")
print("\nModel saved to HDFS.")

spark.stop()
