from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when
import sys

# Only needed if you're on Python 2; remove if on Python 3
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

# OPTIONAL: For quick testing, limit rows (remove for full training)
# df = df.limit(500)  # Uncomment if you want a smaller sample

# Inspect data
print("Schema:")
df.printSchema()
print("First 5 rows:")
df.show(5, truncate=False)
print("Distribution of target variable (category):")
df.groupBy("category").count().orderBy("category").show()

# 3. Data preprocessing
categorical_cols = ["kingdom_name", "phylum_name", "class_name", 
                    "order_name", "family_name", "genus_name"]
target_col = "category"

# Drop rows where target is null
df = df.na.drop(subset=[target_col])

# StringIndexer for each categorical column
indexers = [StringIndexer(inputCol=col, outputCol=col + "_idx", 
                           handleInvalid="keep") for col in categorical_cols]

# OneHotEncoder for indexed columns
# Use OneHotEncoderEstimator (correct for Spark 2.3)
encoder = OneHotEncoderEstimator(
    inputCols=[col + "_idx" for col in categorical_cols],
    outputCols=[col + "_vec" for col in categorical_cols]
)

# Assemble all encoded vectors into a single feature vector
assembler = VectorAssembler(
    inputCols=[col + "_vec" for col in categorical_cols],
    outputCol="features"
)

# Index the target variable
label_indexer = StringIndexer(inputCol=target_col, outputCol="label", 
                              handleInvalid="keep")

# 4. Split data into training and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
print("Training records: {}".format(train_data.count()))
print("Test records: {}".format(test_data.count()))

# 5. Define Random Forest classifier
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=50,          # Reduced for faster training; increase later
    maxDepth=10,          # Adjust as needed
    impurity="gini",
    seed=42
)

# 6. Build pipeline
pipeline = Pipeline(stages=indexers + [encoder, assembler, label_indexer, rf])

# 7. Train model
print("Training Random Forest model...")
model = pipeline.fit(train_data)

# 8. Make predictions
predictions = model.transform(test_data)

# 8a. Debug: Print fitted pipeline stages (optional, remove after verification)
print("Fitted pipeline stages:")
for i, stage in enumerate(model.stages):
    print(f"Stage {i}: {stage.__class__.__name__}")
    if hasattr(stage, 'getOutputCol'):
        print(f"  outputCol = {stage.getOutputCol()}")

# 8b. Convert numeric predictions back to original category names
# Robust retrieval: look for stage with outputCol='label'
label_indexer_model = None
for stage in model.stages:
    if hasattr(stage, 'getOutputCol') and stage.getOutputCol() == 'label':
        label_indexer_model = stage
        break

# Fallback: if not found, assume it's the second-last stage (common in this pipeline)
if label_indexer_model is None:
    print("Warning: Using fallback (stage -2) to locate label indexer.")
    label_indexer_model = model.stages[-2]

labels = label_indexer_model.labels

# Build when-otherwise expression for predicted category
pred_category_col = when(col("prediction") == 0.0, labels[0])
for i in range(1, len(labels)):
    pred_category_col = pred_category_col.when(col("prediction") == float(i), labels[i])
pred_category_col = pred_category_col.otherwise("UNKNOWN")

predictions = predictions.withColumn("predicted_category", pred_category_col)

print("Sample predictions (true vs predicted):")
predictions.select("scientific_name", "category", "predicted_category").show(20, truncate=False)

# 9. Evaluate model
evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1"
)
evaluator_precision = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
)
evaluator_recall = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedRecall"
)

accuracy = evaluator_acc.evaluate(predictions)
f1 = evaluator_f1.evaluate(predictions)
precision = evaluator_precision.evaluate(predictions)
recall = evaluator_recall.evaluate(predictions)

print("\n" + "="*50)
print("MODEL EVALUATION METRICS")
print("="*50)
print("Accuracy:  {:.4f} ({:.2f}%)".format(accuracy, accuracy*100))
print("F1 Score (weighted): {:.4f}".format(f1))
print("Precision (weighted): {:.4f}".format(precision))
print("Recall (weighted):    {:.4f}".format(recall))

# 10. Feature importance
rf_model = model.stages[-1]
importances = rf_model.featureImportances

print("\nTop 10 most important feature indices:")
top_indices = importances.indices[:10]
top_values = importances.values[:10]
for idx, val in zip(top_indices, top_values):
    print("  Index {}: importance = {:.4f}".format(idx, val))

# 11. Save model
model.save("hdfs:///user/maria_dev/sga6/iucn_rf_model")
print("\nModel saved to HDFS.")

spark.stop()
