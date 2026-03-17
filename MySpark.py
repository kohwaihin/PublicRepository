from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when

# 1. Initialize SparkSession (HDP 2.6.5 configuration)
spark = SparkSession.builder \
    .appName("IUCN_Conservation_Prediction") \
    .master("yarn") \
    .config("spark.sql.catalogImplementation", "hive") \
    .config("hive.metastore.uris", "thrift://sandbox-hdp.hortonworks.com:9083") \
    .enableHiveSupport() \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print(f"Spark version: {spark.version}")

# 2. Load data from HDFS CSV
file_path = "hdfs:///user/maria_dev/sga6/threatened-species.csv"
print(f"Loading data from {file_path}...")

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(file_path)

# Inspect data
print("Schema:")
df.printSchema()
print("First 5 rows:")
df.show(5, truncate=False)
print("Distribution of target variable (category):")
df.groupBy("category").count().orderBy("category").show()

# 3. Data preprocessing
# Select relevant columns and drop rows with missing target
categorical_cols = ["kingdom_name", "phylum_name", "class_name", 
                    "order_name", "family_name", "genus_name"]
target_col = "category"

# Drop rows where target is null
df = df.na.drop(subset=[target_col])

# Step 1: StringIndexer for each categorical column
indexers = [StringIndexer(inputCol=col, outputCol=col + "_idx", 
                           handleInvalid="keep") for col in categorical_cols]

# Step 2: OneHotEncoder for the indexed columns
encoder = OneHotEncoderEstimator(
    inputCols=[col + "_idx" for col in categorical_cols],
    outputCols=[col + "_vec" for col in categorical_cols]
)

# Step 3: Assemble all encoded vectors into a single feature vector
assembler = VectorAssembler(
    inputCols=[col + "_vec" for col in categorical_cols],
    outputCol="features"
)

# Step 4: Index the target variable (labels)
label_indexer = StringIndexer(inputCol=target_col, outputCol="label", 
                              handleInvalid="keep")

# 4. Split data into training and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
print(f"Training records: {train_data.count()}")
print(f"Test records: {test_data.count()}")

# 5. Define the classifier (Random Forest)
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=100,
    maxDepth=15,
    impurity="gini",
    seed=42
)

# 6. Build the pipeline
pipeline = Pipeline(stages=indexers + [encoder, assembler, label_indexer, rf])

# 7. Train the model
print("Training Random Forest model...")
model = pipeline.fit(train_data)

# 8. Make predictions on test data
predictions = model.transform(test_data)

# Convert numeric predictions back to original category names
# Retrieve the label mapping from the StringIndexer model
label_indexer_model = [stage for stage in model.stages if isinstance(stage, StringIndexer)][-1]
labels = label_indexer_model.labelsArray

# Create a new column with predicted category name
pred_category_col = when(col("prediction") == 0.0, labels[0])
for i in range(1, len(labels)):
    pred_category_col = pred_category_col.when(col("prediction") == float(i), labels[i])
pred_category_col = pred_category_col.otherwise("UNKNOWN")

predictions = predictions.withColumn("predicted_category", pred_category_col)

print("Sample predictions (true vs predicted):")
predictions.select("scientific_name", "category", "predicted_category").show(20, truncate=False)

# 9. Evaluate model performance
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
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"F1 Score (weighted): {f1:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted):    {recall:.4f}")

# 10. Feature importance
# The Random Forest model is the last stage.
rf_model = model.stages[-1]
importances = rf_model.featureImportances

# The feature vector is assembled from one-hot encoded vectors for each taxonomic rank.
# For simplicity, show the top 10 feature indices.
print("\nTop 10 most important feature indices:")
top_indices = importances.indices[:10]
top_values = importances.values[:10]
for idx, val in zip(top_indices, top_values):
    print(f"  Index {idx}: importance = {val:.4f}")

# 11. Save model for future use
model.save("hdfs:///user/maria_dev/sga6/iucn_rf_model")
print("\nModel saved to HDFS.")

# Stop SparkSession
spark.stop()
