package GettingStartedDL

import org.apache.spark._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.feature.VectorAssembler

object CancerDataPreprocessor {
  def main(args: Array[String]) = {
    val spark: SparkSession = SparkSession.builder().
      appName("churn")
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "E:/Exp/")
      .config("spark.sql.crossJoin.enabled", "true")
      .getOrCreate()

    val data = spark.read.option("maxColumns", 25000).format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load("C:/Users/admin-karim/Desktop/old2/TCGA-PANCAN/TCGA-PANCAN-HiSeq-801x20531/data.csv"); // set your path accordingly

    val numFeatures = data.columns.length
    val numSamples = data.count()
    println("Number of features: " + numFeatures)
    println("Number of samples: " + numSamples)

    val numericDF = data.drop("id") // now 20531 features left

    val labels = spark.read.format("com.databricks.spark.csv").option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types 
      .load("C:/Users/admin-karim/Desktop/old2/TCGA-PANCAN/TCGA-PANCAN-HiSeq-801x20531/labels.csv")

    labels.show(10)

    val indexer = new StringIndexer().setInputCol("Class").setOutputCol("label").setHandleInvalid("skip"); // skip null/invalid values    
    val indexedDF = indexer.fit(labels).transform(labels).select(col("label").cast(DataTypes.IntegerType)); // casting data types to integer

    indexedDF.show()

    val combinedDF = numericDF.join(indexedDF)

    val splits = combinedDF.randomSplit(Array(0.7, 0.3), 12345L) //70% for training, 30% for testing
    val trainingData = splits(0)
    val testData = splits(1)

    println(trainingData.count()); // number of samples in training set
    println(testData.count()); // number of samples in test set

    trainingData.coalesce(1).write
      .format("com.databricks.spark.csv")
      .option("header", "false")
      .option("delimiter", ",")
      .save("output/TCGA_train.csv")

    testData.coalesce(1).write
      .format("com.databricks.spark.csv")
      .option("header", "false")
      .option("delimiter", ",")
      .save("output/TCGA_test.csv")

  }
}