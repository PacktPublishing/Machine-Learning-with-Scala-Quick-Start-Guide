package RegressionAnalysis

import org.apache.spark.ml.regression.{ GeneralizedLinearRegression, GeneralizedLinearRegressionModel }
import org.apache.spark.ml.{ Pipeline, PipelineModel }
import org.apache.spark.ml.tuning.{ CrossValidator, ParamGridBuilder }
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.ml.feature.VectorAssembler

import org.apache.log4j.Logger
import org.apache.log4j.Level

object UrbanTrafficGeneralizedLinearRegression {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "E:/Exp/")
      .appName(s"OneVsRestExample")
      .getOrCreate()
      
    Logger.getLogger("org").setLevel(Level.FATAL)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    import spark.implicits._

    val rawTrafficDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ";")
      .format("com.databricks.spark.csv")
      .load("data/Behavior of the urban traffic of the city of Sao Paulo in Brazil.csv")
      .cache

    rawTrafficDF.show()
    rawTrafficDF.printSchema()
    rawTrafficDF.describe().show()
    
    val newTrafficDF = rawTrafficDF.withColumnRenamed("Slowness in traffic (%)", "label") 
    val colNames = newTrafficDF.columns.dropRight(1)
    
    colNames.foreach(println)
    
    newTrafficDF.printSchema()

    // VectorAssembler for training features
    val assembler = new VectorAssembler()
      .setInputCols(colNames)
      .setOutputCol("features")

    val assembleDF = assembler.transform(newTrafficDF).select("features", "label")  
    assembleDF.printSchema()
    
    val seed = 1357911L
    val splits = assembleDF.randomSplit(Array(0.60, 0.40), seed)
    val (trainingData, testData) = (splits(0), splits(1))

    trainingData.cache
    testData.cache

    // Create an LinerRegression estimator
    val glr = new GeneralizedLinearRegression()
          .setFeaturesCol("features")
          .setLabelCol("label")

    // Building the Pipeline model for transformations and predictor
    println("Building ML regression model")
    val glrModel = glr.fit(trainingData)    

    // **********************************************************************
    println("Evaluating the model on the test set and calculating the regression metrics")
    // **********************************************************************
    val trainPredictionsAndLabels = glrModel.transform(testData).select("label", "prediction")
                                            .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd

    val testRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels)

    val results = "\n=====================================================================\n" +
      s"TrainingData count: ${trainingData.count}\n" +
      s"TestData count: ${testData.count}\n" +
      "=====================================================================\n" +
      s"TestData MSE = ${testRegressionMetrics.meanSquaredError}\n" +
      s"TestData RMSE = ${testRegressionMetrics.rootMeanSquaredError}\n" +
      s"TestData R-squared = ${testRegressionMetrics.r2}\n" +
      s"TestData MAE = ${testRegressionMetrics.meanAbsoluteError}\n" +
      s"TestData explained variance = ${testRegressionMetrics.explainedVariance}\n" +
      "=====================================================================\n"
    println(results)

    // ***********************************************************
    println("Preparing K-fold Cross Validation and Grid Search")
    // ***********************************************************
    val paramGrid = new ParamGridBuilder()
      .addGrid(glr.maxIter, Array(10, 20, 30, 50, 100, 500, 1000))
      .addGrid(glr.regParam, Array(0.001, 0.01, 0.1))
      .addGrid(glr.tol, Array(0.01, 0.1))
      .build()

    val numFolds = 10 //10-fold cross-validation
    val cv = new CrossValidator()
      .setEstimator(glr)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(numFolds)

    // ************************************************************
    println("Training model with Linear Regression algorithm")
    // ************************************************************
    val cvModel = cv.fit(trainingData)

    // Save the workflow
    //cvModel.write.overwrite().save("model/GLR_model")

    // Load the workflow back
    //val sameCV = CrossValidatorModel.load("model/GLR_model")

    // **********************************************************************
    println("Evaluating the cross validated model on the validation set and calculating the regression metrics")
    // **********************************************************************
    val trainPredictionsAndLabelsCV = cvModel.transform(testData).select("label", "prediction")
                                      .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd

    val testRegressionMetricsCV = new RegressionMetrics(trainPredictionsAndLabelsCV)

    val cvResults = "\n=====================================================================\n" +
      s"TrainingData count: ${trainingData.count}\n" +
      s"TestData count: ${testData.count}\n" +
      "=====================================================================\n" +
      s"TestData MSE = ${testRegressionMetricsCV.meanSquaredError}\n" +
      s"TestData RMSE = ${testRegressionMetricsCV.rootMeanSquaredError}\n" +
      s"TestData R-squared = ${testRegressionMetricsCV.r2}\n" +
      s"TestData MAE = ${testRegressionMetricsCV.meanAbsoluteError}\n" +
      s"TestData explained variance = ${testRegressionMetricsCV.explainedVariance}\n" +
      "=====================================================================\n"
    println(cvResults)
    
    // Print the coefficients and intercept for generalized linear regression model
    println(s"Coefficients: ${glrModel.coefficients}")
    println(s"Intercept: ${glrModel.intercept}")

    spark.stop()
  }
}