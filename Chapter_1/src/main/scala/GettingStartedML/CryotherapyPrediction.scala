package GettingStartedML

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier

object CryotherapyPrediction {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "E:/Exp/")
      .appName("CryotherapyPrediction")
      .getOrCreate()

    import spark.implicits._

    var CryotherapyDF = spark.read.option("header", "true")
            .option("inferSchema", "true")
            .csv("data/Cryotherapy.csv")
    
    CryotherapyDF.printSchema()
    CryotherapyDF.show(10)

    //Since Spark ML algorithm expect a 'label' column, which is in our case 'Survived". Let's rename it to 'label'
    CryotherapyDF = CryotherapyDF.withColumnRenamed("Result_of_Treatment", "label")
    CryotherapyDF.printSchema()

    //Select columns for preparing training data using VectorAssembler()
    val selectedCols = Array("sex", "age", "Time", "Number_of_Warts", "Type", "Area")
    
    val vectorAssembler = new VectorAssembler()
          .setInputCols(selectedCols)
          .setOutputCol("features")

    // We convert prepare a training data containing "label" and "features", where the features contains existing numeric features and one hot encoded ones: 
    val numericDF = vectorAssembler.transform(CryotherapyDF)
                    .select("label", "features")
    numericDF.show(10)

    // Spliting the training data into train and test sets. We use 60% for the training and the rest 40% for testing 
    val splits = numericDF.randomSplit(Array(0.8, 0.2))
    val trainDF = splits(0)
    val testDF = splits(1)

    // Train a DecisionTree model.
    val dt = new DecisionTreeClassifier()
      .setImpurity("gini")
      .setMaxBins(10)
      .setMaxDepth(30)
      .setLabelCol("label")
      .setFeaturesCol("features")

    // Train model. This also runs the indexers.
    val dtModel = dt.fit(trainDF)

    // Since it's a binary clasisfication problem, we need BinaryClassificationEvaluator() estimator to evaluatemodel's performance on the test set
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")

    // Making predictions on test set
    val predictionDF = dtModel.transform(testDF)

    //Computing classification accuracy
    val accuracy = evaluator.evaluate(predictionDF)
    println("Accuracy =  " + accuracy)
    
    // Finally, we stop the Spark session by invokin stop() method
    spark.stop()
  }
}