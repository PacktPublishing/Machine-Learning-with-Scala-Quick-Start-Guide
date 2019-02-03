package GettingStartedML

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier

object TitanicSurvivalPrediction {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "E:/Exp/")
      .appName("TitanicSurvival")
      .getOrCreate()

    import spark.implicits._

    val titanicDF = spark.read.option("header", "true").option("inferSchema", "true").csv("data/train.csv")
    titanicDF.printSchema()
    titanicDF.show(5)

    //Handling missing values also called null imputation
    val meanValue = titanicDF.agg(mean(titanicDF("Age"))).first.getDouble(0)
    val fixedAgeDF = titanicDF.na.fill(meanValue, Array("Age"))

    //Since Spark ML algorithm expect a 'label' column, which is in our case 'Survived". Let's rename it to 'label'
    val trainSet = fixedAgeDF.withColumnRenamed("Survived", "label")

    // Also, not all the columns are important such as passenger ID, name (well can be but we'll discuss later on), Ticket and Cabin. 
    // The sex, embarked and pclass features are numeric, we conver them into numeric using StringIndexer()
    val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("Sex_numeric").setHandleInvalid("skip") // skip null/invalid values    
    val embarkedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("Embarked_numeric").setHandleInvalid("skip") // skip null/invalid values   
    val pclassIndexer = new StringIndexer().setInputCol("Pclass").setOutputCol("Pclass_numeric").setHandleInvalid("skip") // skip null/invalid values   

    //Select columns for preparing training data using VectorAssembler()
    val selectedCols = Array("Sex_numeric", "Pclass_numeric", "Embarked_numeric", "SibSp", "Parch", "Age", "Fare")
    val vectorAssembler = new VectorAssembler().setInputCols(selectedCols).setOutputCol("features")

    //Chain string indexers into a pipeline as different stages
    val pipeline = new Pipeline().setStages(Array(genderIndexer, embarkedIndexer, pclassIndexer, vectorAssembler))

    // We convert prepare a training data containing "label" and "features", where the features contains existing numeric features and one hot encoded ones: 
    val numericDF = pipeline.fit(trainSet).transform(trainSet).select("label", "features")
    numericDF.show()

    // Spliting the training data into train and test sets. We use 60% for the training and the rest 40% for testing 
    val splits = numericDF.randomSplit(Array(0.6, 0.4))
    val trainDF = splits(0)
    val testDF = splits(1)

    // Train a DecisionTree model.
    val DT = new DecisionTreeClassifier()
      .setImpurity("gini")
      .setMaxBins(10)
      .setMaxDepth(30)
      .setLabelCol("label")
      .setFeaturesCol("features")

    // Train model. This also runs the indexers.
    val dtModel = DT.fit(trainDF)

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