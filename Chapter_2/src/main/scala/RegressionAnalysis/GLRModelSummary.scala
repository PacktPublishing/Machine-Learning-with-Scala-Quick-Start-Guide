package RegressionAnalysis

import org.apache.spark.ml.regression.{GeneralizedLinearRegression, GeneralizedLinearRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

object GLRModelSummary {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "E:/Exp/")
      .appName(s"OneVsRestExample")
      .getOrCreate()

    import spark.implicits._

    // Create an LinerRegression estimator
    val glr = new GeneralizedLinearRegression()
      .setFamily("gaussian") // continuous values being predicted
      .setLink("identity")
      .setFeaturesCol("features")
      .setLabelCol("label")

    val pipeline = new Pipeline().setStages(Preproessing.stringIndexerStages :+ Preproessing.assembler)
    val pipelineModel = pipeline.fit(Preproessing.trainingData)
    val trainDF = pipelineModel.transform(Preproessing.trainingData).select("features", "label")
    trainDF.show(5)

    // Fit the model
    val model = glr.fit(trainDF)

    // Print the coefficients and intercept for generalized linear regression model
    println(s"Coefficients: ${model.coefficients}")
    println(s"Intercept: ${model.intercept}")

    // Summarize the model over the training set and print out some metrics
    val summary = model.summary
    println(s"Coefficient Standard Errors: ${summary.coefficientStandardErrors.mkString(",")}")
    println(s"T Values: ${summary.tValues.mkString(",")}")
    println(s"P Values: ${summary.pValues.mkString(",")}")
    println(s"Dispersion: ${summary.dispersion}")
    println(s"Null Deviance: ${summary.nullDeviance}")
    println(s"Residual Degree Of Freedom Null: ${summary.residualDegreeOfFreedomNull}")
    println(s"Deviance: ${summary.deviance}")
    println(s"Residual Degree Of Freedom: ${summary.residualDegreeOfFreedom}")
    println(s"AIC: ${summary.aic}")
    println("Deviance Residuals: ")
    summary.residuals().show()

    spark.stop()
  }
}