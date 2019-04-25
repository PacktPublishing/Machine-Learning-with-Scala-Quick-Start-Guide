package org.fit.genomics

import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{ Vectors, Vector }
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.SparkContext
import org.apache.spark.sql.types.{ IntegerType, StringType, StructField, StructType }
import org.apache.spark.ml.feature.{ VectorAssembler, Normalizer }
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.PCA

import water._
import water.fvec.Frame
import water.{ Job, Key }
import water.fvec.Frame
import hex.FrameSplitter
import org.apache.spark.h2o._
import org.apache.spark.h2o.H2OContext

import org.bdgenomics.adam.rdd.ADAMContext._
import org.bdgenomics.formats.avro.{ Genotype, GenotypeAllele }

import java.io.File
import java.io._
import scala.collection.JavaConverters._
import scala.collection.immutable.Range.inclusive
import scala.io.Source

object PopStratClusterings {
  def main(args: Array[String]): Unit = {
    val genotypeFile = "C:/Users/admin-karim/Downloads/1.vcf"
    val panelFile = "C:/Users/admin-karim/Downloads/genotypes.panel"

    val sparkSession: SparkSession = SparkSession.builder.appName("PopStrat").master("local[*]").getOrCreate()
    val sc: SparkContext = sparkSession.sparkContext

    val populations = Set("GBR", "MXL", "ASW", "CHB", "CLM")
    def extract(file: String, filter: (String, String) => Boolean): Map[String, String] = {
      Source
        .fromFile(file)
        .getLines()
        .map(line => {
          val tokens = line.split(Array('\t', ' ')).toList
          tokens(0) -> tokens(1)
        })
        .toMap
        .filter(tuple => filter(tuple._1, tuple._2))
    }

    val panel: Map[String, String] = extract(
      panelFile,
      (sampleID: String, pop: String) => populations.contains(pop))
    val allGenotypes: RDD[Genotype] = sc.loadGenotypes(genotypeFile).rdd
    val genotypes: RDD[Genotype] = allGenotypes.filter(genotype => {
      panel.contains(genotype.getSampleId)
    })

    // Convert the Genotype objects to our own SampleVariant objects to try and conserve memory
    case class SampleVariant(sampleId: String,
      variantId: Int,
      alternateCount: Int)

    def variantId(genotype: Genotype): String = {
      val name = genotype.getVariant.getContigName
      val start = genotype.getVariant.getStart
      val end = genotype.getVariant.getEnd
      s"$name:$start:$end"
    }

    def alternateCount(genotype: Genotype): Int = {
      genotype.getAlleles.asScala.count(_ != GenotypeAllele.REF)
    }

    def toVariant(genotype: Genotype): SampleVariant = {
      // Intern sample IDs as they will be repeated a lot
      new SampleVariant(genotype.getSampleId.intern(),
        variantId(genotype).hashCode(),
        alternateCount(genotype))
    }

    val variantsRDD: RDD[SampleVariant] = genotypes.map(toVariant)
    val variantsBySampleId: RDD[(String, Iterable[SampleVariant])] =
      variantsRDD.groupBy(_.sampleId)
    val sampleCount: Long = variantsBySampleId.count()
    println("Found " + sampleCount + " samples")

    val variantsByVariantId: RDD[(Int, Iterable[SampleVariant])] =
      variantsRDD.groupBy(_.variantId).filter {
        case (_, sampleVariants) => sampleVariants.size == sampleCount
      }

    val variantFrequencies: collection.Map[Int, Int] = variantsByVariantId
      .map {
        case (variantId, sampleVariants) =>
          (variantId, sampleVariants.count(_.alternateCount > 0))
      }
      .collectAsMap()

    val permittedRange = inclusive(11, 11)
    val filteredVariantsBySampleId: RDD[(String, Iterable[SampleVariant])] =
      variantsBySampleId.map {
        case (sampleId, sampleVariants) =>
          val filteredSampleVariants = sampleVariants.filter(
            variant =>
              permittedRange.contains(
                variantFrequencies.getOrElse(variant.variantId, -1)))
          (sampleId, filteredSampleVariants)
      }

    val sortedVariantsBySampleId: RDD[(String, Array[SampleVariant])] =
      filteredVariantsBySampleId.map {
        case (sampleId, variants) =>
          (sampleId, variants.toArray.sortBy(_.variantId))
      }

    println(s"Sorted by Sample ID RDD: " + sortedVariantsBySampleId.first())

    val header = StructType(
      Array(StructField("Region", StringType)) ++
        sortedVariantsBySampleId
        .first()
        ._2
        .map(variant => {
          StructField(variant.variantId.toString, IntegerType)
        }))

    val rowRDD: RDD[Row] = sortedVariantsBySampleId.map {
      case (sampleId, sortedVariants) =>
        val region: Array[String] = Array(panel.getOrElse(sampleId, "Unknown"))
        val alternateCounts: Array[Int] = sortedVariants.map(_.alternateCount)
        Row.fromSeq(region ++ alternateCounts)
    }

    // Create the SchemaRDD from the header and rows and convert the SchemaRDD into a Spark dataframe
    val sqlContext = sparkSession.sqlContext
    val schemaDF = sqlContext.createDataFrame(rowRDD, header).drop("Region")
    schemaDF.printSchema()
    schemaDF.show(10)

    println(schemaDF.columns.length)

    // Using vector assembler to create feature vector 
    val featureCols = schemaDF.columns
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val assembleDF = assembler.transform(schemaDF).select("features")
    assembleDF.show()

    // Elbow method with reduced dimension
    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(5)
      .fit(assembleDF)

    val pcaDF = pca.transform(assembleDF).select("pcaFeatures").withColumnRenamed("pcaFeatures", "features")
    pcaDF.show()

    val iterations = 20
    for (i <- 2 to iterations) {
      // Trains a k-means model.
      val kmeans = new KMeans().setK(i).setSeed(12345L)
      val model = kmeans.fit(pcaDF)

      // Evaluate clustering by computing Within Set Sum of Squared Errors.
      val WCSS = model.computeCost(pcaDF)
      println("Within Set Sum of Squared Errors for k = " + i + " is " + WCSS)
    }
    /*
		Within Set Sum of Squared Errors for k = 2 is 135.0048361804504
		Within Set Sum of Squared Errors for k = 3 is 90.95271589232344
		Within Set Sum of Squared Errors for k = 4 is 73.03991105363087
		Within Set Sum of Squared Errors for k = 5 is 52.712937492025276
		Within Set Sum of Squared Errors for k = 6 is 35.0048649663809
		Within Set Sum of Squared Errors for k = 7 is 33.11707134428616
		Within Set Sum of Squared Errors for k = 8 is 30.546631341918243
		Within Set Sum of Squared Errors for k = 9 is 28.453155497711535
		Within Set Sum of Squared Errors for k = 10 is 24.93179715697327
		Within Set Sum of Squared Errors for k = 11 is 25.56839205985354
		Within Set Sum of Squared Errors for k = 12 is 18.76755804955161
		Within Set Sum of Squared Errors for k = 13 is 18.55123407031501
		Within Set Sum of Squared Errors for k = 14 is 16.140301237245204
		Within Set Sum of Squared Errors for k = 15 is 14.143806816130821
		Within Set Sum of Squared Errors for k = 16 is 15.017971347008297
		Within Set Sum of Squared Errors for k = 17 is 12.266417893931926
		Within Set Sum of Squared Errors for k = 18 is 11.108546956133177
		Within Set Sum of Squared Errors for k = 19 is 11.505990055606803
		Within Set Sum of Squared Errors for k = 20 is 12.26634441065655
    */

    // Evaluate clustering by computing Silhouette score
    val evaluator = new ClusteringEvaluator()

    for (k <- 2 to 20 by 1) {
      val kmeans = new KMeans().setK(k).setSeed(12345L)
      val model = kmeans.fit(pcaDF)
      val transformedDF = model.transform(pcaDF)
      val score = evaluator.evaluate(transformedDF)
      println("Silhouette with squared euclidean distance for k = " + k + " is " + score)
    }
    /*
     * Silhouette with squared euclidean distance for k = 2 is 0.9175803927739566
      Silhouette with squared euclidean distance for k = 3 is 0.8288633816548874
      Silhouette with squared euclidean distance for k = 4 is 0.6376477607336495
      Silhouette with squared euclidean distance for k = 5 is 0.6731472765720269
      Silhouette with squared euclidean distance for k = 6 is 0.6641908680884869
      Silhouette with squared euclidean distance for k = 7 is 0.5758081075880451
      Silhouette with squared euclidean distance for k = 8 is 0.588881352222969
      Silhouette with squared euclidean distance for k = 9 is 0.6485153435398991
      Silhouette with squared euclidean distance for k = 10 is 0.48949118556376964
      Silhouette with squared euclidean distance for k = 11 is 0.5371218728964895
      Silhouette with squared euclidean distance for k = 12 is 0.5569086502410784
      Silhouette with squared euclidean distance for k = 13 is 0.3990728491364654
      Silhouette with squared euclidean distance for k = 14 is 0.5311155969749914
      Silhouette with squared euclidean distance for k = 15 is 0.5457021641983345
      Silhouette with squared euclidean distance for k = 16 is 0.4891629883332554
      Silhouette with squared euclidean distance for k = 17 is 0.5452872742013583
      Silhouette with squared euclidean distance for k = 18 is 0.5304994251201304
      Silhouette with squared euclidean distance for k = 19 is 0.5327466913746908
      Silhouette with squared euclidean distance for k = 20 is 0.45336547054142284
     */

    val kmeansOptimal = new KMeans().setK(2).setSeed(12345L)
    val modelOptimal = kmeansOptimal.fit(pcaDF)

    // Making predictions
    val predictionsOptimalDF = modelOptimal.transform(pcaDF)
    predictionsOptimalDF.show()
    
    // Evaluate clustering by computing Silhouette score
    val evaluatorOptimal = new ClusteringEvaluator()

    val silhouette = evaluatorOptimal.evaluate(predictionsOptimalDF)
    println(s"Silhouette with squared euclidean distance = $silhouette")

    sparkSession.stop()
  }
}
