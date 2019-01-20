package MovieRecommendation

import org.apache.spark.sql.SparkSession

object MovieSimilarities {
  val spark: SparkSession = SparkSession
    .builder()
    .appName("JavaLDAExample")
    .master("local[*]")
    .config("spark.sql.warehouse.dir", "E:/Exp/").
    getOrCreate()

  /**
   * Parameters to regularize correlation.
   */
  val PRIOR_COUNT = 10
  val PRIOR_CORRELATION = 0

  val TRAIN_FILENAME = "data/ua.base"
  val TEST_FIELNAME = "data/ua.test"
  val MOVIES_FILENAME = "data/u.item"

  // get movie names keyed on id
  val movies = spark.sparkContext.textFile(MOVIES_FILENAME)
    .map(line => {
      val fields = line.split("\\|")
      (fields(0).toInt, fields(1))
    })
  val movieNames = movies.collectAsMap() // for local use to map id <-> movie name for pretty-printing

  // extract (userid, movieid, rating) from ratings data
  val ratings = spark.sparkContext.textFile(TRAIN_FILENAME)
    .map(line => {
      val fields = line.split("\t")
      (fields(0).toInt, fields(1).toInt, fields(2).toInt)
    })

  // get num raters per movie, keyed on movie id
  val numRatersPerMovie = ratings
    .groupBy(tup => tup._2)
    .map(grouped => (grouped._1, grouped._2.size))

  // join ratings with num raters on movie id
  val ratingsWithSize = ratings
    .groupBy(tup => tup._2)
    .join(numRatersPerMovie)
    .flatMap(joined => {
      joined._2._1.map(f => (f._1, f._2, f._3, joined._2._2))
    })

  // ratingsWithSize now contains the following fields: (user, movie, rating, numRaters).

  // dummy copy of ratings for self join
  val ratings2 = ratingsWithSize.keyBy(tup => tup._1)

  // join on userid and filter movie pairs such that we don't double-count and exclude self-pairs
  val ratingPairs =
    ratingsWithSize
      .keyBy(tup => tup._1)
      .join(ratings2)
      .filter(f => f._2._1._2 < f._2._2._2)

  // compute raw inputs to similarity metrics for each movie pair
  val vectorCalcs =
    ratingPairs
      .map(data => {
        val key = (data._2._1._2, data._2._2._2)
        val stats =
          (data._2._1._3 * data._2._2._3, // rating 1 * rating 2
            data._2._1._3, // rating movie 1
            data._2._2._3, // rating movie 2
            math.pow(data._2._1._3, 2), // square of rating movie 1
            math.pow(data._2._2._3, 2), // square of rating movie 2
            data._2._1._4, // number of raters movie 1
            data._2._2._4) // number of raters movie 2
        (key, stats)
      })
      .groupByKey()
      .map(data => {
        val key = data._1
        val vals = data._2
        val size = vals.size
        val dotProduct = vals.map(f => f._1).sum
        val ratingSum = vals.map(f => f._2).sum
        val rating2Sum = vals.map(f => f._3).sum
        val ratingSq = vals.map(f => f._4).sum
        val rating2Sq = vals.map(f => f._5).sum
        val numRaters = vals.map(f => f._6).max
        val numRaters2 = vals.map(f => f._7).max
        (key, (size, dotProduct, ratingSum, rating2Sum, ratingSq, rating2Sq, numRaters, numRaters2))
      })

  // compute similarity metrics for each movie pair
  val similarities =
    vectorCalcs
      .map(fields => {
        val key = fields._1
        val (size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq, numRaters, numRaters2) = fields._2
        val corr = correlation(size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq)
        val regCorr = regularizedCorrelation(size, dotProduct, ratingSum, rating2Sum,
          ratingNormSq, rating2NormSq, PRIOR_COUNT, PRIOR_CORRELATION)
        val cosSim = cosineSimilarity(dotProduct, scala.math.sqrt(ratingNormSq), scala.math.sqrt(rating2NormSq))
        val jaccard = jaccardSimilarity(size, numRaters, numRaters2)

        (key, (corr, regCorr, cosSim, jaccard))
      })

  def evaluateModel(movieName: String): Unit = {
    // test a few movies out (substitute the contains call with the relevant movie name
    val sample = similarities.filter(m => {
      val movies = m._1
      (movieNames(movies._1).contains(movieName))
    })

    // collect results, excluding NaNs if applicable
    val result = sample.map(v => {
      val m1 = v._1._1
      val m2 = v._1._2
      val corr = v._2._1
      val rcorr = v._2._2
      val cos = v._2._3
      val j = v._2._4
      (movieNames(m1), movieNames(m2), corr, rcorr, cos, j)
    }).collect().filter(e => !(e._4 equals Double.NaN)) // test for NaNs must use equals rather than ==
      .sortBy(elem => elem._4).take(10)

    // print the top 10 out
    result.foreach(r => println(r._1 + " | " + r._2 + " | " + r._3.formatted("%2.4f") + " | " + r._4.formatted("%2.4f")
      + " | " + r._5.formatted("%2.4f") + " | " + r._6.formatted("%2.4f")))
  }

  // *************************
  // * SIMILARITY MEASURES
  // *************************

  /**
   * The correlation between two vectors A, B is
   *   cov(A, B) / (stdDev(A) * stdDev(B))
   *
   * This is equivalent to
   *   [n * dotProduct(A, B) - sum(A) * sum(B)] /
   *     sqrt{ [n * norm(A)^2 - sum(A)^2] [n * norm(B)^2 - sum(B)^2] }
   */
  def correlation(size: Double, dotProduct: Double, ratingSum: Double,
    rating2Sum: Double, ratingNormSq: Double, rating2NormSq: Double) = {

    val numerator = size * dotProduct - ratingSum * rating2Sum
    val denominator = scala.math.sqrt(size * ratingNormSq - ratingSum * ratingSum) *
      scala.math.sqrt(size * rating2NormSq - rating2Sum * rating2Sum)

    numerator / denominator
  }

  /**
   * Regularize correlation by adding virtual pseudocounts over a prior:
   *   RegularizedCorrelation = w * ActualCorrelation + (1 - w) * PriorCorrelation
   * where w = # actualPairs / (# actualPairs + # virtualPairs).
   */
  def regularizedCorrelation(size: Double, dotProduct: Double, ratingSum: Double,
    rating2Sum: Double, ratingNormSq: Double, rating2NormSq: Double,
    virtualCount: Double, priorCorrelation: Double) = {

    val unregularizedCorrelation = correlation(size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq)
    val w = size / (size + virtualCount)

    w * unregularizedCorrelation + (1 - w) * priorCorrelation
  }

  /**
   * The cosine similarity between two vectors A, B is
   *   dotProduct(A, B) / (norm(A) * norm(B))
   */
  def cosineSimilarity(dotProduct: Double, ratingNorm: Double, rating2Norm: Double) = {
    dotProduct / (ratingNorm * rating2Norm)
  }

  /**
   * The Jaccard Similarity between two sets A, B is
   *   |Intersection(A, B)| / |Union(A, B)|
   */
  def jaccardSimilarity(usersInCommon: Double, totalUsers1: Double, totalUsers2: Double) = {
    val union = totalUsers1 + totalUsers2 - usersInCommon
    usersInCommon / union
  }

  def main(args: Array[String]): Unit = {
    
        //The format of each line is (userID, movieID, rating)
    val new_user_ratings = Seq(
      (0, 260, 4), // Star Wars (1977)
      (0, 1, 3), //Toy Story (1995)
      (0, 16, 3), // Casino (1995)
      (0, 25, 4), // Leaving Las Vegas (1995)
      (0, 32, 4), // Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
      (0, 335, 1), // Flintstones, The (1994)
      (0, 379, 1), // Timecop (1994)
      (0, 296, 3), // Pulp Fiction (1994)
      (0, 858, 5), // Godfather, The (1972)
      (0, 50, 4) // Usual Suspects, The (1995)
      )

    val new_user_ratings_RDD = spark.sparkContext.parallelize(new_user_ratings)
    val dfWithSchema = spark.createDataFrame(new_user_ratings_RDD).toDF("userID", "movieID", "Rating")
    dfWithSchema.show()
    
    evaluateModel("Die Hard (1988)")
    evaluateModel("Postino, Il (1994)")
    evaluateModel("Star Wars (1977)")
  }
}