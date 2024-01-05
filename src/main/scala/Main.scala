import org.apache.spark.SparkContext
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}

import java.io.PrintWriter

/**
 * Data WareHouses & Data Mining Course, Aristotle University of Thessaloniki
 * Semester programming assignment (Winter 2023-2024)
 *
 * @author Vasileios Papastergios (ID: 3651), Lazaros Gogos (ID: 3877)
 */
object Main {
  private var xMin: Double = 0.0
  private var xMax: Double = 1.0
  private var yMin: Double = 0.0
  private var yMax: Double = 1.0

  private final val NANOS_PER_SEC: Double = 1e9d
  private final val DEFAULT_DATA_FILE: String = "data.csv"

  def main(args: Array[String]): Unit = {

    //    0. Initialize timer, SparkSession and SparkContext
    val startTimestamp = System.nanoTime
    val ss = initializeSparkSession(cores = "*", appName = "PapaGo_Clustering")
    val sc = ss.sparkContext
    sc.setLogLevel("ERROR")

    //    1. Read input file
    val rawData = readInputDataFile(sc, args)

    //    2. Perform data cleaning
    val validCoordinates = performDataCleaning(rawData)

    //    3. Scale data to [0, 1]
    computeMinMaxValues(validCoordinates) // computes xMin, xMax, yMin, yMax and saves them to object attributes
    val transformedCoordinates = performMinMaxScaling(validCoordinates)

    //   4. Implement K-means clustering for k >> 5   +   5. Detect Outliers
    val dataFrame = createDataFrameFromCoordinatesRDD(transformedCoordinates, ss)
    runKMeans(k = 200, seed = System.nanoTime(), dataFrame = dataFrame,
      predictionsDir = "predictions.d", centersOutFile = "kMeans_centers.csv",
      detectOutliers = true, removeOutliers = true
    )

    //    Calculate execution time
    val duration = (System.nanoTime - startTimestamp) / NANOS_PER_SEC
    println(s"\nTotal execution time: $duration sec")
    sc.stop(0)
  }

  /**
   * Initializes a SparkSession instance with defined parameters.
   *
   * @param cores   the number of cores to use as String. If '*', all available cores are used.
   * @param appName the name of the application.
   * @return a SparkSession instance with the defined parameters.
   */
  private def initializeSparkSession(cores: String, appName: String): SparkSession = {
    SparkSession.builder()
      .master(s"local[$cores]")
      .appName(appName)
      .getOrCreate()
  }

  /**
   * Extracts the input file name from command line arguments array, if specified.
   * In case there are no command line arguments, a default file name is used. The value of the default file
   * name is specified as object's constant attribute.
   * Corresponds to assignment TASK 1.
   *
   * @param args the command line arguments of function main.
   * @return the file name as String (if existent in arguments), else the default file name.
   */
  private def getInputFileName(args: Array[String]): String = {
    if (args.length == 0) {
      return DEFAULT_DATA_FILE
    }
    args(0)
  }

  /**
   * Reads the input data file and constructs an RDD with the data.
   * Corresponds to assignment TASK 1.
   *
   * @param sc   the SparkContext object.
   * @param args the command line arguments array of function main.
   * @return an RDD with the raw input data.
   */
  private def readInputDataFile(sc: SparkContext, args: Array[String]): RDD[String] = {
    val inputFile = getInputFileName(args)
    sc.textFile(inputFile)
  }


  /**
   * Preforms data cleaning on an RDD[String] containing "x,y" coordinates.
   * Removes lines with missing values, splits the "x,y" String per row to (Double, Double)
   * coordinates and constructs an RDD with the clean data.
   * Corresponds to assignment TASK 2.
   *
   * @param rawData the RDD[String] to clean.
   * @return an RDD[(Double, Double)] containing the clean data.
   */
  private def performDataCleaning(rawData: RDD[String]): RDD[(Double, Double)] = {
    val cleanData = filterOutLinesWithMissingValues(rawData)
    splitLinesToDoubleCoordinates(cleanData)
  }

  /**
   * Given an RDD[String] containing "x,y" coordinates, filters out lines that either
   * are missing x or y coordinate, or are empty.
   * Corresponds to assignment TASK 2.
   *
   * @param rawData an RDD[String] containing the lines of the input raw data.
   * @return an RDD[String] without the corrupted lines.
   */
  private def filterOutLinesWithMissingValues(rawData: RDD[String]): RDD[String] = {
    rawData.filter(line =>
      !line.startsWith(",") &&
        !line.endsWith(",") &&
        line.nonEmpty
    )
  }

  /**
   * Given an RDD[String] containing "x,y" coordinates, splits the coordinates
   * to Double x, y values and constructs an RDD[(Double, Double)] with the result.
   * Corresponds to assignment TASK 2.
   *
   * @param stringData the RDD[String] containing "x,y" coordinates to split.
   * @return an RDD[(Double, Double)] with the split coordinates.
   */
  private def splitLinesToDoubleCoordinates(stringData: RDD[String]): RDD[(Double, Double)] = {
    stringData.map(line => {
      val coordinates = line.split(",")
      (coordinates(0).toDouble, coordinates(1).toDouble)
    })
  }


  /**
   * Scales a given RDD containing coordinates to [0, 1] in both dimensions.
   * Before calling the function, one should have executed computeMinMaxValues() function.
   * Corresponds to assignment TASK 3.
   *
   * @param unscaled the RDD[(Double, Double)] containing coordinates to scale.
   * @return an RDD[(Double, Double)] with the scaled values.
   */
  private def performMinMaxScaling(unscaled: RDD[(Double, Double)]): RDD[(Double, Double)] = {
    unscaled.map(coordinates => {
      val x = coordinates._1
      val y = coordinates._2
      // xMin, xMax, yMin, yMax are computed by computeMinMaxValues()
      ((x - xMin) / (xMax - xMin), (y - yMin) / (yMax - yMin))
    })
  }

  /**
   * Computes the minimum and maximum value for each dimension, given an RDD that contains
   * coordinates. The computed values are stored in object's attributes xMin, xMax, yMin, yMax.
   * Corresponds to assignment TASK 3.
   *
   * @param coordinates the RDD to compute min and max values on.
   */
  private def computeMinMaxValues(coordinates: RDD[(Double, Double)]): Unit = {
    val x = coordinates.map(coordinates => coordinates._1)
    val y = coordinates.map(coordinates => coordinates._2)
    xMin = x.min
    xMax = x.max
    yMin = y.min
    yMax = y.max
  }

  /**
   * Transforms a given, scaled (x, y) pair to its original values (before MinMax scaling).
   * The function uses the Object's attributes xMin, xMax, yMin, yMax to perform the inverse transformation.
   * Corresponds to assignment TASK 3.
   *
   * @param xScaled the scaled value of x-coordinate.
   * @param yScaled the scaled value of y-coordinate.
   * @return a Vector containing the original (before scaling) coordinates values.
   */
  private def getOriginalCoordinates(xScaled: Double, yScaled: Double): Vector[Double] = {
    Vector(
      xScaled * (xMax - xMin) + xMin,
      yScaled * (yMax - yMin) + yMin
    )
  }


  /**
   * Creates a DataFrame from a RDD[(Double, Double)] containing coordinates.
   * Corresponds to assignment TASK 4.
   *
   * @param coordinates the RDD to be converted to DataFrame.
   * @param ss          the SparkSession object.
   * @return a DataFrame containing the given RDD data with the appropriate schema.
   */
  private def createDataFrameFromCoordinatesRDD(coordinates: RDD[(Double, Double)], ss: SparkSession): DataFrame = {
    val schema = StructType(
      StructField(name = "x", dataType = DoubleType, nullable = false) ::
        StructField(name = "y", dataType = DoubleType, nullable = false) :: Nil
    )
    val data = coordinates.map(line => Row(
      line._1.asInstanceOf[Number].doubleValue(),
      line._2.asInstanceOf[Number].doubleValue()
    ))
    ss.createDataFrame(data, schema)
  }

  /**
   * Executes K-Means algorithm with the specified hyper-parameters and execution flags. Writes the
   * clustering results (predictions) to csv file(s).
   *
   * @param k              the hyper-parameter k of the K-means algorithm.
   * @param seed           the random seed for initial centers.
   * @param dataFrame      the DataFrame to train the K-means model on.
   * @param predictionsDir the output directory to write predictions in.
   * @param centersOutFile the output file to write the cluster centers in.
   * @param detectOutliers execution flag to detect outliers or not.
   * @param removeOutliers execution flag to remove outliers or not. If detectOutliers is set to false, it is ignored.
   */
  private def runKMeans(k: Integer = 10, seed: Long = 22L, dataFrame: DataFrame,
                        predictionsDir: String, centersOutFile: String,
                        detectOutliers: Boolean, removeOutliers: Boolean): Unit = {

    // Create features column
    val assembler = new VectorAssembler()
      .setInputCols(Array("x", "y"))
      .setOutputCol("features")
    val features = assembler.transform(dataFrame)

    // Train the model
    val kMeans = new KMeans().setK(k).setSeed(seed)
    val model = kMeans.fit(features)

    // Make predictions
    var predictions = model.transform(features).selectExpr("x as xScaled", "y as yScaled", "prediction as clusterId")

    // Retrieve original coordinates of the cluster centers
    val originalCenters = model.clusterCenters.map(scaledCoordinates => getOriginalCoordinates(scaledCoordinates(0), scaledCoordinates(1)))
    writeCenters(centers = originalCenters, outFile = centersOutFile)

    // Detect and remove outliers, if respective flags are set
    if (detectOutliers) {
      predictions = this.detectOutliers(predictions, originalCenters, removeOutliers)
    }

    // Write the predictions is csv format to use in R
    writeDataFrame(dataFrame = predictions, outDir = predictionsDir)
  }

  /**
   * Detects and prints outliers found in the predictions DataFrame, based on the simple algorithm.
   * Outliers are defined as data points, whose distance from the cluster center is greater than a
   * specified threshold.
   *
   * @param predictions     the predictions DataFrame.
   * @param originalCenters the original (non scaled) coordinates of the cluster centers found by K-means.
   * @param removeOutliers  execution flag. If set to true, outliers are removed, else preserved in the returned
   *                        DataFrame.
   * @return a DataFrame with the extended predictions data, with/without the outliers, based on
   *         the removeOutliers flag.
   */
  private def detectOutliers(predictions: DataFrame, originalCenters: Array[Vector[Double]], removeOutliers: Boolean): DataFrame = {
    val predictionsExpanded = predictions
      .withColumn("xOriginal", predictions("xScaled") * (xMax - xMin) + xMin)
      .withColumn("yOriginal", predictions("yScaled") * (yMax - yMin) + yMin)
      .rdd.map(row => {
        val xScaled = row.getDouble(0)
        val yScaled = row.getDouble(1)
        val clusterId = row.getInt(2)
        val xOriginal = row.getDouble(3)
        val yOriginal = row.getDouble(4)
        val xClusterCenter = originalCenters(clusterId)(0)
        val yClusterCenter = originalCenters(clusterId)(1)
        val euclideanDistanceFromCenter = math.sqrt(
          math.pow(xOriginal - xClusterCenter, 2) +
            math.pow(yOriginal - yClusterCenter, 2)
        )
        Row(xScaled, yScaled, clusterId, xOriginal, yOriginal,
          xClusterCenter, yClusterCenter, euclideanDistanceFromCenter)
      })

    val meanDistances = computeMeanDistances(predictionsExpanded)
    val stdDistances = computeStdDistances(predictionsExpanded)

    // outlier criterion: distanceFromCenter > meanDistanceOfCluster + 3.5 * stdDistanceFromCenter
    val outliers = predictionsExpanded.filter(row =>
      row.getDouble(7) > getOutlierThreshold(meanDistances(row.getInt(2))._2, stdDistances(row.getInt(2))._2))
    printOutliers(outliers)

    if (removeOutliers) {
      predictionsExpanded.filter(row =>
        row.getDouble(7) <= getOutlierThreshold(meanDistances(row.getInt(2))._2, stdDistances(row.getInt(2))._2))
    }
    createDataFrameFromPredictionsRDD(predictionsExpanded)
  }

  /**
   * A utility function that prints the outliers to stdout, formatted as (x, y) pairs.
   *
   * @param outliers the outliers to print.
   */
  private def printOutliers(outliers: RDD[Row]): Unit = {
    printf("Detected Outliers:\n\n")
    outliers.foreach(row => printf("(%.5f, %.5f)\n", row.getDouble(3), row.getDouble(4)))
  }

  /**
   * A utility function that converts a predictions RDD to a DataFrame with appropriate schema.
   *
   * @param predictionsExpanded the RDD[Row] to convert to DataFrame.
   * @return a DataFrame with the data contained in the given RDD.
   */
  private def createDataFrameFromPredictionsRDD(predictionsExpanded: RDD[Row]): DataFrame = {
    val schema = StructType(
      StructField(name = "xScaled", dataType = DoubleType, nullable = false) ::
        StructField(name = "yScaled", dataType = DoubleType, nullable = false) ::
        StructField(name = "clusterId", dataType = IntegerType, nullable = false) ::
        StructField(name = "xOriginal", dataType = DoubleType, nullable = false) ::
        StructField(name = "yOriginal", dataType = DoubleType, nullable = false) ::
        StructField(name = "xClusterCenter", dataType = DoubleType, nullable = false) ::
        StructField(name = "yClusterCenter", dataType = DoubleType, nullable = false) ::
        StructField(name = "euclideanDistanceFromCenter", dataType = DoubleType, nullable = false) :: Nil
    )
    SparkSession.builder().getOrCreate().createDataFrame(predictionsExpanded, schema)
  }

  /**
   * Given a mean and a std value, the function computes the outlier threshold, i.e. the upper bound of
   * distance between a point and its cluster's center, above which the point shall be considered outlier.
   *
   * @param meanDistance the mean distance of the points from the respective cluster center.
   * @param stdDistance  the std distance of the points from the respective cluster center.
   * @return the outlier threshold.
   */
  private def getOutlierThreshold(meanDistance: Double, stdDistance: Double): Double = {
    meanDistance + 3.5 * stdDistance
  }

  /**
   * Computes a List[(Int, Double)] of mean distances between all cluster points and the respective
   * cluster center for each class.
   *
   * @param predictionsExpanded an RDD[Row] containing predictions.
   * @return a List with the mean distances per class.
   */
  private def computeMeanDistances(predictionsExpanded: RDD[Row]): List[(Int, Double)] = {
    predictionsExpanded.map(row => (row.getInt(2), row.getDouble(7)))
      .mapValues(distance => (distance, 1))
      .reduceByKey((pair1, pair2) => (pair1._1 + pair2._1, pair1._2 + pair2._2))
      .mapValues(pair => pair._1 / pair._2)
      .collect()
      .toList
      .sortBy(pair => pair._1)
  }

  /**
   * Computes a List[(Int, Double)] of standard deviation of distances between all cluster points and the
   * respective cluster center for each class.
   *
   * @param predictionsExpanded an RDD[Row] containing predictions.
   * @return a List with the standard deviation of distances per class.
   */
  private def computeStdDistances(predictionsExpanded: RDD[Row]): List[(Int, Double)] = {
    predictionsExpanded.map(row => (row.getInt(2), row.getDouble(7)))
      .mapValues(x => (1, x, x * x))
      .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2, x._3 + y._3))
      .mapValues(x => math.sqrt(x._3 / x._1 - math.pow(x._2 / x._1, 2)))
      .collect()
      .toList
      .sortBy(pair => pair._1)
  }

  /**
   * A utility function that writes K-means predictions to the specified directory as csv file(s).
   *
   * @param dataFrame the predictions DataFrame.
   * @param outDir    the directory to write predictions in.
   */
  private def writeDataFrame(dataFrame: DataFrame, outDir: String): Unit = {
    dataFrame.write.option("header", value = true).mode(SaveMode.Overwrite).csv(outDir)
  }

  /**
   * A utility function that writes the cluster centers found by K-means algorithm in csv format (header included).
   *
   * @param centers the cluster centers specified by K-means algorithm execution.
   * @param outFile the file to write centers in.
   */
  private def writeCenters(centers: Array[Vector[Double]], outFile: String): Unit = {
    writeToFile(outFile) { printer =>
      printer.println("x,y,clusterId")
      centers.zipWithIndex.foreach(vector => printer.printf("%f,%f,%d\n", vector._1(0), vector._1(1), vector._2))
    }
  }

  /**
   * A utility function that writes content to the specified file.
   *
   * @param outFile the file to write in.
   * @param op      the operation to perform.
   */
  private def writeToFile(outFile: String)(op: java.io.PrintWriter => Unit): Unit = {
    val p = new PrintWriter(outFile)
    try {
      op(p)
    } finally {
      p.close()
    }
  }
}