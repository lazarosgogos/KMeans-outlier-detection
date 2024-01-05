import org.apache.spark.SparkContext
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

import java.io.PrintWriter

/**
 * Data WareHouses & Data Mining Course, AUTh
 * Semester programming assignment (Winter 2023-2024)
 * @author Vasileios Papastergios (ID: 3651), Lazaros Gogos (ID: 3877)
 */
object Main {
  private var xMin: Double = 0.0
  private var xMax: Double = 1.0
  private var yMin: Double = 0.0
  private var yMax: Double = 1.0

  private final val NANOS_PER_SEC: Double = 1e9d

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

    val duration = (System.nanoTime - startTimestamp) / NANOS_PER_SEC
    println(s"\nTotal execution time: $duration sec")
    sc.stop(0)
  }

  /**
   * A utility function that initializes a SparkSession instance with defined parameters.
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
   * A utility function that extracts the input file name from command line arguments array, if specified.
   * In case there are no command line arguments, a default file name is used (data.csv).
   * Corresponds to assignment TASK 1.
   *
   * @param args the command line arguments of function main.
   * @return the file name as String (if existent in arguments), else a default file name "data.csv".
   */
  private def getInputFileName(args: Array[String]): String = {
    val defaultFileName = "data.csv"
    if (args.length == 0) {
      return defaultFileName
    }
    args(0)
  }

  /**
   * A utility function that is responsible for reading the input data file and constructing an RDD with the data.
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
   * A utility function that preforms data cleaning on an RDD[String] containing "x,y" coordinates.
   * The function removes lines with missing values, splits the "x,y" String per row to (Double, Double)
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
   * A utility function that, given an RDD[String] containing "x,y" coordinates, filters out lines that either
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
   * A utility function that, given an RDD[String] containing "x,y" coordinates, splits the coordinates
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
   * A utility function that scales a given RDD containing coordinates to [0, 1] in both dimensions.
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
   * A utility function that computes the minimum and maximum value for each dimension, given an RDD that contains
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
   * A utility function that transforms a given, scaled (x, y) pair to its original values
   * (before MinMax scaling). The function uses the Object's attributes xMin, xMax, yMin, yMax
   * to perform the inverse transformation.
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
   * A utility function that creates a DataFrame from a RDD[(Double, Double)] containing coordinates.
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
    val predictions = model.transform(features).selectExpr("x as xScaled", "y as yScaled", "prediction as clusterId")
    writePredictions(dataFrame = predictions, outDir = predictionsDir)

    // Retrieve original coordinates of the cluster centers
    val originalCenters = model.clusterCenters.map(scaledCoordinates => getOriginalCoordinates(scaledCoordinates(0), scaledCoordinates(1)))
    writeCenters(centers = originalCenters, outFile = centersOutFile)

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

    val meanDistances = predictionsExpanded.map(row => (row.getInt(2), row.getDouble(7)))
      .mapValues(distance => (distance, 1))
      .reduceByKey((pair1, pair2) => (pair1._1 + pair2._1, pair1._2 + pair2._2))
      .mapValues(pair => pair._1 / pair._2)
      .collect()
      .toList
      .sortBy(pair => pair._1)

    val stdDistances = predictionsExpanded.map(row => (row.getInt(2), row.getDouble(7)))
      .mapValues(x => (1, x, x * x))
      .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2, x._3 + y._3))
      .mapValues(x => math.sqrt(x._3 / x._1 - math.pow(x._2 / x._1, 2)))
      .collect()
      .toList
      .sortBy(pair => pair._1)

    printf("Detected Outliers:\n\n")
    predictionsExpanded.filter(row => row.getDouble(7) > meanDistances(row.getInt(2))._2 + 3.5 * stdDistances(row.getInt(2))._2)
      .foreach(row => printf("(%.5f, %.5f)\n", row.getDouble(3), row.getDouble(4)))
  }

  /**
   * A utility function that writes K-means predictions to the specified directory as csv file(s).
   *
   * @param dataFrame the predictions DataFrame.
   * @param outDir    the directory to write predictions in.
   */
  private def writePredictions(dataFrame: DataFrame, outDir: String): Unit = {
    dataFrame.write.mode(SaveMode.Overwrite).csv(outDir)
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