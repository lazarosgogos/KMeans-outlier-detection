
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql._
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
object Main {

  def main(args: Array[String]): Unit = {
    val ss = SparkSession.builder()
      .master("local[2]")
      .appName("PapaGo_Clustering")
      .getOrCreate()
    val sc = ss.sparkContext

    //    1. Read input file
    val inputFile = getInputFileName(args)
    val rawDataRDD = sc.textFile(inputFile)

    //    2. Perform data cleaning
    //      2a. Filter out lines with missing values
    val cleanDataRDD = rawDataRDD.filter(line =>
      !line.startsWith(",") &&
      !line.endsWith(",") &&
       line.nonEmpty
    )

    //      2b. Perform data splitting
    val validCoordinatesRDD = cleanDataRDD.map(line => {
      val coordinates = line.split(",")
      (coordinates(0).trim.toDouble, coordinates(1).trim.toDouble)
    })

    //    3. Transform data to [0, 1]
    val xCoordinatesRDD = validCoordinatesRDD.map(coordinates => coordinates._1)
    val yCoordinatesRDD = validCoordinatesRDD.map(coordinates => coordinates._2)
    val xMin = xCoordinatesRDD.min
    val xMax = xCoordinatesRDD.max
    val yMin = yCoordinatesRDD.min
    val yMax = yCoordinatesRDD.max
    val transformedCoordinatesRDD = validCoordinatesRDD.map(coordinates => {
      val x = coordinates._1
      val y = coordinates._2
      ((x - xMin)/(xMax - xMin), (y - yMin)/(yMax - yMin))
    })

    // ----------------------- K MEANS -----------------------
    // for more info refer to:
    // https://spark.apache.org/docs/latest/ml-clustering.html
    val schema = StructType(
    StructField(name = "x", dataType = DoubleType, nullable = false) ::
    StructField(name = "y", dataType = DoubleType, nullable = false) :: Nil
    )
    // How and why?
    // here's the answer
    // https://stackoverflow.com/questions/41042809/tuple-to-data-frame-in-spark-scala

    val data = transformedCoordinatesRDD.map(line => Row(
      line._1.asInstanceOf[Number].doubleValue(),
      line._2.asInstanceOf[Number].doubleValue()
    ))
    val dataFrame = ss.createDataFrame(data, schema)

    //    df.head(50).foreach(println)
    val kMeans = new KMeans().setK(20).setSeed(199L)

    //creating features column
    val assembler = new VectorAssembler()
      .setInputCols(Array("x", "y"))
      .setOutputCol("features")

    val features = assembler.transform(dataFrame)
    // Train the model
    val model = kMeans.fit(features)


    // Make predictions
    val predictions = model.transform(features)

    val evaluator = new ClusteringEvaluator()

    val silhouette = evaluator.evaluate(predictions)
    println(s"Silhouette with squared euclidean distance = $silhouette")

    // show the result
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)
  }

  def getInputFileName(args: Array[String]) : String = {
    var defaultFileName = "points.csv"
    if (args.length == 0) {
      defaultFileName
      return defaultFileName
    }
    args(0)
  }
}