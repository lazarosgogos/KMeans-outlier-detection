
import org.apache.spark.SparkContext._
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

    var inputFile = "points.csv"
    if (args.length > 0){
      inputFile = args(0)
      println(args(0))
    }

    //    Read input file
    val rawDataRDD = sc.textFile(inputFile)

    //    Data cleaning
    val clearDataRDD = rawDataRDD.filter(line => !line.startsWith(",") && !line.endsWith(",") && line.nonEmpty)

    //    Data splitting
    val validCoordinatesRDD = clearDataRDD.map(line => {
      val numbers = line.split(",")
      (numbers(0).trim.toDouble, numbers(1).trim.toDouble)
    })

    //    Data transformation
    val xCoordinatesRDD = validCoordinatesRDD.map(coordinates => coordinates._1)
    val yCoordinatesRDD = validCoordinatesRDD.map(coordinates => coordinates._2)
    val minX = xCoordinatesRDD.min
    val maxX = xCoordinatesRDD.max
    val minY = yCoordinatesRDD.min
    val maxY = yCoordinatesRDD.max
    val transformedCoordinatesRDD = validCoordinatesRDD.map(coordinates => {
      val x = coordinates._1
      val y = coordinates._2
      ((x - minX)/(maxX - minX), (y - minY)/(maxY - minY))
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
}