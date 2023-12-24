
import org.apache.spark.SparkContext._
import org.apache.spark.{SparkConf, SparkContext}
object Main {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setMaster("local[2]") // run using 2 threads, use local[*] to run with as many threads as possible
      .setAppName("DataMinerApp")

    val sc = new SparkContext()

    val inputfile = "points.csv"
    val src = scala.io.Source.fromFile(inputfile)
    val lines = src.getLines()

    val validPoints = lines.flatMap(parsePoint).toList
    val (minX, minY, maxX, maxY) = findMinMaxValues(validPoints)

    var scaledPoints = List[(Double, Double)]()
    validPoints.foreach(p => {
      var x = p._1
      var y = p._2
      x = scaleValue(x, minX, maxX)
      y = scaleValue(y, minY, maxY)
      val newP = (x,y)
      scaledPoints = scaledPoints :+ newP
    })

//    scaledPoints.foreach(println)
  }

  def scaleValue(value: Double, min: Double, max: Double): Double = {
    (value - min) / (max - min)
  }

  def findMinMaxValues(points: List[(Double, Double)]): (Double, Double, Double, Double) = {
    var minX = Double.MaxValue
    var minY = Double.MaxValue
    var maxX = Double.MinValue
    var maxY = Double.MinValue
    points.foreach(point => {
      val x = point._1
      val y = point._2
      // update min and max values
      minX = math.min(minX, x)
      minY = math.min(minY, y)
      maxX = math.max(maxX, x)
      maxY = math.max(maxY, y)
    })
    (minX, minY, maxX, maxY)
  }

  def parsePoint(line: String): Option[(Double, Double)] = {
    // Split the line based on comma
    val values = line.split(",")

    // Check if the line has both x and y values
    if (values.length == 2) {
      try {
        val x = values(0).trim.toDouble
        val y = values(1).trim.toDouble
        Some((x, y))
      } catch {
        case e: NumberFormatException =>
          // Ignore lines that cannot be parsed as doubles
//          println("exception:" + line)
          None
      }
    } else {
      // Ignore lines that do not have both x and y values
//      println("Length not 2:" + line)
      None
    }
  }
}