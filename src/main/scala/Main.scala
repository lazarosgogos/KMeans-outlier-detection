object Main {
  def main(args: Array[String]): Unit = {
    val inputfile = "points.csv"
    val src = scala.io.Source.fromFile(inputfile)
    val lines = src.getLines()

    val validPoints = lines.flatMap(parsePoint).toList

    println("length:" + validPoints.length)
  }

  def parsePoint(line: String): Option[(Double, Double)] = {
    // Split the line based on comma
    val values = line.split(",")

    // Check if the line has both x and y values
    if (values.length == 2) {
      try {
        val x = values(0).trim.toDouble
        val y = values(1).trim.toDouble
        Some((x,y))
      } catch {
        case e: NumberFormatException =>
          // Ignore lines that cannot be parsed as doubles
          println("exception:"+line)
          None
      }
    } else {
      // Ignore lines that do not have both x and y values
      println("Length not 2:"+line)
      None
    }

  }
}