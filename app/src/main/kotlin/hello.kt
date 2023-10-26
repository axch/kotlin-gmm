// -*- kotlin-tab-width: 2 -*-

import java.util.*
import java.io.File
import kotlin.math.*
import org.jetbrains.letsPlot.geom.*
import org.jetbrains.letsPlot.letsPlot
import org.jetbrains.letsPlot.commons.encoding.Base64
import org.jetbrains.letsPlot.core.plot.export.PlotImageExport
import org.jetbrains.letsPlot.intern.Plot
import org.jetbrains.letsPlot.intern.toSpec

// Basic stats

data class Gaussian(val mean: Double, val stddev: Double)

fun gaussianDensity(x: Double) : Double {
  return exp(- x * x / 2) / sqrt(2 * PI)
}

fun logpGaussian(x: Double, params: Gaussian) : Double {
  val mean = params.mean
  val stddev = params.stddev
  val z = (x - mean) / stddev
  val prob = - z * z / 2 - ln(stddev)
  return prob - 0.5 * ln(2 * PI)
}

fun logsumexp(xs: Collection<Double>) : Double {
  val max = xs.max()
  return ln(xs.map {exp(it - max)}.sum()) + max
}

// Draw a sample from a categorical distribution whose (un-normalized)
// log weights are given by the argument.
fun sampleCategoricalByLog(rng: Random, logps: Collection<Double>) : Int {
  val norm = logsumexp(logps)
  val ps = logps.map { exp(it - norm) }
  return sampleCategoricalNormalized(rng, ps)
}

fun sampleCategoricalNormalized(rng: Random, ps: Collection<Double>) : Int {
  var x = rng.nextDouble()
  var ind = 0
  for (p in ps) {
    if (x <= p) {
      return ind
    } else {
      ind += 1
      x -= p
    }
  }
  // Only happens if the normalization was wrong, e.g. due to
  // numerical error.
  return 0
}

// Iterating Doubles

infix fun ClosedRange<Double>.step(step: Double): Iterable<Double> {
  require(start.isFinite())
  require(endInclusive.isFinite())
  require(step > 0.0) { "Step must be positive, was: $step." }
  val sequence = generateSequence(start) { previous ->
    if (previous == Double.POSITIVE_INFINITY) return@generateSequence null
    val next = previous + step
    if (next > endInclusive) null else next
  }
  return sequence.asIterable()
}

// GMM functions

fun sampleAssignment(rng: Random, npoints: Int, nclusters: Int) : IntArray {
  // There will be room to add a prior on the peakiness of the
  // clustering distribution, but for now just assign uniformly.
  return IntArray(npoints) { rng.nextInt(nclusters) }
}

fun sampleParameters(rng: Random, nclusters: Int) : Array<Gaussian> {
  // Here I really do want to eventually make room for a prior on the
  // cluster parameters, probably the usual conjugate
  // normal-inverse-gamma nonsense.  But not yet.
  return Array(nclusters) { sampleParametersOne(rng) }
}

// Sample the parameters of one cluster from the prior
fun sampleParametersOne(rng: Random) : Gaussian {
  // Hard-code a Gaussian prior on the mean
  val mean = 10.0 * rng.nextGaussian()
  // The Java ecosystem doesn't have a library implementation of the
  // inverse Gamma sampler?  Seriously?  OK, hack a uniform for now.
  val stddev = 1.0 + 10.0 * rng.nextDouble()
  return Gaussian(mean, stddev)
}

fun samplePositionsGivenAssignmentParameters(
    rng: Random, assignment: IntArray, parameters: Array<Gaussian>)
    : DoubleArray {
  // Question: Kotlin's inline value classes are supposed to be like
  // Haskell newtypes.  But, if I make an inline value class around a
  // primitive Double, to rename it to "Position", will an Array of
  // those end up with the same performance as a DoubleArray, or not?
  return DoubleArray(assignment.size) { index ->
    val cluster = assignment[index]
    val params = parameters[cluster]
    samplePositionGivenParameters(rng, params)
  }
}

fun samplePositionGivenParameters(rng: Random, params: Gaussian) : Double {
  return params.mean + params.stddev * rng.nextGaussian()
}

fun logpPositionGivenParametersIntegratingAssignment(
    position: Double, params: Array<Gaussian>) : Double {
  val pCluster = 1.0 / params.size  // Hardcoding equal cluster probs
  return logsumexp(params.map { ln(pCluster) + logpGaussian(position, it) })
}

fun sampleAssignmentGivenPositionsParameters(
    rng: Random, positions: DoubleArray, parameters: Array<Gaussian>)
    : IntArray {
  // This is only independent across points if that's true of the
  // cluster prior, so will need to change if we move to a Dirichlet
  // process mixture.
  return IntArray(positions.size) {
    sampleOneAssignmentGivenPositionParameters(rng, positions[it], parameters)
  }
}

fun sampleOneAssignmentGivenPositionParameters(
    rng: Random, position: Double, parameters: Array<Gaussian>)
    : Int {
  // Could put the cluster assignment prior here, but while it's
  // uniform it cancels out.
  // TODO: Use a real DoubleArray here.  Can pull that off by adding
  // the obvious mapDouble method to Array<T>, but in general will
  // need a quadratic number of these.
  val probs = parameters.map { logpGaussian(position, it) }
  return sampleCategoricalByLog(rng, probs)
}

// Plotting

fun histogram(dat: Collection<Double>) : Plot {
  val datMap = mapOf("value" to dat)
  var p = letsPlot(datMap)
  p += geomHistogram(bins=250, color = "dark_green", alpha = .3) { x = "value"; y = "..density.." }
  return p
}

fun histogram(dat: Collection<Double>, density: (Double) -> Double) : Plot {
  var p = histogram(dat)

  val values = (dat.min())..(dat.max()) step 0.01
  val densityVals = mapOf(
      "value" to values,
      "density" to values.map { density(it) }
  )

  p += geomLine(data=densityVals) { x = "value"; y = "density" }
  return p
}

fun writePlot(p: Plot, name: String) { 
  val image = PlotImageExport.buildImageFromRawSpecs(
      plotSpec = p.toSpec(),
      format = PlotImageExport.Format.PNG,
      scalingFactor = 2.0,
      targetDPI = Double.NaN
  )

  val file = File(".", name)
  file.createNewFile()
  file.writeBytes(image.bytes)
  println("Gaussian histogram drawn in " + file.getCanonicalPath())
}

// Driving

fun gaussianHistogram(sz: Int) {
  val rng = Random(1L)
  val dat = List(sz) { rng.nextGaussian() }

  val p = histogram(dat, ::gaussianDensity)
  writePlot(p, "gaussian.png")
}

fun main() {
  // gaussianHistogram(100000)
  val npoints = 1000000
  val nclusters = 4
  val rng = Random(1L)
  val params = sampleParameters(rng, nclusters)
  for (p in params) {
    println(p)
  }
  val assignment = sampleAssignment(rng, npoints, nclusters)
  val positions = samplePositionsGivenAssignmentParameters(rng, assignment, params)
  println(positions.toList().max())
  val p = histogram(positions.toList()) { exp(logpPositionGivenParametersIntegratingAssignment(it, params)) }
  writePlot(p, "two-gaussians.png")
}
