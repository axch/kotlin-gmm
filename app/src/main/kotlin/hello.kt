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

fun sampleGaussian(rng: Random, dist: Gaussian) : Double {
  return dist.mean + rng.nextGaussian() * dist.stddev
}

fun unitGaussianDensity(x: Double) : Double {
  return exp(- x * x / 2) / sqrt(2 * PI)
}

fun logpGaussian(x: Double, params: Gaussian) : Double {
  val mean = params.mean
  val stddev = params.stddev
  val z = (x - mean) / stddev
  val prob = - z * z / 2 - ln(stddev)
  return prob - 0.5 * ln(2 * PI)
}

data class GaussianStat(var count: Int, var total: Double, var discrep: Double) {
  constructor() : this(0, 0.0, 0.0)

  fun incorporate(x: Double) {
    // discrepancy is always n times the variance.  This is Q in
    // https://en.wikipedia.org/wiki/Standard_deviation#Rapid_calculation_methods
    if (this.count > 0) {
      val prev_mean = this.mean()
      this.discrep += (x - prev_mean) * (x - prev_mean) * this.count / (this.count + 1)
    }
    this.count += 1
    this.total += x
  }

  fun mean() : Double {
    return this.total / this.count
  }

  fun variance() : Double {
    return this.discrep / this.count
  }

  fun discrepancy() : Double {
    return this.discrep
  }
}

// This function solves the Gaussian-Gaussian model with known
// observation standard deviation.
//
// The model posits an unknown mean with a Gaussian prior, and an IID
// Gaussian likelihood with that mean and a given variance.
//
// Given the prior, the standard deviation, and a GaussianStat object
// summarizing the observed data, return the posterior on the mean
// (which is also a Gaussian).
fun conjugateUpdateGaussianGaussian(
    prior: Gaussian, likelihoodStdDev: Double, stats: GaussianStat) : Gaussian {
  if (stats.count == 0) { return prior }
  val priorPrec = 1.0 / (prior.stddev * prior.stddev)
  val dataPrec = stats.count / (likelihoodStdDev * likelihoodStdDev)
  val mean = (priorPrec * prior.mean + dataPrec * stats.mean()) / (priorPrec + dataPrec)
  val prec = priorPrec + dataPrec
  return Gaussian(mean,  sqrt(1.0 / prec))
}

// Transcribed from Apache commons-rng, since I don't want to bother
// reading the original paper again.  Citation: Marsaglia and Tsang, A
// Simple Method for Generating Gamma Variables. ACM Transactions on
// Mathematical Software, Volume 26 Issue 3, September, 2000.
// The parameterization is in terms of the shape alpha, assuming
// constant scale.
fun sampleMarsagliaTsangGamma(rng: Random, alpha: Double) : Double {
  val dOptim = alpha - 1.0 / 3
  val cOptim = 1.0 / (3 * sqrt(dOptim))
  while (true) {
    val unitG = rng.nextGaussian()
    val oPcTx = 1 + cOptim * unitG
    val v = oPcTx * oPcTx * oPcTx
    if (v <= 0) continue
    val x2 = unitG * unitG
    val u = rng.nextDouble()
    // Squeeze
    if (u < 1 - 0.0331 * x2 * x2) {
      return dOptim * v
    }
    if (ln(u) < 0.5 * x2 + dOptim * (1 - v + ln(v))) {
      return dOptim * v
    }
  }
}

// This represents an instance of the normal-gamma distribution.  This
// distribution is a particular family of non-independent two-valued
// distribution that is useful because it provides a conjugate prior
// to a normal distribution with unknown mean and precision.
//
// To wit, the model is
// prec ~ Gamma(alpha, beta)
// mu ~ Normal(mean, precision=pseudocount * prec)
data class NormalGamma(
    val mean: Double, val pseudocount: Double,
    val alpha: Double, val beta: Double) {

  // Draw (the parameters of) a Gaussian from this distribution
  fun sampleGaussian(rng: Random) : Gaussian {
    val stdGamma = sampleMarsagliaTsangGamma(rng, this.alpha)
    val prec = stdGamma / this.beta  // beta is the rate, i.e., inverse scale
    val stddev = sqrt(1.0 / prec)
    val mean = sampleGaussian(rng, Gaussian(this.mean, stddev * 1.0/sqrt(this.pseudocount)))
    return Gaussian(mean, stddev)
  }
}

// I got these formulas off the Internet.  TODO: Unit test
fun conjugateUpdateNormalGammaGaussian(
    prior: NormalGamma, stats: GaussianStat) : NormalGamma {
  if (stats.count == 0) return prior
  val newAlpha = prior.alpha + stats.count / 2.0
  val discrepancy = stats.mean() - prior.mean
  val discrepancyAdj = discrepancy * discrepancy * stats.count * prior.pseudocount * 0.5 / (stats.count + prior.pseudocount)
  val newBeta = prior.beta + 0.5 * stats.discrepancy() + discrepancyAdj
  val newMean = (stats.total + prior.mean * prior.pseudocount) / (stats.count + prior.pseudocount)
  return NormalGamma(newMean, prior.pseudocount + stats.count, newAlpha, newBeta)
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
  val logps = params.map { ln(pCluster) + logpGaussian(position, it) }
  return logsumexp(logps)
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

fun sampleParametersGivenPositionsAssignment(
    rng: Random, nclusters: Int, positions: DoubleArray, assignment: IntArray)
    : Array<Gaussian> {
  val stats = Array (nclusters) { GaussianStat() }
  for (i in 0..positions.size-1) {
    stats[assignment[i]].incorporate(positions[i])
  }
  return Array(stats.size) { sampleOneParametersGivenStats(rng, stats[it]) }
}

fun sampleOneParametersGivenStats(rng: Random, stats: GaussianStat) : Gaussian {
  // TODO: This hardcodes a different cluster parameter prior than the
  // one used to generate the data.
  val prior = NormalGamma(0.0, 1.0, 1.0, 1.0)
  val posterior = conjugateUpdateNormalGammaGaussian(prior, stats)
  val ans = posterior.sampleGaussian(rng)
  return ans
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

  val p = histogram(dat, ::unitGaussianDensity)
  writePlot(p, "gaussian.png")
}

fun synthesizeData(rng: Random, nclusters: Int, npoints: Int) : DoubleArray {
  val params = sampleParameters(rng, nclusters)
  println("Generating data with true clusters")
  for (p in params) {
    println(p)
  }
  val assignment = sampleAssignment(rng, npoints, nclusters)
  val positions = samplePositionsGivenAssignmentParameters(rng, assignment, params)
  println(positions.toList().max())
  val mean = positions.sum() / positions.size
  println("Data mean " + mean)
  val discrep = positions.map { (it - mean) * (it - mean) }.sum()
  println("Data discrepancy " + discrep)
  return positions
}

fun fitGibbs(rng: Random, nclusters: Int, nsteps: Int, positions: DoubleArray)
    : Array<Gaussian> {
  val npoints = positions.size
  var params = sampleParameters(rng, nclusters)
  var assignment = sampleAssignmentGivenPositionsParameters(rng, positions, params)
  for (i in 1..nsteps) {
    println("Starting sweep " + i)
    params = sampleParametersGivenPositionsAssignment(
        rng, nclusters, positions, assignment)
    assignment = sampleAssignmentGivenPositionsParameters(rng, positions, params)
  }
  // TODO Should also return the assignment, and diagnostics of the
  // fitting process
  return params
}

fun tryClustering() {
  val npoints = 10000
  val nclusters = 2
  val rng = Random(1L)
  val positions = synthesizeData(rng, nclusters, npoints)

  val params = fitGibbs(rng, nclusters, 50, positions)
  for (p in params) {
    println(p)
  }
  val p = histogram(positions.toList()) { exp(logpPositionGivenParametersIntegratingAssignment(it, params)) }
  writePlot(p, "clustering-fit.png")
}

fun gammaHistogram() {
  val rng = Random(1L)
  val dat = List(10000) { sampleMarsagliaTsangGamma(rng, 5.0) }

  val p = histogram(dat)
  writePlot(p, "gamma.png")
}

fun main() {
  // gaussianHistogram(100000)
  tryClustering()
  // gammaHistogram()
}
