# Learning Kotlin by implementing a Gaussian mixture model in it

Status quo
- We have a 1-D Gaussian mixture model with a conjugate component prior
- The cluster weights sizes are hard-coded to be equal
- We have a fully-uncollapsed Gibbs sampler for the model
- It visually fits on a couple synthetic data sets in a few sweeps

Things that could reasonably be improved:

1. Learn cluster weights
    - Add a Dirichlet prior on the cluster weights
    - Plumb the weight sample around to the functions that need it
    - Add a step to Gibbs to resample the weights given the current
      assignment of points to clusters
    - Update the data generation to generate unbalanced clusters
    - Would exercise Kotlin refactoring support

2. Cluster multi-dimensional data
    - Upgrade to multidimensional Gaussians
    - Maybe upgrade to a Wishart distribution in the prior if we allow
      covariance across dimensions
    - Would exercise type-level support for getting the data dimension
      right
    - May exercise taking Java dependencies if I want to stop coding
      my own basic samplers (and if Apache Commons or someone even has
      the Wishart)
    - Would further exercise plotting

3. Add stopping control to provide a tuning-free "fit this model to
data" operation
    - Research and choose an MCMC stopping control method (maybe R-hat on
      parameters? maybe something else?)
    - Implement it
    - Validate that it really does consistently fit well
      - Across fitting runs
      - Across data sets
      - On real data?
    - Would further exercise plotting, and practice MCMC debugging methods
    - This may require scaling work, depending on the computational cost
      of the method and the validation

4. Scale up
    - Make sure the hot paths consistently use good data structures
      (no intermediate Lists, please, unless they're more efficient in
      Kotlin than I would think)
      - Are there places where I should eliminate intermediate arrays
        by updating in-place?
    - Exploit parallelism
    - Would exercise Kotlin performance measurement
    - Would exercise Kotlin parallelism support
    - Could exercise my ability to do the structure of arrays
      transformation in user-space without breaking my abstractions
      all over the place

5. Add unit tests
    - The gamma sampler actually had a bug in it, so could unit-test it
      further
    - The normal-gamma update function has intricate formulas, could
      unit-test that it works well in isolation (more than just
      eyeballing how well the model performs when hard-coded to one
      cluster)
    - Could add a (some?) Geweke-style test that the Gibbs sampler is
      consistent with the forward model
    - The histogram plot seems to be a little off, maybe overlaps bins
      or something?  Is there a way to test / debug it?  Like maybe check
      that its reported bin areas add up to 1?

6. Pick nits
    - Why does ./gradlew run make the working directory be app/?  I don't
      want the plots files to keep ending up there all the time.
