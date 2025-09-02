namespace UniCluster.Net;

public class OptimalKMeans1D
{
    private double[] _prefixSums = default!;

    private double[] _prefixSumOfSquares = default!;

    private double[,] _dpTable = default!;

    private double[,] _bestSplitIndices = default!;

    internal double[,] DpTable => _dpTable;

    internal double[] PrefixSums => _prefixSums;

    internal double[] PrefixSumOfSquares => _prefixSumOfSquares;

    /// <summary>
    /// Performs optimal 1D K-means clustering on the provided array of values using dynamic programming.
    /// This method finds the globally optimal clustering solution by minimizing the total within-cluster
    /// sum of squared distances (WCSS).
    /// </summary>
    /// <param name="values">
    /// The array of numerical values to be clustered. The values can be in any order as they will be
    /// sorted internally. Must not be null, empty, or contain NaN/Infinity values.
    /// </param>
    /// <param name="k">
    /// The number of clusters to create. Must be a positive integer greater than 0 and less than or
    /// equal to the number of values in the input array.
    /// </param>
    /// <returns>
    /// A <see cref="ClusteringResult"/> containing the optimal clustering solution with k clusters,
    /// including the clusters themselves and the total cost (sum of squared distances).
    /// </returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="values"/> is null.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when:
    /// - <paramref name="values"/> is empty
    /// - <paramref name="k"/> is less than or equal to 0
    /// - <paramref name="k"/> is greater than the number of values
    /// - <paramref name="values"/> contains NaN or Infinity values
    /// </exception>
    /// <remarks>
    /// <para>
    /// This implementation uses the optimal 1D K-means algorithm based on dynamic programming,
    /// which guarantees finding the globally optimal solution. The algorithm has a time complexity
    /// of O(k * n²) where n is the number of data points and k is the number of clusters.
    /// </para>
    /// <para>
    /// The input values are automatically sorted before clustering, so the order of the input
    /// array does not affect the result. Each resulting cluster will contain contiguous values
    /// when sorted.
    /// </para>
    /// <para>
    /// For k=1, the method returns a single cluster containing all values with the centroid
    /// being the arithmetic mean of all values.
    /// </para>
    /// <para>
    /// The algorithm minimizes the within-cluster sum of squares (WCSS), which is equivalent
    /// to minimizing the sum of squared distances from each point to its cluster centroid.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var kmeans = new OptimalKMeans1D();
    /// var values = new double[] { 1.0, 2.0, 8.0, 9.0, 10.0 };
    /// var result = kmeans.Fit(values, k: 2);
    /// 
    /// // Result will have 2 clusters:
    /// // Cluster 1: [1.0, 2.0] with centroid 1.5
    /// // Cluster 2: [8.0, 9.0, 10.0] with centroid 9.0
    /// </code>
    /// </example>
    public ClusteringResult Fit(double[] values, int k)
    {
        ArgumentNullException.ThrowIfNull(values);

        if (values.Length == 0)
        {
            throw new ArgumentException("Input array cannot be empty.", nameof(values));
        }

        if (k <= 0)
        {
            throw new ArgumentException("Number of clusters must be greater than zero.", nameof(k));
        }

        if (k > values.Length)
        {
            throw new ArgumentException("Number of clusters must be less than or equal to the number of values.", nameof(k));
        }

        if (values.Any(v => double.IsNaN(v) || double.IsInfinity(v)))
        {
            throw new ArgumentException("Input array cannot contain NaN or Infinity values.", nameof(values));
        }

        Array.Sort(values);

        if (k == 1)
        {
            var centroid = values.Average();
            return new ClusteringResult([new([..values], centroid)], values.Sum(v => (v - centroid) * (v - centroid)));
        }

        ComputePrefixSums(values);
        ComputeClusterCosts(values, k);

        var clusterPoints = Backtrack(values, k);
        var clusters = new List<Cluster>();

        foreach (var clusterPointsList in clusterPoints)
        {
            var points = clusterPointsList.ToList();
            var centroid = points.Average();
            clusters.Add(new Cluster(points, centroid));
        }

        var totalCost = _dpTable[values.Length, k];

        return new ClusteringResult(clusters, totalCost);
    }

    /// <summary>
    /// Uses dynamic programming to compute the cost of clustering different combinations
    /// of i points into k clusters.
    /// 
    /// The function <see cref="ComputeCost(int, int)"/> is used to compute the cost of clustering
    /// a set x[j], x[j+1], x[j+2], ..., x[i] into 1 cluster.
    /// 
    /// Therefore we can safely assume that for k = 1, and i = [1 -> number of values],
    /// then DP(i, k) = cost(i, 1). This is the cost of clustering i points into 1 cluster.
    /// 
    /// As k grows beyond 1 (1 &lt;= k &lt;= number of clusters), we can use the previously computed
    /// values to find the optimal cost as follows:
    /// 
    /// DP(i, k) = min(for j = [k - 1] -> [i - 1]: [DP(j, k-1) + cost(j+1, i)])
    /// 
    /// because DP(j, k-1) is the optimal cost of the first j points in k-1 clusters, 
    /// 
    /// and cost(j+1, i) is the cost of putting the next block [j+1..i] into one cluster.
    /// 
    /// Consider the following, for k = 3, i = 5
    /// 
    /// To calculate DP(5, 3) marked as an X in the table below:
    /// 
    ///  k 0 1 2 3
    /// i  
    /// 0
    /// 1        
    /// 2      A
    /// 3      B 
    /// 4      C  
    /// 5        X
    /// 
    /// we consider j = {2, 3, 4} (since j goes from k -1 to i -1) and therefore
    /// we consider the cells DP(2, 2), DP(3, 2), DP(4, 2), marked as A, B and C
    /// respectively.
    /// 
    /// which means that if we want to find the OPTIMAL cost of clustering 5 points into 3
    /// clusters, then we need to consider: 
    /// - The precalculated OPTIMAL cost of clustering 4 points into 2 clusters + the cost
    /// of clustering 1 point into 1 cluster.
    /// - The precalculated OPTIMAL cost of clustering 3 points into 2 clusters + the cost
    /// of clustering 2 points into 1 cluster.
    /// - The precalculated OPTIMAL cost of clustering 2 points into 2 clusters + the cost
    /// of clustering 3 points into 1 cluster.
    /// 
    /// and finding the smallest value from the latter 3 values to represent the optimal
    /// cost of DP(5, 3)
    /// </summary>
    internal void ComputeClusterCosts(double[] values, int numberOfClusters)
    {
        int length = values.Length + 1, width = numberOfClusters + 1;

        InitializeTable(length, width);

        for (var k = 1; k < width; k++)
        {
            for (var i = k; i < length; i++)
            {
                if (k == 1)
                {
                    _dpTable[i, k] = ComputeCost(1, i);
                }
                else
                {
                    var minCost = double.PositiveInfinity;

                    var bestIndex = k - 1;

                    for (var j = k - 1; j <= i - 1; j++)
                    {
                        var cost = _dpTable[j, k - 1] + ComputeCost(j + 1, i);
                        if (cost < minCost)
                        {
                            minCost = cost;
                            bestIndex = j;
                        }
                    }

                    _bestSplitIndices[i, k] = bestIndex;

                    _dpTable[i, k] = minCost;
                }
            }
        }
    }

    internal void ComputePrefixSums(double[] values)
    {
        _prefixSums = new double[values.Length + 1];
        _prefixSumOfSquares = new double[values.Length + 1];

        for (var i = 0; i < values.Length; i++)
        {
            _prefixSums[i + 1] = _prefixSums[i] + values[i];
            _prefixSumOfSquares[i + 1] = _prefixSumOfSquares[i] + values[i] * values[i];
        }
    }

    private IEnumerable<IEnumerable<double>> Backtrack(double[] values, int numberOfClusters)
    {
        List<List<double>> clusters = [];

        var i = values.Length;
        for (var k = numberOfClusters; k >= 1; k--)
        {
            var cluster = new List<double>();

            var splitIndex = (int)_bestSplitIndices[i, k];

            for (var j = splitIndex + 1; j <= i; j++)
            {
                cluster.Add(values[j - 1]);
            }

            i = splitIndex;

            clusters.Add(cluster);
        }

        for (var index = clusters.Count - 1; index >= 0; index--)
        {
            yield return clusters[index];
        }
    }

    private void InitializeTable(int length, int width)
    {
        _dpTable = new double[length, width];
        _bestSplitIndices = new double[length, width];

        for (var i = 0; i < length; i++)
        {
            for (var j = 0; j < width; j++)
            {
                if (i == 0 && j == 0)
                {
                    continue;
                }

                _dpTable[i, j] = double.PositiveInfinity;
            }
        }
    }

    /// <summary>
    /// Computes the cost of clustering a contiguous subset of values into a single cluster.
    /// 
    /// <para><strong>Problem:</strong> Given a set x[j], x[j+1], x[j+2], ..., x[i]</para>
    /// 
    /// <para>
    /// The cost of clustering this set (referred to as 'cost(j, i)') is the sum of squared
    /// distances from each point to the mean of the set.
    /// </para>
    /// 
    /// <para><strong>Mathematical Derivation:</strong></para>
    /// 
    /// <para>
    /// <strong>Equation 1:</strong> cost(j, i) = Σ(t=j to i) (x[t] - μ)²
    /// </para>
    /// 
    /// <para>
    /// <strong>Equation 2A:</strong> μ = (Σ(t=j to i) x[t]) / n, where n = i - j + 1 ['n' is the size of the set]
    /// </para>
    /// 
    /// <para>
    /// <strong>Equation 2B:</strong> n * μ = Σ(t=j to i) x[t]
    /// </para>
    /// 
    /// <para>
    /// Expanding the square in Equation 1:
    /// </para>
    /// 
    /// <para>
    /// cost(j, i) = Σ(t=j to i) (x[t]² - 2 * x[t] * μ + μ²)
    /// </para>
    /// 
    /// <para>
    /// cost(j, i) = (Σ(t=j to i) x[t]²) - (2 * μ * Σ(t=j to i) x[t]) + (Σ(t=j to i) μ²)
    /// </para>
    /// 
    /// <para>
    /// Since μ is constant, Σ(t=j to i) μ² simplifies to n * μ² [Recall that 'n' is the size of the set]
    /// </para>
    /// 
    /// <para>
    /// <strong>Equation 3:</strong> cost(j, i) = (Σ(t=j to i) x[t]²) - (2 * μ * Σ(t=j to i) x[t]) + (n * μ²)
    /// </para>
    /// 
    /// <para>
    /// Substituting Equation 2B into Equation 3:
    /// </para>
    /// 
    /// <para>
    /// cost(j, i) = (Σ(t=j to i) x[t]²) - (2 * n * μ²) + (n * μ²)
    /// </para>
    /// 
    /// <para>
    /// <strong>Equation 5:</strong> cost(j, i) = (Σ(t=j to i) x[t]²) - (n * μ²)
    /// </para>
    /// 
    /// <para>
    /// Substituting Equation 2A into Equation 5:
    /// </para>
    /// 
    /// <para>
    /// cost(j, i) = (Σ(t=j to i) x[t]²) - (((Σ(t=j to i) x[t])²) / n)
    /// </para>
    /// 
    /// <para>
    /// <strong>Final Formula:</strong> cost(j, i) = (Σ(t=j to i) x[t]²) - (((Σ(t=j to i) x[t])²) / (i - j + 1))
    /// </para>
    /// 
    /// <para>
    /// This represents the cost of clustering the contiguous set x[j], x[j+1], ..., x[i] into 1 cluster.
    /// </para>
    /// </summary>
    internal double ComputeCost(int start, int end)
    {
        var differenceInPrefixSums = _prefixSums[end] - _prefixSums[start - 1];

        return _prefixSumOfSquares[end] - _prefixSumOfSquares[start - 1]
            - (differenceInPrefixSums * differenceInPrefixSums / (end - start + 1));
    }
}
