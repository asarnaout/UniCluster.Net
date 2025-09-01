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

    public ClusteringResult Fit(double[] values, int numberOfClusters, bool preSortedArray = false)
    {
        ArgumentNullException.ThrowIfNull(values);

        if (values.Length == 0)
        {
            throw new ArgumentException("Input array cannot be empty.", nameof(values));
        }

        if (numberOfClusters <= 0)
        {
            throw new ArgumentException("Number of clusters must be greater than zero.", nameof(numberOfClusters));
        }

        if (numberOfClusters > values.Length)
        {
            throw new ArgumentException("Number of clusters must be less than or equal to the number of values.");
        }

        if (!preSortedArray)
        {
            Array.Sort(values);
        }

        ComputePrefixSums(values);
        ComputeClusterCosts(values, numberOfClusters);

        var clusterPoints = Backtrack(values, numberOfClusters);
        var clusters = new List<Cluster>();

        foreach (var clusterPointsList in clusterPoints)
        {
            var points = clusterPointsList.ToList();
            var centroid = points.Average();
            clusters.Add(new Cluster(points, centroid));
        }

        var totalCost = _dpTable[values.Length, numberOfClusters];

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
    /// As k grows beyond 1 (1 <= k <= number of clusters), we can use the previously computed
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
    /// Given a set x[j], x[j+1], x[j+2], ..., x[i]
    /// 
    /// The cost of clustering this set (will be referred to as "cost(j, i)") is the
    /// sum of squared distances from each point to the mean of the set.
    /// 
    /// Therefore cost(j, i) = Sum[t = j -> i](x[t] - mean) ^ 2 --> Equation 1
    /// 
    /// The mean of this cluster is (Sum[t = j -> i](x[t])) / n...Where n = i - j + 1. --> Equation 2A
    /// 
    /// We can represent this as mean = (Sum[t = j -> i](x[t]) / n) which implies that
    /// 
    /// n * mean = Sum[t = j -> i](x[t]). --> Equation 2B
    /// 
    /// Back to Equation 1, if we expand the square it becomes:
    /// 
    /// cost(j, i) = Sum[t = j -> i] (x[t]^2 - 2 * x[t] * mean + mean^2)
    /// 
    /// cost(j, i) = (Sum[t = j -> i] (x[t]^2)) - (2 * mean * Sum[t = j -> i] (x[t])) + (Sum[t = j -> i] (mean^2))
    /// 
    /// For the last part of the equation above, we can simplify Sum[t = j -> i](mean ^ 2) 
    /// to n * (mean ^ 2) since n = i - j + 1.
    /// 
    /// Therefore, the function becomes:
    /// 
    /// cost(j, i) = (Sum[t = j -> i] (x[t]^2)) - (2 * mean * Sum[t = j -> i] (x[t])) + (n * (mean ^ 2)) --> Equation 3
    /// 
    /// If we plug equation 2B into Equation 3 we get:
    /// 
    /// cost(j, i) = (Sum[t = j -> i] (x[t]^2)) - (2 * n * (mean ^ 2)) + (n * (mean ^ 2))
    /// 
    /// cost(j, i) = (Sum[t = j -> i] (x[t]^2)) - (n * (mean ^ 2)) --> Equation 5
    /// 
    /// If we plug equation 2A into Equation 5 we get:
    /// 
    /// cost(j, i) = (Sum[t => j -> i] (x[t]^2)) - (((Sum[t => j -> i] (x[t])) ^ 2) / n)
    /// 
    /// Replacing n, we get
    /// 
    /// cost(j, i) = (Sum[t => j -> i] (x[t]^2)) - (((Sum[t => j -> i] (x[t])) ^ 2) / (i - j + 1))
    /// 
    /// Which is the cost of clustering the set x_j, x_j+1, ..., x_i into 1 cluster.
    /// </summary>
    internal double ComputeCost(int start, int end)
    {
        var differenceInPrefixSums = _prefixSums[end] - _prefixSums[start - 1];

        return _prefixSumOfSquares[end] - _prefixSumOfSquares[start - 1]
            - (differenceInPrefixSums * differenceInPrefixSums / (end - start + 1));
    }
}
