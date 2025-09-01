namespace UniCluster.Net;

public class KMeans1D
{
    internal double[] PrefixSums = default!;

    internal double[] PrefixSumOfSquares = default!;

    internal double[,] DpTable = default!;

    public void Fit(double[] values, int numberOfClusters, bool preSortedArray = false)
    {
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
    }

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
                    DpTable[i, k] = ComputeCost(1, i);
                }
                else
                {
                    var minCost = double.PositiveInfinity;

                    for (var j = k - 1; j <= i - 1; j++)
                    {
                        var cost = DpTable[j, k - 1] + ComputeCost(j + 1, i);
                        if (cost < minCost)
                        {
                            minCost = cost;
                        }
                    }

                    DpTable[i, k] = minCost;
                }
            }
        }
    }

    private void InitializeTable(int length, int width)
    {
        DpTable = new double[length, width];

        for (var i = 0; i < length; i++)
        {
            for (var j = 0; j < width; j++)
            {
                if (i == 0 && j == 0)
                {
                    continue;
                }

                DpTable[i, j] = double.PositiveInfinity;
            }
        }
    }

    internal void ComputePrefixSums(double[] values)
    {
        PrefixSums = new double[values.Length + 1];
        PrefixSumOfSquares = new double[values.Length + 1];

        for (var i = 0; i < values.Length; i++)
        {
            PrefixSums[i + 1] = PrefixSums[i] + values[i];
            PrefixSumOfSquares[i + 1] = PrefixSumOfSquares[i] + values[i] * values[i];
        }
    }

    /// <summary>
    /// Given a set x_j, x_j+1, x_j+2, ..., x_i
    /// 
    /// The cost of clustering this set (will be referred to as "cost(j, i)") is the
    /// sum of squared distances from each point to the mean of the set.
    /// 
    /// Therefore cost(j, i) = Sum[t = j -> i](x_t - mean) ^ 2 --> Equation 1
    /// 
    /// The mean of this cluster is (Sum[t = j -> i](x_t)) / n...Where n = i - j + 1. --> Equation 2A
    /// 
    /// We can represent this as mean = (Sum[t = j -> i](x_t) / n) which implies that
    /// 
    /// n * mean = Sum[t = j -> i](x_t). --> Equation 2B
    /// 
    /// Back to Equation 1, if we expand the square it becomes:
    /// 
    /// cost(j, i) = Sum[t = j -> i] (x_t^2 - 2 * x_t * mean + mean^2)
    /// 
    /// cost(j, i) = (Sum[t = j -> i] (x_t^2)) - (2 * mean * Sum[t = j -> i] (x_t)) + (Sum[t = j -> i] (mean^2))
    /// 
    /// For the last part of the equation above, we can simplify Sum[t = j -> i](mean ^ 2) 
    /// to n * (mean ^ 2) since n = i - j + 1.
    /// 
    /// Therefore, the function becomes:
    /// 
    /// cost(j, i) = (Sum[t = j -> i] (x_t^2)) - (2 * mean * Sum[t = j -> i] (x_t)) + (n * (mean ^ 2)) --> Equation 3
    /// 
    /// If we plug equation 2B into Equation 3 we get:
    /// 
    /// cost(j, i) = (Sum[t = j -> i] (x_t^2)) - (2 * n * (mean ^ 2)) + (n * (mean ^ 2))
    /// 
    /// cost(j, i) = (Sum[t = j -> i] (x_t^2)) - (n * (mean ^ 2)) --> Equation 5
    /// 
    /// If we plug equation 2A into Equation 5 we get:
    /// 
    /// cost(j, i) = (Sum[t => j -> i] (x_t^2)) - (((Sum[t => j -> i] (x_t)) ^ 2) / n)
    /// 
    /// Replacing n, we get
    /// 
    /// cost(j, i) = (Sum[t => j -> i] (x_t^2)) - (((Sum[t => j -> i] (x_t)) ^ 2) / (i - j + 1))
    /// 
    /// Which is the cost of clustering the set x_j, x_j+1, ..., x_i into 1 cluster.
    /// </summary>
    internal double ComputeCost(int start, int end)
    {
        var differenceInPrefixSums = PrefixSums[end] - PrefixSums[start - 1];

        return PrefixSumOfSquares[end] - PrefixSumOfSquares[start - 1]
            - (differenceInPrefixSums * differenceInPrefixSums / (end - start + 1));
    }
}
