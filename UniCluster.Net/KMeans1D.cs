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

    internal double ComputeCost(int start, int end)
    {
        var differenceInPrefixSums = PrefixSums[end] - PrefixSums[start - 1];

        return PrefixSumOfSquares[end] - PrefixSumOfSquares[start - 1]
            - (differenceInPrefixSums * differenceInPrefixSums / (end - start + 1));
    }
}
