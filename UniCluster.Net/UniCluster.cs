namespace UniCluster.Net;

public class UniCluster
{
    private double[] _prefixSums = default!;

    private double[] _prefixSumOfSquares = default!;

    //TODO: Emphasize that values should be sorted
    public void Fit(double[] values, int numberOfClusters)
    {
        if (numberOfClusters > values.Length)
        {
            throw new ArgumentException("Number of clusters must be less than or equal to the number of values.");
        }

    }

    private void ComputePrefixSums(double[] values)
    {
        _prefixSums = new double[values.Length + 1];
        _prefixSumOfSquares = new double[values.Length + 1];

        for (var i = 0; i < values.Length; i++)
        {
            _prefixSums[i + 1] = _prefixSums[i] + values[i];
            _prefixSumOfSquares[i + 1] = _prefixSumOfSquares[i] + values[i] * values[i];
        }
    }

    private double ComputeCost(int j, int i)
    {
        return _prefixSumOfSquares[i] - _prefixSumOfSquares[j]
            - (Math.Pow(_prefixSums[i] - _prefixSums[j - 1], 2) / (i - j + 1));
    }
}
