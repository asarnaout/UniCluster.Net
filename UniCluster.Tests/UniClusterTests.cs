using UniCluster.Net;

namespace UniCluster.Tests;

public class UniClusterTests
{
    [Fact]
    public void ItGeneratesThePrefixSums()
    {
        var uniCluster = new KMeans1D();

        uniCluster.ComputePrefixSums([1, 2, 3, 10, 11, 12]);

        double[] expectedSums = [0, 1, 3, 6, 16, 27, 39];

        Assert.Equal(expectedSums, uniCluster.PrefixSums);
    }

    [Fact]
    public void ItGeneratesThePrefixSumSquares()
    {
        var uniCluster = new KMeans1D();

        uniCluster.ComputePrefixSums([1, 2, 3, 10, 11, 12]);

        double[] expectedSumOfSquares = [0, 1, 5, 14, 114, 235, 379];

        Assert.Equal(expectedSumOfSquares, uniCluster.PrefixSumOfSquares);
    }
}