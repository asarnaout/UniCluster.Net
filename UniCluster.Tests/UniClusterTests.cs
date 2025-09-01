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

    [Fact]
    public void ComputeClusterCosts_WithSingleCluster_ShouldCalculateCorrectCosts()
    {
        // Arrange
        var uniCluster = new KMeans1D();
        double[] values = [1, 2, 3, 4, 5];
        int numberOfClusters = 1;

        uniCluster.ComputePrefixSums(values);

        // Act
        uniCluster.ComputeClusterCosts(values, numberOfClusters);

        // Assert
        // For k=1, DpTable[i, 1] should equal ComputeCost(1, i)
        Assert.Equal(0, uniCluster.DpTable[1, 1]); // Cost of clustering 1 point into 1 cluster is 0

        // For 2 points [1, 2], mean = 1.5, cost = (1-1.5)² + (2-1.5)² = 0.25 + 0.25 = 0.5
        Assert.Equal(0.5, uniCluster.DpTable[2, 1], 3);

        // For all 5 points [1,2,3,4,5], mean = 3, cost = 4+1+0+1+4 = 10
        Assert.Equal(10.0, uniCluster.DpTable[5, 1], 3);
    }

    [Fact]
    public void ComputeClusterCosts_WithTwoClusters_ShouldFindOptimalSplit()
    {
        // Arrange
        var uniCluster = new KMeans1D();
        double[] values = [1, 2, 8, 9]; // Clear separation between [1,2] and [8,9]
        int numberOfClusters = 2;

        uniCluster.ComputePrefixSums(values);

        // Act
        uniCluster.ComputeClusterCosts(values, numberOfClusters);

        // Assert
        // DpTable[4, 2] should be the cost of optimal 2-cluster solution
        // Optimal split should be [1,2] and [8,9]
        // Cost for [1,2]: mean=1.5, cost = 0.5
        // Cost for [8,9]: mean=8.5, cost = 0.5
        // Total optimal cost = 1.0
        Assert.Equal(1.0, uniCluster.DpTable[4, 2], 3);

        // Verify that 2-cluster solution is better than 1-cluster
        Assert.True(uniCluster.DpTable[4, 2] < uniCluster.DpTable[4, 1]);
    }

    [Fact]
    public void ComputeClusterCosts_WithThreeClusters_ShouldCalculateCorrectDP()
    {
        // Arrange
        var uniCluster = new KMeans1D();
        double[] values = [1, 5, 10]; // Each point can be its own cluster
        int numberOfClusters = 3;

        uniCluster.ComputePrefixSums(values);

        // Act
        uniCluster.ComputeClusterCosts(values, numberOfClusters);

        // Assert
        // With 3 clusters for 3 points, optimal cost should be 0 (each point in its own cluster)
        Assert.Equal(0.0, uniCluster.DpTable[3, 3], 3);

        // Verify progression: 3 clusters should be better than 2 clusters
        Assert.True(uniCluster.DpTable[3, 3] <= uniCluster.DpTable[3, 2]);
    }

    [Fact]
    public void ComputeClusterCosts_WithIdenticalValues_ShouldHandleCorrectly()
    {
        // Arrange
        var uniCluster = new KMeans1D();
        double[] values = [5, 5, 5, 5]; // All identical values
        int numberOfClusters = 2;

        uniCluster.ComputePrefixSums(values);

        // Act
        uniCluster.ComputeClusterCosts(values, numberOfClusters);

        // Assert
        // Cost should be 0 for any number of clusters since all values are identical
        Assert.Equal(0.0, uniCluster.DpTable[4, 1], 3);
        Assert.Equal(0.0, uniCluster.DpTable[4, 2], 3);
    }

    [Fact]
    public void ComputeClusterCosts_TableInitialization_ShouldSetCorrectDimensions()
    {
        // Arrange
        var uniCluster = new KMeans1D();
        double[] values = [1, 2, 3];
        int numberOfClusters = 2;

        uniCluster.ComputePrefixSums(values);

        // Act
        uniCluster.ComputeClusterCosts(values, numberOfClusters);

        // Assert
        // Table should be (values.Length + 1) x (numberOfClusters + 1)
        Assert.Equal(4, uniCluster.DpTable.GetLength(0)); // length dimension
        Assert.Equal(3, uniCluster.DpTable.GetLength(1)); // width dimension
    }

    [Fact]
    public void ComputeClusterCosts_TableInitialization_ShouldSetInfinityForInvalidStates()
    {
        // Arrange
        var uniCluster = new KMeans1D();
        double[] values = [1, 2, 3];
        int numberOfClusters = 2;

        uniCluster.ComputePrefixSums(values);

        // Act
        uniCluster.ComputeClusterCosts(values, numberOfClusters);

        // Assert
        // Invalid states (i < k) should remain as positive infinity
        Assert.Equal(double.PositiveInfinity, uniCluster.DpTable[1, 2]); // 1 point, 2 clusters
        Assert.Equal(double.PositiveInfinity, uniCluster.DpTable[0, 1]); // 0 points, 1 cluster
        Assert.Equal(double.PositiveInfinity, uniCluster.DpTable[0, 2]); // 0 points, 2 clusters
    }

    [Fact]
    public void ComputeClusterCosts_WithLargerDataset_ShouldMaintainOptimality()
    {
        // Arrange
        var uniCluster = new KMeans1D();
        double[] values = [1, 2, 3, 10, 11, 12]; // Two clear clusters
        int numberOfClusters = 2;

        uniCluster.ComputePrefixSums(values);

        // Act
        uniCluster.ComputeClusterCosts(values, numberOfClusters);

        // Assert
        // The optimal 2-cluster solution should split at [1,2,3] and [10,11,12]
        // Cost for [1,2,3]: mean=2, cost = 1+0+1 = 2
        // Cost for [10,11,12]: mean=11, cost = 1+0+1 = 2
        // Total optimal cost = 4
        Assert.Equal(4.0, uniCluster.DpTable[6, 2], 3);

        // Verify monotonicity: more clusters should not increase cost
        Assert.True(uniCluster.DpTable[6, 2] <= uniCluster.DpTable[6, 1]);
    }

    [Fact]
    public void ComputeClusterCosts_EdgeCase_SinglePoint_ShouldHandleCorrectly()
    {
        // Arrange
        var uniCluster = new KMeans1D();
        double[] values = [42]; // Single point
        int numberOfClusters = 1;

        uniCluster.ComputePrefixSums(values);

        // Act
        uniCluster.ComputeClusterCosts(values, numberOfClusters);

        // Assert
        // Cost of clustering a single point should be 0
        Assert.Equal(0.0, uniCluster.DpTable[1, 1], 3);
    }

    [Fact]
    public void ComputeClusterCosts_ValidatesKEqualsNumberOfPoints_ShouldGiveZeroCost()
    {
        // Arrange
        var uniCluster = new KMeans1D();
        double[] values = [1, 5, 10, 15];
        int numberOfClusters = 4; // Same as number of points

        uniCluster.ComputePrefixSums(values);

        // Act
        uniCluster.ComputeClusterCosts(values, numberOfClusters);

        // Assert
        // When k equals number of points, each point is its own cluster, cost = 0
        Assert.Equal(0.0, uniCluster.DpTable[4, 4], 3);
    }

    [Fact]
    public void ComputeClusterCosts_DynamicProgrammingProperty_ShouldSatisfyOptimalSubstructure()
    {
        // Arrange
        var uniCluster = new KMeans1D();
        double[] values = [1, 3, 7, 9];
        int numberOfClusters = 3;

        uniCluster.ComputePrefixSums(values);

        // Act
        uniCluster.ComputeClusterCosts(values, numberOfClusters);

        // Assert
        // Verify that DP[i, k] satisfies the recurrence relation
        // DP[4, 3] should be minimum of:
        // DP[2, 2] + cost(3, 4) = optimal cost of first 2 points in 2 clusters + cost of points 3,4 in 1 cluster
        // DP[3, 2] + cost(4, 4) = optimal cost of first 3 points in 2 clusters + cost of point 4 in 1 cluster

        var dp_4_3 = uniCluster.DpTable[4, 3];

        // Calculate what the cost should be based on the recurrence
        var option1 = uniCluster.DpTable[2, 2] + uniCluster.ComputeCost(3, 4);
        var option2 = uniCluster.DpTable[3, 2] + uniCluster.ComputeCost(4, 4);
        var expectedMinimum = Math.Min(option1, option2);

        Assert.Equal(expectedMinimum, dp_4_3, 3);
    }

    [Fact]
    public void ComputeClusterCosts_WithNegativeValues_ShouldHandleCorrectly()
    {
        var uniCluster = new KMeans1D();
        double[] values = [-5, -2, 0, 3, 6]; // Mix of negative and positive values
        int numberOfClusters = 2;

        uniCluster.ComputePrefixSums(values);

        uniCluster.ComputeClusterCosts(values, numberOfClusters);

        Assert.True(uniCluster.DpTable[5, 2] >= 0); // Cost should be non-negative
        Assert.True(uniCluster.DpTable[5, 2] <= uniCluster.DpTable[5, 1]); // More clusters shouldn't increase cost
        Assert.True(double.IsFinite(uniCluster.DpTable[5, 2])); // Should be a finite number
    }

    [Fact]
    public void ComputeClusterCosts_MonotonicityProperty_MoreClustersShouldNotIncreaseCost()
    {
        var uniCluster = new KMeans1D();
        double[] values = [1, 4, 7, 10, 13];
        int maxClusters = 4;

        uniCluster.ComputePrefixSums(values);

        uniCluster.ComputeClusterCosts(values, maxClusters);

        for (int k = 1; k < maxClusters; k++)
        {
            Assert.True(uniCluster.DpTable[5, k + 1] <= uniCluster.DpTable[5, k],
                $"Cost with {k + 1} clusters should be <= cost with {k} clusters");
        }
    }

    [Fact]
    public void ComputeDpTable_ToyExample_K1toK3_MatchesExpectedCells()
    {
        // Data and expected DP (rows = i points 1..6, cols = k clusters 1..3)
        // k=1: [0.0, 0.5, 2.0, 50.0, 89.2, 125.5]
        // k=2: [∞,   0.0, 0.5,  2.0,   2.5,   4.0]
        // k=3: [∞,   ∞,   0.0,  0.5,   1.0,   2.5]
        var values = new double[] { 1, 2, 3, 10, 11, 12 };

        var kmeans = new KMeans1D();
        kmeans.Fit(values, numberOfClusters: 3, preSortedArray: true);

        var dp = kmeans.DpTable;

        // k = 1
        AssertAlmost(dp[1, 1], 0.0);
        AssertAlmost(dp[2, 1], 0.5);
        AssertAlmost(dp[3, 1], 2.0);
        AssertAlmost(dp[4, 1], 50.0);
        AssertAlmost(dp[5, 1], 89.2);
        AssertAlmost(dp[6, 1], 125.5);

        // k = 2
        Assert.True(double.IsPositiveInfinity(dp[1, 2]));
        AssertAlmost(dp[2, 2], 0.0);
        AssertAlmost(dp[3, 2], 0.5);
        AssertAlmost(dp[4, 2], 2.0);
        AssertAlmost(dp[5, 2], 2.5);
        AssertAlmost(dp[6, 2], 4.0);

        // k = 3
        Assert.True(double.IsPositiveInfinity(dp[1, 3]));
        Assert.True(double.IsPositiveInfinity(dp[2, 3]));
        AssertAlmost(dp[3, 3], 0.0);
        AssertAlmost(dp[4, 3], 0.5);
        AssertAlmost(dp[5, 3], 1.0);
        AssertAlmost(dp[6, 3], 2.5);
        
        static void AssertAlmost(double actual, double expected, double tol = 1e-9)
            => Assert.True(Math.Abs(actual - expected) <= tol, $"Expected {expected}, got {actual}");
    }

    [Fact]
    public void Fit_WithValidInput_ShouldReturnCorrectNumberOfClusters()
    {
        // Arrange
        var kmeans = new KMeans1D();
        double[] values = [1, 2, 3, 8, 9, 10];
        int numberOfClusters = 2;

        // Act
        var result = kmeans.Fit(values, numberOfClusters);

        // Assert
        Assert.Equal(numberOfClusters, result.Clusters.Count);
        Assert.True(result.TotalCost >= 0);
    }

    [Fact]
    public void Fit_WithSingleCluster_ShouldReturnAllPointsInOneCluster()
    {
        // Arrange
        var kmeans = new KMeans1D();
        double[] values = [5, 2, 8, 1, 9];
        int numberOfClusters = 1;

        // Act
        var result = kmeans.Fit(values, numberOfClusters);

        // Assert
        Assert.Single(result.Clusters);
        Assert.Equal(values.Length, result.Clusters[0].Points.Count);
        Assert.Equal(values.Average(), result.Clusters[0].Centroid, 10);
    }

    [Fact]
    public void Fit_WithNumberOfClustersEqualToDataSize_ShouldCreateSingletonClusters()
    {
        // Arrange
        var kmeans = new KMeans1D();
        double[] values = [1, 5, 10, 15];
        int numberOfClusters = values.Length;

        // Act
        var result = kmeans.Fit(values, numberOfClusters);

        // Assert
        Assert.Equal(numberOfClusters, result.Clusters.Count);
        Assert.Equal(0.0, result.TotalCost, 10); // Each point in its own cluster = 0 cost
        
        // Each cluster should have exactly one point
        foreach (var cluster in result.Clusters)
        {
            Assert.Single(cluster.Points);
            Assert.Equal(cluster.Points[0], cluster.Centroid);
        }
    }

    [Fact]
    public void Fit_WithTooManyClusters_ShouldThrowArgumentException()
    {
        // Arrange
        var kmeans = new KMeans1D();
        double[] values = [1, 2, 3];
        int numberOfClusters = 5; // More clusters than data points

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() => kmeans.Fit(values, numberOfClusters));
        Assert.Contains("Number of clusters must be less than or equal to the number of values", exception.Message);
    }

    [Fact]
    public void Fit_WithUnsortedData_ShouldSortAndClusterCorrectly()
    {
        // Arrange
        var kmeans = new KMeans1D();
        double[] values = [10, 1, 11, 2, 12, 3]; // Unsorted with clear separation
        int numberOfClusters = 2;

        // Act
        var result = kmeans.Fit(values, numberOfClusters, preSortedArray: false);

        // Assert
        Assert.Equal(2, result.Clusters.Count);
        
        // Should split into [1,2,3] and [10,11,12]
        var sortedClusters = result.Clusters.OrderBy(c => c.Centroid).ToList();
        
        Assert.Equal(3, sortedClusters[0].Points.Count);
        Assert.Equal(3, sortedClusters[1].Points.Count);
        Assert.True(sortedClusters[0].Centroid < sortedClusters[1].Centroid);
    }

    [Fact]
    public void Fit_WithPreSortedFlag_ShouldNotResortData()
    {
        // Arrange
        var kmeans = new KMeans1D();
        double[] values = [1, 2, 3, 10, 11, 12]; // Already sorted
        int numberOfClusters = 2;

        // Act
        var result = kmeans.Fit(values, numberOfClusters, preSortedArray: true);

        // Assert
        Assert.Equal(2, result.Clusters.Count);
        Assert.Equal(4.0, result.TotalCost, 3); // Expected cost for optimal split
    }

    [Fact]
    public void Fit_WithIdenticalValues_ShouldHandleCorrectly()
    {
        // Arrange
        var kmeans = new KMeans1D();
        double[] values = [5, 5, 5, 5, 5];
        int numberOfClusters = 3;

        // Act
        var result = kmeans.Fit(values, numberOfClusters);

        // Assert
        Assert.Equal(numberOfClusters, result.Clusters.Count);
        Assert.Equal(0.0, result.TotalCost, 10); // No variance = 0 cost
        
        // All centroids should be the same value
        foreach (var cluster in result.Clusters)
        {
            Assert.Equal(5.0, cluster.Centroid, 10);
        }
    }

    [Fact]
    public void Fit_WithNegativeValues_ShouldClusterCorrectly()
    {
        // Arrange
        var kmeans = new KMeans1D();
        double[] values = [-10, -5, 0, 5, 10];
        int numberOfClusters = 2;

        // Act
        var result = kmeans.Fit(values, numberOfClusters);

        // Assert
        Assert.Equal(2, result.Clusters.Count);
        Assert.True(result.TotalCost >= 0);
        
        // Verify all points are assigned
        var totalPoints = result.Clusters.Sum(c => c.Points.Count);
        Assert.Equal(values.Length, totalPoints);
    }

    [Fact]
    public void Fit_WithSinglePoint_ShouldCreateSingleCluster()
    {
        // Arrange
        var kmeans = new KMeans1D();
        double[] values = [42];
        int numberOfClusters = 1;

        // Act
        var result = kmeans.Fit(values, numberOfClusters);

        // Assert
        Assert.Single(result.Clusters);
        Assert.Single(result.Clusters[0].Points);
        Assert.Equal(42, result.Clusters[0].Points[0]);
        Assert.Equal(42, result.Clusters[0].Centroid);
        Assert.Equal(0.0, result.TotalCost);
    }

    [Fact]
    public void Fit_WithLargeGaps_ShouldFindOptimalSeparation()
    {
        // Arrange
        var kmeans = new KMeans1D();
        double[] values = [1, 2, 100, 101]; // Large gap between 2 and 100
        int numberOfClusters = 2;

        // Act
        var result = kmeans.Fit(values, numberOfClusters);

        // Assert
        Assert.Equal(2, result.Clusters.Count);
        
        var sortedClusters = result.Clusters.OrderBy(c => c.Centroid).ToList();
        
        // First cluster should contain [1, 2]
        Assert.Equal(2, sortedClusters[0].Points.Count);
        Assert.Contains(1.0, sortedClusters[0].Points);
        Assert.Contains(2.0, sortedClusters[0].Points);
        
        // Second cluster should contain [100, 101]
        Assert.Equal(2, sortedClusters[1].Points.Count);
        Assert.Contains(100.0, sortedClusters[1].Points);
        Assert.Contains(101.0, sortedClusters[1].Points);
    }

    [Fact]
    public void Fit_ResultClusters_ShouldHaveCorrectCentroids()
    {
        // Arrange
        var kmeans = new KMeans1D();
        double[] values = [1, 2, 3, 10, 11, 12];
        int numberOfClusters = 2;

        // Act
        var result = kmeans.Fit(values, numberOfClusters);

        // Assert
        foreach (var cluster in result.Clusters)
        {
            var expectedCentroid = cluster.Points.Average();
            Assert.Equal(expectedCentroid, cluster.Centroid, 10);
        }
    }

    [Fact]
    public void Fit_ResultClusters_ShouldContainAllOriginalPoints()
    {
        // Arrange
        var kmeans = new KMeans1D();
        double[] values = [7, 3, 9, 1, 5, 8, 2];
        int numberOfClusters = 3;

        // Act
        var result = kmeans.Fit(values, numberOfClusters);

        // Assert
        var allAssignedPoints = result.Clusters.SelectMany(c => c.Points).ToList();
        Assert.Equal(values.Length, allAssignedPoints.Count);
        
        // Verify each original point appears exactly once
        foreach (var value in values)
        {
            Assert.Contains(value, allAssignedPoints);
        }
    }

    [Fact]
    public void Fit_WithDecimalValues_ShouldHandlePrecisionCorrectly()
    {
        // Arrange
        var kmeans = new KMeans1D();
        double[] values = [1.1, 1.2, 1.3, 5.1, 5.2, 5.3];
        int numberOfClusters = 2;

        // Act
        var result = kmeans.Fit(values, numberOfClusters);

        // Assert
        Assert.Equal(2, result.Clusters.Count);
        
        var sortedClusters = result.Clusters.OrderBy(c => c.Centroid).ToList();
        
        // Verify precision is maintained
        Assert.Equal(1.2, sortedClusters[0].Centroid, 10); // (1.1 + 1.2 + 1.3) / 3
        Assert.Equal(5.2, sortedClusters[1].Centroid, 10); // (5.1 + 5.2 + 5.3) / 3
    }

    [Fact]
    public void Fit_MonotonicityProperty_MoreClustersShouldNotIncreaseCost()
    {
        // Arrange
        var kmeans = new KMeans1D();
        double[] values = [1, 3, 5, 7, 9, 11];

        // Act & Assert
        double previousCost = double.MaxValue;
        
        for (int k = 1; k <= values.Length; k++)
        {
            var result = kmeans.Fit(values, k);
            
            Assert.True(result.TotalCost <= previousCost, 
                $"Cost with {k} clusters ({result.TotalCost}) should be <= cost with {k-1} clusters ({previousCost})");
            
            previousCost = result.TotalCost;
        }
    }

    [Fact]
    public void Fit_DeterministicResults_ShouldProduceSameResultsForSameInput()
    {
        // Arrange
        var kmeans1 = new KMeans1D();
        var kmeans2 = new KMeans1D();
        double[] values = [3, 1, 4, 1, 5, 9, 2, 6];
        int numberOfClusters = 3;

        // Act
        var result1 = kmeans1.Fit(values, numberOfClusters);
        var result2 = kmeans2.Fit(values, numberOfClusters);

        // Assert
        Assert.Equal(result1.TotalCost, result2.TotalCost, 10);
        Assert.Equal(result1.Clusters.Count, result2.Clusters.Count);
        
        // Sort clusters by centroid for comparison
        var clusters1 = result1.Clusters.OrderBy(c => c.Centroid).ToList();
        var clusters2 = result2.Clusters.OrderBy(c => c.Centroid).ToList();
        
        for (int i = 0; i < clusters1.Count; i++)
        {
            Assert.Equal(clusters1[i].Centroid, clusters2[i].Centroid, 10);
            Assert.Equal(clusters1[i].Points.Count, clusters2[i].Points.Count);
        }
    }

    [Fact]
    public void Fit_OptimalSolution_ShouldBeBetterThanSuboptimalSplit()
    {
        // Arrange
        var kmeans = new KMeans1D();
        double[] values = [1, 2, 3, 10, 11, 12]; // Clear optimal split at [1,2,3] | [10,11,12]
        int numberOfClusters = 2;

        // Act
        var result = kmeans.Fit(values, numberOfClusters);

        // Assert
        // Manual calculation of optimal cost:
        // Cluster 1: [1,2,3], mean=2, cost = (1-2)² + (2-2)² + (3-2)² = 1 + 0 + 1 = 2
        // Cluster 2: [10,11,12], mean=11, cost = (10-11)² + (11-11)² + (12-11)² = 1 + 0 + 1 = 2
        // Total optimal cost = 4
        Assert.Equal(4.0, result.TotalCost, 3);
        
        // Verify this is better than any suboptimal split
        // E.g., splitting at [1,2] | [3,10,11,12] would have higher cost
    }

    [Fact]
    public void Fit_EmptyInput_ShouldThrowException()
    {
        // Arrange
        var kmeans = new KMeans1D();
        double[] values = [];
        int numberOfClusters = 1;

        // Act & Assert
        Assert.Throws<ArgumentException>(() => kmeans.Fit(values, numberOfClusters));
    }

    [Fact]
    public void Fit_ZeroClusters_ShouldThrowException()
    {
        // Arrange
        var kmeans = new KMeans1D();
        double[] values = [1, 2, 3];
        int numberOfClusters = 0;

        // Act & Assert
        Assert.Throws<ArgumentException>(() => kmeans.Fit(values, numberOfClusters));
    }

    [Fact]
    public void Fit_NegativeClusters_ShouldThrowException()
    {
        // Arrange
        var kmeans = new KMeans1D();
        double[] values = [1, 2, 3];
        int numberOfClusters = -1;

        // Act & Assert
        Assert.Throws<ArgumentException>(() => kmeans.Fit(values, numberOfClusters));
    }
}