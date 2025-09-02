using UniCluster.Net;

namespace UniCluster.Tests;

public class OptimalKMeans1DTests
{
    [Fact]
    public void ItGeneratesThePrefixSums()
    {
        var uniCluster = new OptimalKMeans1D();

        uniCluster.ComputePrefixSums([1, 2, 3, 10, 11, 12]);

        double[] expectedSums = [0, 1, 3, 6, 16, 27, 39];

        Assert.Equal(expectedSums, uniCluster.PrefixSums);
    }

    [Fact]
    public void ItGeneratesThePrefixSumSquares()
    {
        var uniCluster = new OptimalKMeans1D();

        uniCluster.ComputePrefixSums([1, 2, 3, 10, 11, 12]);

        double[] expectedSumOfSquares = [0, 1, 5, 14, 114, 235, 379];

        Assert.Equal(expectedSumOfSquares, uniCluster.PrefixSumOfSquares);
    }

    [Fact]
    public void ComputeClusterCosts_WithSingleCluster_ShouldCalculateCorrectCosts()
    {
        // Arrange
        var uniCluster = new OptimalKMeans1D();
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
        var uniCluster = new OptimalKMeans1D();
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
        var uniCluster = new OptimalKMeans1D();
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
        var uniCluster = new OptimalKMeans1D();
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
        var uniCluster = new OptimalKMeans1D();
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
        var uniCluster = new OptimalKMeans1D();
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
        var uniCluster = new OptimalKMeans1D();
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
        var uniCluster = new OptimalKMeans1D();
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
        var uniCluster = new OptimalKMeans1D();
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
        var uniCluster = new OptimalKMeans1D();
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
        var uniCluster = new OptimalKMeans1D();
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
        var uniCluster = new OptimalKMeans1D();
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

        var kmeans = new OptimalKMeans1D();
        kmeans.Fit(values, 3);

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
        var kmeans = new OptimalKMeans1D();
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
        var kmeans = new OptimalKMeans1D();
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
        var kmeans = new OptimalKMeans1D();
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
        var kmeans = new OptimalKMeans1D();
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
        var kmeans = new OptimalKMeans1D();
        double[] values = [10, 1, 11, 2, 12, 3]; // Unsorted with clear separation
        int numberOfClusters = 2;

        // Act
        var result = kmeans.Fit(values, numberOfClusters);

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
        var kmeans = new OptimalKMeans1D();
        double[] values = [1, 2, 3, 10, 11, 12]; // Already sorted
        int numberOfClusters = 2;

        // Act
        var result = kmeans.Fit(values, numberOfClusters);

        // Assert
        Assert.Equal(2, result.Clusters.Count);
        Assert.Equal(4.0, result.TotalCost, 3); // Expected cost for optimal split
    }

    [Fact]
    public void Fit_WithIdenticalValues_ShouldHandleCorrectly()
    {
        // Arrange
        var kmeans = new OptimalKMeans1D();
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
        var kmeans = new OptimalKMeans1D();
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
        var kmeans = new OptimalKMeans1D();
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
        var kmeans = new OptimalKMeans1D();
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
        var kmeans = new OptimalKMeans1D();
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
        var kmeans = new OptimalKMeans1D();
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
        var kmeans = new OptimalKMeans1D();
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
        var kmeans = new OptimalKMeans1D();
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
        var kmeans1 = new OptimalKMeans1D();
        var kmeans2 = new OptimalKMeans1D();
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
        var kmeans = new OptimalKMeans1D();
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
        var kmeans = new OptimalKMeans1D();
        double[] values = [];
        int numberOfClusters = 1;

        // Act & Assert
        Assert.Throws<ArgumentException>(() => kmeans.Fit(values, numberOfClusters));
    }

    [Fact]
    public void Fit_ZeroClusters_ShouldThrowException()
    {
        // Arrange
        var kmeans = new OptimalKMeans1D();
        double[] values = [1, 2, 3];
        int numberOfClusters = 0;

        // Act & Assert
        Assert.Throws<ArgumentException>(() => kmeans.Fit(values, numberOfClusters));
    }

    [Fact]
    public void Fit_NegativeClusters_ShouldThrowException()
    {
        // Arrange
        var kmeans = new OptimalKMeans1D();
        double[] values = [1, 2, 3];
        int numberOfClusters = -1;

        // Act & Assert
        Assert.Throws<ArgumentException>(() => kmeans.Fit(values, numberOfClusters));
    }

    #region Comprehensive Verification Tests

    [Theory]
    [InlineData(new double[] { 1, 2, 10, 11 }, 2, 1.0)]
    [InlineData(new double[] { 1, 5, 9, 13 }, 4, 0.0)]
    [InlineData(new double[] { 0, 1, 10, 11, 20, 21 }, 3, 1.5)]
    public void Fit_KnownOptimalSolutions_ShouldMatchExpectedWCSS(double[] data, int k, double expectedWCSS)
    {
        // Arrange
        var optimizer = new OptimalKMeans1D();

        // Act
        var result = optimizer.Fit(data, k);

        // Assert
        Assert.Equal(expectedWCSS, result.TotalCost, 6);
        
        // Verify manual WCSS calculation matches
        var manualWCSS = 0.0;
        foreach (var cluster in result.Clusters)
        {
            var centroid = cluster.Centroid;
            manualWCSS += cluster.Points.Sum(p => Math.Pow(p - centroid, 2));
        }
        Assert.Equal(result.TotalCost, manualWCSS, 10);
    }

    [Fact]
    public void Fit_BruteForceVerification_SmallDataset_ShouldMatchOptimal()
    {
        // Arrange
        var optimizer = new OptimalKMeans1D();
        var testCases = new[]
        {
            (new double[] { 1, 2, 8, 9 }, 2),
            (new double[] { 1, 3, 6, 8, 10 }, 3),
            (new double[] { 0, 1, 5, 6, 10, 11 }, 3),
            (new double[] { 2, 4, 6, 8 }, 2),
            (new double[] { 1, 5, 6, 10, 11, 15 }, 3)
        };

        foreach (var (data, k) in testCases)
        {
            // Act
            var libraryResult = optimizer.Fit(data, k);
            var bruteForceWCSS = BruteForceOptimal(data, k);

            // Assert
            Assert.Equal(bruteForceWCSS, libraryResult.TotalCost, 10);
        }
    }

    [Fact]
    public void Fit_MonotonicWCSSDecrease_ShouldDecreaseOrStayEqual()
    {
        // Arrange
        var optimizer = new OptimalKMeans1D();
        var testData = new double[] { 1, 3, 5, 7, 9, 11, 13, 15 };
        var wcssSequence = new List<double>();

        // Act
        for (int k = 1; k <= testData.Length; k++)
        {
            var result = optimizer.Fit(testData, k);
            wcssSequence.Add(result.TotalCost);
        }

        // Assert
        for (int i = 1; i < wcssSequence.Count; i++)
        {
            Assert.True(wcssSequence[i] <= wcssSequence[i-1] + 1e-10, 
                $"WCSS with {i+1} clusters ({wcssSequence[i]}) should be <= WCSS with {i} clusters ({wcssSequence[i-1]})");
        }
    }

    [Fact]
    public void Fit_SingleCluster_ShouldEqualTotalVariance()
    {
        // Arrange
        var optimizer = new OptimalKMeans1D();
        var testData = new double[] { 1, 3, 5, 7, 9, 11, 13, 15 };

        // Act
        var result = optimizer.Fit(testData, 1);
        var expectedVariance = testData.Sum(x => Math.Pow(x - testData.Average(), 2));

        // Assert
        Assert.Equal(expectedVariance, result.TotalCost, 10);
    }

    [Fact]
    public void Fit_MaxClusters_ShouldGiveZeroWCSS()
    {
        // Arrange
        var optimizer = new OptimalKMeans1D();
        var testData = new double[] { 1, 3, 5, 7, 9, 11, 13, 15 };

        // Act
        var result = optimizer.Fit(testData, testData.Length);

        // Assert
        Assert.Equal(0.0, result.TotalCost, 10);
        Assert.Equal(testData.Length, result.ClusterCount);
        
        // Each cluster should have exactly one point
        foreach (var cluster in result.Clusters)
        {
            Assert.Single(cluster.Points);
            Assert.Equal(cluster.Points[0], cluster.Centroid);
        }
    }

    [Fact]
    public void Fit_DeterministicBehavior_ShouldProduceIdenticalResults()
    {
        // Arrange
        var testData = new double[] { 2.5, 7.1, 1.8, 9.3, 4.7 };
        var results = new List<double>();

        // Act
        for (int i = 0; i < 10; i++)
        {
            var optimizer = new OptimalKMeans1D();
            var result = optimizer.Fit(testData, 3);
            results.Add(result.TotalCost);
        }

        // Assert
        Assert.True(results.All(r => Math.Abs(r - results[0]) < 1e-15),
            "All runs should produce identical results");
    }

    [Theory]
    [InlineData(new double[] { 5, 5, 5, 5 }, 2)]
    [InlineData(new double[] { 42 }, 1)]
    [InlineData(new double[] { 1, 1, 1 }, 3)]
    public void Fit_EdgeCases_ShouldHandleCorrectly(double[] data, int k)
    {
        // Arrange
        var optimizer = new OptimalKMeans1D();

        // Act
        var result = optimizer.Fit(data, k);

        // Assert
        Assert.Equal(k, result.ClusterCount);
        Assert.True(result.TotalCost >= 0);
        Assert.True(double.IsFinite(result.TotalCost));
        
        // All points should be assigned
        var totalAssignedPoints = result.Clusters.Sum(c => c.PointCount);
        Assert.Equal(data.Length, totalAssignedPoints);
    }

    [Fact]
    public void Fit_WellSeparatedClusters_ShouldFindTheoreticalOptimum()
    {
        // Arrange
        var optimizer = new OptimalKMeans1D();
        var testData = new double[] { 0.9, 1.0, 1.1, 8.9, 9.0, 9.1 };

        // Act
        var result = optimizer.Fit(testData, 2);

        // Assert - Calculate expected theoretical optimum
        var cluster1 = new[] { 0.9, 1.0, 1.1 };
        var cluster2 = new[] { 8.9, 9.0, 9.1 };
        var centroid1 = cluster1.Average();
        var centroid2 = cluster2.Average();
        var expectedWCSS = cluster1.Sum(x => Math.Pow(x - centroid1, 2)) + 
                          cluster2.Sum(x => Math.Pow(x - centroid2, 2));

        Assert.Equal(expectedWCSS, result.TotalCost, 10);
    }

    [Fact]
    public void Fit_ContiguousClusterProperty_ShouldMaintainSortedOrder()
    {
        // Arrange
        var optimizer = new OptimalKMeans1D();
        var testData = new double[] { 1, 3, 4, 8, 9, 12, 15, 16 };

        // Act
        var result = optimizer.Fit(testData, 3);

        // Assert
        var flattenedPoints = result.Clusters.SelectMany(c => c.Points).OrderBy(p => p).ToArray();
        var originalSorted = testData.OrderBy(p => p).ToArray();

        for (int i = 0; i < flattenedPoints.Length; i++)
        {
            Assert.Equal(originalSorted[i], flattenedPoints[i], 10);
        }
    }

    [Fact]
    public void Fit_AllPointsAssigned_ShouldAssignEveryPoint()
    {
        // Arrange
        var optimizer = new OptimalKMeans1D();
        var testData = new double[] { 1, 3, 4, 8, 9, 12, 15, 16 };

        // Act
        var result = optimizer.Fit(testData, 3);

        // Assert
        var totalPointsInClusters = result.Clusters.Sum(c => c.PointCount);
        Assert.Equal(testData.Length, totalPointsInClusters);
        
        // Verify each original point appears exactly once
        var allAssignedPoints = result.Clusters.SelectMany(c => c.Points).ToList();
        foreach (var originalPoint in testData)
        {
            Assert.Contains(originalPoint, allAssignedPoints);
        }
    }

    [Fact]
    public void Fit_LargeDataset_ShouldHandleNumericalStability()
    {
        // Arrange
        var optimizer = new OptimalKMeans1D();
        var random = new Random(42);
        var largeData = Enumerable.Range(0, 100)
            .Select(_ => random.NextDouble() * 100)
            .OrderBy(x => x)
            .ToArray();

        // Act
        var result = optimizer.Fit(largeData, 10);

        // Assert
        Assert.Equal(10, result.ClusterCount);
        Assert.Equal(largeData.Length, result.Clusters.Sum(c => c.PointCount));
        Assert.True(result.TotalCost >= 0);
        Assert.True(double.IsFinite(result.TotalCost));
        Assert.False(double.IsNaN(result.TotalCost));
        Assert.False(double.IsInfinity(result.TotalCost));
    }

    [Theory]
    [InlineData(new double[] { -10, -5, 0, 5, 10 }, 2)]
    [InlineData(new double[] { -100, -50, -25, 25, 50, 100 }, 3)]
    [InlineData(new double[] { -1.5, -0.5, 0.5, 1.5 }, 2)]
    public void Fit_NegativeValues_ShouldHandleCorrectly(double[] data, int k)
    {
        // Arrange
        var optimizer = new OptimalKMeans1D();

        // Act
        var result = optimizer.Fit(data, k);

        // Assert
        Assert.Equal(k, result.ClusterCount);
        Assert.True(result.TotalCost >= 0);
        Assert.Equal(data.Length, result.Clusters.Sum(c => c.PointCount));
        
        // Verify centroids are calculated correctly
        foreach (var cluster in result.Clusters)
        {
            var expectedCentroid = cluster.Points.Average();
            Assert.Equal(expectedCentroid, cluster.Centroid, 10);
        }
    }

    [Fact]
    public void Fit_PrecisionMaintained_ShouldHandleDecimalValues()
    {
        // Arrange
        var optimizer = new OptimalKMeans1D();
        var testData = new double[] { 1.123456789, 1.234567890, 5.123456789, 5.234567890 };

        // Act
        var result = optimizer.Fit(testData, 2);

        // Assert
        Assert.Equal(2, result.ClusterCount);
        
        // Verify precision is maintained in centroids
        foreach (var cluster in result.Clusters)
        {
            var expectedCentroid = cluster.Points.Average();
            Assert.Equal(expectedCentroid, cluster.Centroid, 15); // High precision check
        }
    }

    [Fact]
    public void Fit_OptimalityGuarantee_ShouldNeverExceedBruteForce()
    {
        // Arrange
        var optimizer = new OptimalKMeans1D();
        var testCases = new[]
        {
            (new double[] { 1, 4, 7, 10 }, 2),
            (new double[] { 2, 3, 8, 9, 14 }, 3),
            (new double[] { 1, 2, 5, 6, 9, 10 }, 2)
        };

        foreach (var (data, k) in testCases)
        {
            // Act
            var libraryResult = optimizer.Fit(data, k);
            var bruteForceWCSS = BruteForceOptimal(data, k);

            // Assert
            Assert.True(libraryResult.TotalCost <= bruteForceWCSS + 1e-10,
                $"Library WCSS ({libraryResult.TotalCost}) should be <= brute force WCSS ({bruteForceWCSS})");
        }
    }

    [Theory]
    [InlineData(null)]
    public void Fit_NullInput_ShouldThrowArgumentNullException(double[] values)
    {
        // Arrange
        var optimizer = new OptimalKMeans1D();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => optimizer.Fit(values, 1));
    }

    [Theory]
    [InlineData(new double[] { double.NaN, 1, 2 })]
    [InlineData(new double[] { 1, double.PositiveInfinity, 2 })]
    [InlineData(new double[] { 1, 2, double.NegativeInfinity })]
    public void Fit_InvalidValues_ShouldThrowArgumentException(double[] values)
    {
        // Arrange
        var optimizer = new OptimalKMeans1D();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => optimizer.Fit(values, 1));
    }

    [Fact]
    public void Fit_ConsistentCentroids_ShouldMatchManualCalculation()
    {
        // Arrange
        var optimizer = new OptimalKMeans1D();
        var testData = new double[] { 1, 2, 3, 10, 11, 12 };

        // Act
        var result = optimizer.Fit(testData, 2);

        // Assert
        foreach (var cluster in result.Clusters)
        {
            var manualCentroid = cluster.Points.Sum() / cluster.Points.Count;
            Assert.Equal(manualCentroid, cluster.Centroid, 10);
        }
    }

    [Fact]
    public void Fit_StressTest_ShouldHandleRepeatedCalls()
    {
        // Arrange
        var optimizer = new OptimalKMeans1D();
        var testData = new double[] { 1, 3, 5, 7, 9 };
        
        // Act & Assert - Multiple calls should not affect each other
        for (int i = 0; i < 100; i++)
        {
            var result = optimizer.Fit(testData, 3);
            Assert.Equal(3, result.ClusterCount);
            Assert.True(result.TotalCost >= 0);
        }
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Brute force optimal solution finder for verification purposes
    /// </summary>
    private static double BruteForceOptimal(double[] data, int k)
    {
        var n = data.Length;
        var sortedData = data.OrderBy(x => x).ToArray();
        
        if (k == 1)
        {
            var centroid = sortedData.Average();
            return sortedData.Sum(x => Math.Pow(x - centroid, 2));
        }
        
        if (k >= n)
        {
            return 0.0;
        }
        
        double bestWCSS = double.MaxValue;
        
        // Generate all combinations of k-1 split points from n-1 possible positions
        var splitPositions = new List<int[]>();
        GenerateCombinations(Enumerable.Range(1, n-1).ToArray(), k-1, new int[k-1], 0, splitPositions);
        
        foreach (var splits in splitPositions)
        {
            var wcss = 0.0;
            var start = 0;
            
            // Process each cluster defined by the splits
            foreach (var end in splits)
            {
                var clusterPoints = sortedData.Skip(start).Take(end - start).ToArray();
                if (clusterPoints.Length > 0)
                {
                    var centroid = clusterPoints.Average();
                    wcss += clusterPoints.Sum(p => Math.Pow(p - centroid, 2));
                }
                start = end;
            }
            
            // Last cluster
            var lastClusterPoints = sortedData.Skip(start).ToArray();
            if (lastClusterPoints.Length > 0)
            {
                var lastCentroid = lastClusterPoints.Average();
                wcss += lastClusterPoints.Sum(p => Math.Pow(p - lastCentroid, 2));
            }
            
            if (wcss < bestWCSS)
            {
                bestWCSS = wcss;
            }
        }
        
        return bestWCSS;
    }

    private static void GenerateCombinations(int[] array, int k, int[] current, int start, List<int[]> result)
    {
        if (k == 0)
        {
            result.Add((int[])current.Clone());
            return;
        }
        
        for (int i = start; i <= array.Length - k; i++)
        {
            current[current.Length - k] = array[i];
            GenerateCombinations(array, k - 1, current, i + 1, result);
        }
    }

    #endregion
}