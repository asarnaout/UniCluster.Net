# UniCluster.Net

[![NuGet Version](https://img.shields.io/nuget/v/UniCluster.Net.svg)](https://www.nuget.org/packages/UniCluster.Net)
[![NuGet Downloads](https://img.shields.io/nuget/dt/UniCluster.Net.svg)](https://www.nuget.org/packages/UniCluster.Net)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](License.md)
[![.NET](https://img.shields.io/badge/.NET-8.0-512BD4?logo=dotnet&logoColor=white)](https://dotnet.microsoft.com/en-us/download/dotnet/8.0)

High-performance .NET library for optimal 1D K-means clustering using dynamic programming. Guarantees globally optimal solutions for one-dimensional data with deterministic results and predictable O(k·n) time complexity, achieving 39-951x performance improvements over ML.NET K-means.

## Key Features

- **Guaranteed Global Optimum**: Uses dynamic programming to find the mathematically optimal clustering solution
- **Deterministic Results**: Always produces the same result for the same input data
- **High Performance**: Optimized O(k·n) implementation, 39-951x faster than ML.NET K-means
- **Simple API**: Easy-to-use interface for 1D clustering scenarios
- **Well-Tested**: Comprehensive test suite with extensive validation

## Installation

```bash
dotnet add package UniCluster.Net
```

## Quick Start

```csharp
using UniCluster.Net;

var kmeans = new OptimalKMeans1D();
var data = new double[] { 1.0, 2.0, 8.0, 9.0, 10.0 };
var result = kmeans.Fit(data, k: 2);

Console.WriteLine($"Total Cost (WCSS): {result.TotalCost:F2}");
Console.WriteLine($"Number of Clusters: {result.ClusterCount}");

foreach (var cluster in result.Clusters)
{
    Console.WriteLine($"Cluster: [{string.Join(", ", cluster.Points)}]");
    Console.WriteLine($"Centroid: {cluster.Centroid:F2}");
}
```

### Expected Output

```
Total Cost (WCSS): 2.50
Number of Clusters: 2

Cluster 1: [1, 2]
Centroid: 1.50
  
Cluster 2: [8, 9, 10]  
Centroid: 9.00
```

The algorithm automatically identifies the optimal clustering that minimizes within-cluster sum of squares (WCSS), grouping similar values together while maximizing separation between clusters.

## Comparison with ML.NET K-means

| Aspect | UniCluster.Net | ML.NET K-means |
|--------|---------------|-------------------|
| **Solution Quality** | Globally optimal (guaranteed) | Locally optimal (variable) |
| **Performance** | 39-951x faster | Baseline |
| **Consistency** | Deterministic, same every time | Random initialization, varies |
| **Space Complexity** | O(k·n) | O(n + k) iterative |
| **Time Complexity** | O(k·n) guaranteed | O(i·k·n) where i = iterations |
| **Dimensions** | 1D only | Multi-dimensional |


## When to Use UniCluster.Net

### ✅ Ideal Use Cases
- **1D data clustering** where global optimum is required
- **Performance-critical applications** needing fast, consistent results
- **Reproducible analysis** requiring deterministic outcomes
- **Small to medium k values** (typically k ≤ 20)
- **Applications requiring mathematical guarantees**

### ⚠️ Consider Alternatives When
- Working with high-dimensional data (use traditional K-means)
- Need very large k values (memory usage scales with k·n)
- Data preprocessing/feature engineering is the bottleneck
- Working with streaming or constantly changing data

## Algorithm Details

The algorithm minimizes the within-cluster sum of squares (WCSS):
```
WCSS = Σᵢ Σⱼ (xᵢⱼ - cᵢ)²
```
where xᵢⱼ is the j-th point in cluster i, and cᵢ is the centroid of cluster i.

### How it works (optimal 1D K‑means via Dynamic Programming):

Goal: split sorted data into k contiguous groups to minimize total squared error.

1) Precompute prefix sums so that looking up the cost of clustering any segment [a..b] into one cluster is an O(1) operation.
2) Fill the first column: DP(i,1) = cost(1..i).
3) For k = 2..K and i = k..n, choose a split j < i that minimizes:
   DP(j, k−1) + cost(j+1..i). This is the (previously computed) cost of clustering j elements in k-1 clusters, in addition to clustering the rest of the set (j+1..i) into 1 more cluster cluster.
4) Record j to later backtrack and reconstruct the clusters.

Example x = [1,2,3,10,11,12], k = 2:

For i = 6, we try j = 1..5. The best is j = 3:
- Left: DP(3,1) = cost(1..3) = 2.0
- Right: cost(4..6) = 2.0
- Total = 4.0 → clusters [1,2,3] and [10,11,12]

Note: You don’t need to compute any of this yourself—`Fit(data, k)` does it deterministically.

### How the Monotonicity property allows this to run in O(k·n) time rather than O(k·n²):

- Naive DP evaluates, for each state (i, k), all split positions j ∈ [k−1, i−1], which is O(k·n²) overall.
- However in 1D Clustering, the optimal split index j* that minimizes DP(j, k−1) + cost(j+1..i) is non-decreasing as i grows. 
- Therefore, in UniCluster.Net we carry j* forward and only advance it when it reduces cost. Each j is visited at most once per k, making the total time complexity of constructing the DP table O(k·n).

## Benchmarking

### Performance Results

Benchmarks conducted on Windows 11, 13th Gen Intel Core i7-13620H @ 2.40GHz, .NET 8.0:

| Dataset | UniCluster.Net | ML.NET Single Run | ML.NET Best-of-N | Improvement (Single) |
|---------|----------------|-------------------|------------------|---------------------|
| **Small** (30 points, k=3) | 840 ns | 798,276 ns | 8,309,976 ns (10 runs) | **951x faster** |
| **Medium** (300 points, k=5) | 11,840 ns | 1,169,301 ns | 11,197,798 ns (10 runs) | **99x faster** |
| **Large** (1000 points, k=8) | 61,293 ns | 2,406,017 ns | 11,902,864 ns (5 runs) | **39x faster** |

### Run Benchmarks Yourself

```bash
cd UniCluster.Net.Benchmarks
dotnet run -c Release
```

The benchmark suite includes:
- Small datasets (30 points, k=3)
- Medium datasets (300 points, k=5) 
- Large datasets (1000 points, k=8)
- Comparison with ML.NET single-run and best-of-N approaches

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [License.md](License.md) file for details.

## Acknowledgments

- Algorithm based on the research by Grønlund et al. in "Fast Exact k-Means, k-Medians and Bregman Divergence Clustering in 1D" ([arXiv:1701.07204](https://arxiv.org/abs/1701.07204), 2017)
- Benchmarked against Microsoft ML.NET for validation
- Inspired by the need for deterministic, high-performance clustering