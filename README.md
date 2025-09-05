# UniCluster.Net

High-performance .NET library for optimal 1D K-means clustering using dynamic programming. Guarantees globally optimal solutions for one-dimensional data with deterministic results and predictable O(k·n) time complexity, achieving 28-142x performance improvements over ML.NET K-means.

## Key Features

- **Guaranteed Global Optimum**: Uses dynamic programming to find the mathematically optimal clustering solution
- **Deterministic Results**: Always produces the same result for the same input data
- **High Performance**: Optimized O(k·n) implementation, 28-142x faster than ML.NET K-means
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

## Comparison with ML.NET K-means

| Aspect | UniCluster.Net | ML.NET K-means |
|--------|---------------|-------------------|
| **Solution Quality** | Globally optimal (guaranteed) | Locally optimal (variable) |
| **Performance** | 28-142x faster | Baseline |
| **Consistency** | Deterministic, same every time | Random initialization, varies |
| **Memory Usage** | O(k·n) for DP table | O(n + k) iterative |
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

UniCluster.Net implements the optimal 1D K-means algorithm using dynamic programming with the following innovations:

- **Monotonicity Optimization**: Exploits the monotonicity property of 1D clustering to achieve O(k·n) time complexity
- **Efficient Cost Computation**: Uses prefix sums for O(1) cluster cost calculation
- **Memory-Optimized DP**: Optimized dynamic programming table structure

The algorithm minimizes the within-cluster sum of squares (WCSS):
```
WCSS = Σᵢ Σⱼ (xᵢⱼ - cᵢ)²
```
where xᵢⱼ is the j-th point in cluster i, and cᵢ is the centroid of cluster i.

## Benchmarking

Run the included benchmarks to compare with other implementations:

```bash
cd UniCluster.Net.Benchmarks
dotnet run -c Release
```

The benchmark suite includes:
- Small datasets (30 points)
- Medium datasets (300 points) 
- Large datasets (1000+ points)
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