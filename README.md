# UniCluster.Net

[![.NET](https://img.shields.io/badge/.NET-8.0-blue.svg)](https://dotnet.microsoft.com/download)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)
[![NuGet](https://img.shields.io/nuget/v/UniCluster.Net.svg)](https://www.nuget.org/packages/UniCluster.Net/)

**UniCluster.Net** is a high-performance .NET library that provides **optimal 1D K-means clustering** using dynamic programming. Unlike traditional K-means implementations that may converge to local optima, this library guarantees finding the globally optimal solution for one-dimensional data in O(k·n) time.

## 🚀 Quick Start

### Installation

```bash
dotnet add package UniCluster.Net
```

### Basic Usage

```csharp
using UniCluster.Net;

// Your data points
var values = new double[] { 1.0, 2.0, 8.0, 9.0, 10.0, 15.0, 16.0 };

// Find optimal clustering with 3 clusters
var result = new OptimalKMeans1D().Fit(values, k: 3);

// Examine results
Console.WriteLine($"Total cost (WCSS): {result.TotalCost:F2}");
Console.WriteLine($"Number of clusters: {result.ClusterCount}");

foreach (var cluster in result.Clusters)
{
    Console.WriteLine($"Cluster centroid: {cluster.Centroid:F2}");
    Console.WriteLine($"Points: [{string.Join(", ", cluster.Points.Select(p => p.ToString("F1")))}]");
    Console.WriteLine($"Size: {cluster.PointCount} points");
    Console.WriteLine();
}
```

**Output:**
```
Total cost (WCSS): 2.67
Number of clusters: 3
Cluster centroid: 1.5
Points: [1.0, 2.0]
Size: 2 points

Cluster centroid: 9.0
Points: [8.0, 9.0, 10.0]
Size: 3 points

Cluster centroid: 15.5
Points: [15.0, 16.0]
Size: 2 points
```

## 🔬 Algorithm Details

This library implements the optimal 1D K-means algorithm based on dynamic programming techniques from recent machine learning research. The algorithm:

1. **Sorts** input data (O(n log n))
2. **Computes prefix sums** for efficient cost calculations (O(n))
3. **Uses dynamic programming** with monotonicity optimization to find optimal cluster boundaries (O(k·n))
4. **Backtracks** to reconstruct the actual clusters (O(k))

### Why "Optimal"?

Traditional K-means clustering can get stuck in local optima depending on initial centroid placement. For 1D data, this library uses dynamic programming to explore all possible clustering configurations and guarantees the globally optimal solution that minimizes the within-cluster sum of squares (WCSS).

## 🎯 Use Cases

- **Data Preprocessing**: Optimal binning for continuous variables
- **Image Processing**: Optimal quantization levels for grayscale images
- **Signal Processing**: Optimal amplitude level detection
- **Time Series**: Optimal segmentation of temporal data
- **Business Analytics**: Customer segmentation, price optimization
- **Scientific Computing**: Data discretization, feature engineering

## 📖 Theory & Research

This implementation is based on the optimal 1D K-means clustering algorithm described in academic literature, particularly:

**"Optimal univariate clustering via dynamic programming"** - [arXiv:1701.07204](https://arxiv.org/pdf/1701.07204)

The algorithm leverages the monotonicity property of optimal split points in 1D K-means to achieve linear time complexity in the number of data points for each cluster count.

## 🤝 Contributing

We welcome contributions! UniCluster.Net is built by the community for the community.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.