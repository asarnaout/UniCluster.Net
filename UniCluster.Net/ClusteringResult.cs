namespace UniCluster.Net;

/// <summary>
/// Represents the result of a 1D K-means clustering operation, containing the clusters
/// and the total clustering cost.
/// </summary>
/// <param name="clusters">
/// A read-only collection of clusters produced by the clustering algorithm.
/// Each cluster contains the data points assigned to it and its centroid.
/// </param>
/// <param name="totalCost">
/// The total within-cluster sum of squares (WCSS) for the clustering solution.
/// This represents the sum of squared distances from each point to its cluster centroid.
/// Lower values indicate better clustering with points closer to their centroids.
/// </param>
/// <remarks>
/// <para>
/// The <see cref="ClusteringResult"/> is immutable and represents a complete clustering solution.
/// The total cost is calculated as the sum of squared distances from each data point to its
/// assigned cluster centroid, which is the objective function minimized by the K-means algorithm.
/// </para>
/// <para>
/// For optimal 1D K-means, this result represents the globally optimal solution that minimizes
/// the total cost among all possible ways to partition the data into k clusters.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var kmeans = new OptimalKMeans1D();
/// var values = new double[] { 1.0, 2.0, 8.0, 9.0 };
/// var result = kmeans.Fit(values, k: 2);
/// 
/// Console.WriteLine($"Number of clusters: {result.ClusterCount}");
/// Console.WriteLine($"Total cost: {result.TotalCost:F2}");
/// 
/// foreach (var cluster in result.Clusters)
/// {
///     Console.WriteLine($"Cluster with {cluster.PointCount} points, centroid: {cluster.Centroid:F2}");
/// }
/// </code>
/// </example>
public class ClusteringResult(IReadOnlyList<Cluster> clusters, double totalCost)
{
    /// <summary>
    /// Gets the collection of clusters produced by the clustering algorithm.
    /// Each cluster contains the data points assigned to it and its centroid.
    /// </summary>
    /// <value>
    /// A read-only list of <see cref="Cluster"/> objects representing the clustering solution.
    /// The clusters are ordered based on the clustering algorithm's internal processing.
    /// </value>
    public IReadOnlyList<Cluster> Clusters { get; } = clusters;

    /// <summary>
    /// Gets the total within-cluster sum of squares (WCSS) for the clustering solution.
    /// </summary>
    /// <value>
    /// A double representing the sum of squared distances from each point to its cluster centroid.
    /// Lower values indicate better clustering quality with points closer to their centroids.
    /// This value is always non-negative.
    /// </value>
    /// <remarks>
    /// The total cost is calculated as: Σᵢ Σⱼ (xᵢⱼ - cᵢ)²
    /// where xᵢⱼ is the j-th point in cluster i, and cᵢ is the centroid of cluster i.
    /// </remarks>
    public double TotalCost { get; } = totalCost;

    /// <summary>
    /// Gets the number of clusters in the clustering solution.
    /// </summary>
    /// <value>
    /// An integer representing the count of clusters. This value equals the k parameter
    /// that was specified when creating the clustering solution.
    /// </value>
    public int ClusterCount => Clusters.Count;
}

/// <summary>
/// Represents a single cluster in a 1D K-means clustering solution, containing
/// the data points assigned to the cluster and the cluster's centroid.
/// </summary>
/// <param name="points">
/// A read-only collection of data points (numerical values) assigned to this cluster.
/// For 1D K-means, these points will be contiguous when the original data is sorted.
/// </param>
/// <param name="centroid">
/// The centroid (center point) of the cluster, calculated as the arithmetic mean
/// of all points in the cluster.
/// </param>
/// <remarks>
/// <para>
/// A <see cref="Cluster"/> is immutable and represents a subset of the original data
/// that has been grouped together based on similarity. In 1D K-means clustering,
/// the centroid minimizes the sum of squared distances to all points in the cluster.
/// </para>
/// <para>
/// For optimal 1D K-means, clusters will always contain contiguous ranges of values
/// when the original data is sorted, meaning there are no "gaps" within a cluster
/// where points from other clusters are interspersed.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Assuming we have a cluster from a clustering result
/// var cluster = result.Clusters[0];
/// 
/// Console.WriteLine($"Cluster centroid: {cluster.Centroid:F2}");
/// Console.WriteLine($"Number of points: {cluster.PointCount}");
/// Console.WriteLine("Points in cluster:");
/// foreach (var point in cluster.Points)
/// {
///     Console.WriteLine($"  {point:F2}");
/// }
/// 
/// // Calculate cluster's contribution to total cost
/// var clusterCost = cluster.Points.Sum(p => Math.Pow(p - cluster.Centroid, 2));
/// Console.WriteLine($"Cluster cost: {clusterCost:F2}");
/// </code>
/// </example>
public class Cluster(IReadOnlyList<double> points, double centroid)
{
    /// <summary>
    /// Gets the collection of data points assigned to this cluster.
    /// </summary>
    /// <value>
    /// A read-only list of double values representing the numerical data points
    /// that belong to this cluster. For 1D K-means, these points represent
    /// contiguous values when the original dataset is sorted.
    /// </value>
    /// <remarks>
    /// The points in this collection are the original data values that were
    /// determined to be most similar to each other and assigned to the same cluster.
    /// The order of points in this collection may not reflect their original order
    /// in the input data.
    /// </remarks>
    public IReadOnlyList<double> Points { get; } = points;

    /// <summary>
    /// Gets the centroid (arithmetic mean) of all points in this cluster.
    /// </summary>
    /// <value>
    /// A double representing the center point of the cluster, calculated as
    /// the arithmetic mean of all data points in the cluster.
    /// </value>
    /// <remarks>
    /// <para>
    /// The centroid is the point that minimizes the sum of squared distances
    /// to all points in the cluster. It is calculated as:
    /// centroid = (Σᵢ xᵢ) / n
    /// where xᵢ are the points in the cluster and n is the number of points.
    /// </para>
    /// <para>
    /// In K-means clustering, the centroid serves as the representative point
    /// for the entire cluster and is used to assign new points to clusters.
    /// </para>
    /// </remarks>
    public double Centroid { get; } = centroid;

    /// <summary>
    /// Gets the number of data points in this cluster.
    /// </summary>
    /// <value>
    /// An integer representing the count of points assigned to this cluster.
    /// This value is always positive for valid clusters.
    /// </value>
    /// <remarks>
    /// This property provides a convenient way to access the size of the cluster
    /// without iterating through the Points collection.
    /// </remarks>
    public int PointCount => Points.Count;
}