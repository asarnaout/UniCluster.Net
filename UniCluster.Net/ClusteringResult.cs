namespace UniCluster.Net;

public class ClusteringResult(IReadOnlyList<Cluster> clusters, double totalCost)
{
    public IReadOnlyList<Cluster> Clusters { get; } = clusters;

    public double TotalCost { get; } = totalCost;

    public int ClusterCount => Clusters.Count;
}

public class Cluster(IReadOnlyList<double> points, double centroid)
{
    public IReadOnlyList<double> Points { get; } = points;

    public double Centroid { get; } = centroid;

    public int PointCount => Points.Count;
}