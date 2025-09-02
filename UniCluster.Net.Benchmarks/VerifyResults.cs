using Microsoft.ML;
using Microsoft.ML.Data;
using UniCluster.Net;

namespace UniCluster.Net.Benchmarks;

public class VerifyResults
{
    public static void RunVerification()
    {
        Console.WriteLine("🔍 Verifying Benchmark Results");
        Console.WriteLine("==============================");
        Console.WriteLine();

        // Test case: well-separated clusters where optimal solution should be obvious
        var testData = new double[] 
        {
            // Cluster 1: around 1.0
            0.8, 0.9, 1.0, 1.1, 1.2,
            // Cluster 2: around 5.0  
            4.8, 4.9, 5.0, 5.1, 5.2,
            // Cluster 3: around 9.0
            8.8, 8.9, 9.0, 9.1, 9.2
        };

        var uniCluster = new OptimalKMeans1D();
        var uniResult = uniCluster.Fit(testData, 3);

        // ML.NET comparison
        var mlContext = new MLContext(seed: 42);
        var dataPoints = testData.Select(x => new DataPoint { Value = (float)x }).ToArray();
        var dataView = mlContext.Data.LoadFromEnumerable(dataPoints);

        var pipeline = mlContext.Transforms.Concatenate("Features", nameof(DataPoint.Value))
            .Append(mlContext.Clustering.Trainers.KMeans("Features", numberOfClusters: 3));

        var model = pipeline.Fit(dataView);
        var predictions = model.Transform(dataView);
        var results = mlContext.Data.CreateEnumerable<ClusterPrediction>(predictions, reuseRowObject: false).ToArray();

        // Calculate ML.NET WCSS
        var clusterGroups = new Dictionary<uint, List<double>>();
        for (int i = 0; i < results.Length; i++)
        {
            var clusterId = results[i].ClusterId;
            if (!clusterGroups.ContainsKey(clusterId))
                clusterGroups[clusterId] = new List<double>();
            clusterGroups[clusterId].Add(testData[i]);
        }

        var mlWCSS = 0.0;
        foreach (var group in clusterGroups.Values)
        {
            var centroid = group.Average();
            mlWCSS += group.Sum(point => Math.Pow(point - centroid, 2));
        }

        Console.WriteLine($"Test data: {string.Join(", ", testData.Select(x => x.ToString("F1")))}");
        Console.WriteLine($"Expected clusters: [0.8-1.2], [4.8-5.2], [8.8-9.2]");
        Console.WriteLine();
        Console.WriteLine($"UniCluster.Net WCSS: {uniResult.TotalCost:F6}");
        Console.WriteLine($"ML.NET WCSS:        {mlWCSS:F6}");
        Console.WriteLine();

        Console.WriteLine("UniCluster.Net clusters:");
        foreach (var cluster in uniResult.Clusters)
        {
            Console.WriteLine($"  Centroid: {cluster.Centroid:F2}, Points: [{string.Join(", ", cluster.Points.Select(p => p.ToString("F1")))}]");
        }
        
        Console.WriteLine();
        Console.WriteLine("ML.NET clusters:");
        foreach (var (clusterId, points) in clusterGroups.OrderBy(kvp => kvp.Value.Min()))
        {
            var centroid = points.Average();
            Console.WriteLine($"  Centroid: {centroid:F2}, Points: [{string.Join(", ", points.OrderBy(x => x).Select(p => p.ToString("F1")))}]");
        }

        var isOptimal = uniResult.TotalCost <= mlWCSS + 1e-10;
        Console.WriteLine();
        Console.WriteLine($"✅ UniCluster.Net finds optimal solution: {isOptimal}");
        if (!isOptimal)
        {
            Console.WriteLine($"   ML.NET found better solution by: {uniResult.TotalCost - mlWCSS:F6}");
        }
        else if (Math.Abs(uniResult.TotalCost - mlWCSS) < 1e-10)
        {
            Console.WriteLine("   Both algorithms found the same optimal solution");
        }
        else
        {
            Console.WriteLine($"   UniCluster.Net found better solution by: {mlWCSS - uniResult.TotalCost:F6}");
        }
    }

    public class DataPoint
    {
        public float Value { get; set; }
    }

    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint ClusterId { get; set; }

        [ColumnName("Score")]
        public float[] Distances { get; set; } = Array.Empty<float>();
    }
}
