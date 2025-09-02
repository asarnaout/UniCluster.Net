using System.Diagnostics;
using Microsoft.ML;
using Microsoft.ML.Data;
using UniCluster.Net;

namespace UniCluster.Net.Benchmarks;

public class QuickBenchmark
{
    public static void RunQuickComparison()
    {
        Console.WriteLine("🚀 Quick Performance & Quality Comparison");
        Console.WriteLine("==========================================");
        Console.WriteLine();

        var random = new Random(42);
        
        // Test different dataset sizes
        var datasets = new[]
        {
            ("Small (100 points)", GenerateClusteredData(100, 3, random), 3),
            ("Medium (1,000 points)", GenerateClusteredData(1000, 5, random), 5),
            ("Large (10,000 points)", GenerateClusteredData(10000, 7, random), 7)
        };

        var uniCluster = new OptimalKMeans1D();

        foreach (var (name, data, k) in datasets)
        {
            Console.WriteLine($"📊 {name}, K={k}");
            Console.WriteLine(new string('-', 50));

            // UniCluster.Net benchmark
            var sw = Stopwatch.StartNew();
            var uniResult = uniCluster.Fit(data, k);
            sw.Stop();
            var uniTime = sw.Elapsed.TotalMilliseconds;

            // ML.NET benchmark
            sw.Restart();
            var mlResult = RunMLNetClustering(data, k);
            sw.Stop();
            var mlTime = sw.Elapsed.TotalMilliseconds;

            // Quality comparison - calculate WCSS for ML.NET
            var mlWCSS = CalculateMLNetWCSS(data, mlResult);

            Console.WriteLine($"UniCluster.Net: {uniTime:F2} ms, WCSS: {uniResult.TotalCost:F4}");
            Console.WriteLine($"ML.NET:        {mlTime:F2} ms, WCSS: {mlWCSS:F4}");
            Console.WriteLine($"Speed ratio:    {mlTime / uniTime:F1}x (UniCluster.Net vs ML.NET)");
            Console.WriteLine($"Quality ratio:  {mlWCSS / uniResult.TotalCost:F2}x (ML.NET WCSS / UniCluster WCSS)");
            Console.WriteLine();
        }

        // Stability test
        Console.WriteLine("🔄 Stability Test (overlapping clusters)");
        Console.WriteLine(new string('-', 50));
        RunStabilityTest();
    }

    private static double[] GenerateClusteredData(int count, int naturalClusters, Random random)
    {
        var data = new List<double>();
        var pointsPerCluster = count / naturalClusters;
        
        for (int cluster = 0; cluster < naturalClusters; cluster++)
        {
            var center = cluster * 10.0;
            var pointsInThisCluster = cluster == naturalClusters - 1 
                ? count - (naturalClusters - 1) * pointsPerCluster
                : pointsPerCluster;
                
            for (int i = 0; i < pointsInThisCluster; i++)
            {
                data.Add(center + NextGaussian(random) * 1.5);
            }
        }
        
        return data.ToArray();
    }

    private static ClusterPrediction[] RunMLNetClustering(double[] data, int k)
    {
        var mlContext = new MLContext(seed: 42);
        var dataPoints = data.Select(x => new DataPoint { Value = (float)x }).ToArray();
        var dataView = mlContext.Data.LoadFromEnumerable(dataPoints);

        // Transform the scalar feature into a vector for ML.NET
        var pipeline = mlContext.Transforms.Concatenate("Features", nameof(DataPoint.Value))
            .Append(mlContext.Clustering.Trainers.KMeans(
                featureColumnName: "Features",
                numberOfClusters: k));

        var model = pipeline.Fit(dataView);
        var predictions = model.Transform(dataView);
        
        return mlContext.Data.CreateEnumerable<ClusterPrediction>(predictions, reuseRowObject: false).ToArray();
    }

    private static double CalculateMLNetWCSS(double[] data, ClusterPrediction[] predictions)
    {
        var clusterCenters = new Dictionary<uint, List<double>>();
        
        for (int i = 0; i < predictions.Length; i++)
        {
            var clusterId = predictions[i].ClusterId;
            if (!clusterCenters.ContainsKey(clusterId))
                clusterCenters[clusterId] = new List<double>();
            clusterCenters[clusterId].Add(data[i]);
        }
        
        var wcss = 0.0;
        foreach (var cluster in clusterCenters.Values)
        {
            var centroid = cluster.Average();
            wcss += cluster.Sum(point => Math.Pow(point - centroid, 2));
        }
        
        return wcss;
    }

    private static void RunStabilityTest()
    {
        var uniCluster = new OptimalKMeans1D();
        var testData = GenerateOverlappingClusters();
        
        // UniCluster.Net stability (should be 100% stable)
        var uniResults = new List<double>();
        for (int i = 0; i < 10; i++)
        {
            var result = uniCluster.Fit(testData, 3);
            uniResults.Add(result.TotalCost);
        }
        
        // ML.NET stability (may vary)
        var mlResults = new List<double>();
        for (int i = 0; i < 10; i++)
        {
            var predictions = RunMLNetClustering(testData, 3, seed: i);
            var wcss = CalculateMLNetWCSS(testData, predictions);
            mlResults.Add(wcss);
        }
        
        var uniStdDev = CalculateStandardDeviation(uniResults);
        var mlStdDev = CalculateStandardDeviation(mlResults);
        
        Console.WriteLine($"UniCluster.Net: Mean WCSS = {uniResults.Average():F4}, Std Dev = {uniStdDev:F6}");
        Console.WriteLine($"ML.NET:        Mean WCSS = {mlResults.Average():F4}, Std Dev = {mlStdDev:F6}");
        Console.WriteLine($"Stability:      UniCluster is {(uniStdDev < 1e-10 ? "100%" : "variable")}, ML.NET varies by {mlStdDev:F4}");
        Console.WriteLine();
    }

    private static double[] GenerateOverlappingClusters()
    {
        var random = new Random(42);
        var data = new List<double>();
        
        // Create overlapping clusters to test stability
        for (int i = 0; i < 30; i++) data.Add(NextGaussian(random) * 2.0 + 5.0);   // Cluster 1
        for (int i = 0; i < 30; i++) data.Add(NextGaussian(random) * 2.0 + 8.0);   // Cluster 2 (overlapping)
        for (int i = 0; i < 30; i++) data.Add(NextGaussian(random) * 2.0 + 15.0);  // Cluster 3
        
        return data.ToArray();
    }

    private static ClusterPrediction[] RunMLNetClustering(double[] data, int k, int seed)
    {
        var mlContext = new MLContext(seed: seed);
        var dataPoints = data.Select(x => new DataPoint { Value = (float)x }).ToArray();
        var dataView = mlContext.Data.LoadFromEnumerable(dataPoints);

        // Transform the scalar feature into a vector for ML.NET
        var pipeline = mlContext.Transforms.Concatenate("Features", nameof(DataPoint.Value))
            .Append(mlContext.Clustering.Trainers.KMeans(
                featureColumnName: "Features",
                numberOfClusters: k));

        var model = pipeline.Fit(dataView);
        var predictions = model.Transform(dataView);
        
        return mlContext.Data.CreateEnumerable<ClusterPrediction>(predictions, reuseRowObject: false).ToArray();
    }

    private static double CalculateStandardDeviation(List<double> values)
    {
        var mean = values.Average();
        var sumOfSquaredDifferences = values.Sum(x => Math.Pow(x - mean, 2));
        return Math.Sqrt(sumOfSquaredDifferences / values.Count);
    }

    private static double NextGaussian(Random random, double mean = 0, double stdDev = 1)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * randStdNormal;
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
