using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Toolchains.InProcess.Emit;
using Microsoft.ML;
using Microsoft.ML.Data;
using UniCluster.Net;

namespace UniCluster.Net.Benchmarks;

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net80)]
public class ClusteringBenchmark
{
    private double[]? _smallDataset;
    private double[]? _mediumDataset;
    private double[]? _largeDataset;
    private double[]? _veryLargeDataset;
    
    private MLContext? _mlContext;
    private OptimalKMeans1D? _optimalKMeans;

    [GlobalSetup]
    public void Setup()
    {
        _mlContext = new MLContext(seed: 42);
        _optimalKMeans = new OptimalKMeans1D();
        
        // Create datasets with different characteristics
        var random = new Random(42);
        
        // Small dataset (100 points)
        _smallDataset = GenerateClusteredData(100, 3, random);
        
        // Medium dataset (1,000 points)
        _mediumDataset = GenerateClusteredData(1000, 5, random);
        
        // Large dataset (10,000 points)
        _largeDataset = GenerateClusteredData(10000, 7, random);
        
        // Very large dataset (50,000 points)
        _veryLargeDataset = GenerateClusteredData(50000, 10, random);
    }

    private static double[] GenerateClusteredData(int count, int naturalClusters, Random random)
    {
        var data = new List<double>();
        var pointsPerCluster = count / naturalClusters;
        
        for (int cluster = 0; cluster < naturalClusters; cluster++)
        {
            var center = cluster * 10.0; // Centers at 0, 10, 20, etc.
            var pointsInThisCluster = cluster == naturalClusters - 1 
                ? count - (naturalClusters - 1) * pointsPerCluster  // Handle remainder
                : pointsPerCluster;
                
            for (int i = 0; i < pointsInThisCluster; i++)
            {
                // Generate points around the center with some variance
                data.Add(center + random.NextGaussian() * 1.5);
            }
        }
        
        return data.ToArray();
    }

    // Performance benchmarks for different dataset sizes
    [Benchmark]
    [Arguments(3)]
    public ClusteringResult UniCluster_Small_K3(int k) => _optimalKMeans!.Fit(_smallDataset!, k);

    [Benchmark]
    [Arguments(3)]
    public ClusterPrediction[] MLNet_Small_K3(int k) => RunMLNetClustering(_smallDataset!, k);

    [Benchmark]
    [Arguments(5)]
    public ClusteringResult UniCluster_Medium_K5(int k) => _optimalKMeans!.Fit(_mediumDataset!, k);

    [Benchmark]
    [Arguments(5)]
    public ClusterPrediction[] MLNet_Medium_K5(int k) => RunMLNetClustering(_mediumDataset!, k);

    [Benchmark]
    [Arguments(7)]
    public ClusteringResult UniCluster_Large_K7(int k) => _optimalKMeans!.Fit(_largeDataset!, k);

    [Benchmark]
    [Arguments(7)]
    public ClusterPrediction[] MLNet_Large_K7(int k) => RunMLNetClustering(_largeDataset!, k);

    [Benchmark]
    [Arguments(10)]
    public ClusteringResult UniCluster_VeryLarge_K10(int k) => _optimalKMeans!.Fit(_veryLargeDataset!, k);

    [Benchmark]
    [Arguments(10)]
    public ClusterPrediction[] MLNet_VeryLarge_K10(int k) => RunMLNetClustering(_veryLargeDataset!, k);

    private ClusterPrediction[] RunMLNetClustering(double[] data, int k)
    {
        var dataPoints = data.Select(x => new DataPoint { Value = (float)x }).ToArray();
        var dataView = _mlContext!.Data.LoadFromEnumerable(dataPoints);

        var pipeline = _mlContext.Transforms.Concatenate("Features", nameof(DataPoint.Value))
            .Append(_mlContext.Clustering.Trainers.KMeans(
                featureColumnName: "Features",
                numberOfClusters: k));

        var model = pipeline.Fit(dataView);
        var predictions = model.Transform(dataView);
        
        return _mlContext.Data.CreateEnumerable<ClusterPrediction>(predictions, reuseRowObject: false).ToArray();
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

// Quality comparison benchmark - measures clustering quality
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net80)]
public class QualityComparisonBenchmark
{
    private double[]? _testData;
    private MLContext? _mlContext;
    private OptimalKMeans1D? _optimalKMeans;

    [GlobalSetup]
    public void Setup()
    {
        _mlContext = new MLContext(seed: 42);
        _optimalKMeans = new OptimalKMeans1D();
        
        // Create a dataset with known optimal structure
        var random = new Random(42);
        var data = new List<double>();
        
        // Three well-separated clusters
        for (int i = 0; i < 50; i++) data.Add(random.NextGaussian() * 0.5 + 1.0);   // Cluster 1: around 1.0
        for (int i = 0; i < 50; i++) data.Add(random.NextGaussian() * 0.5 + 10.0);  // Cluster 2: around 10.0  
        for (int i = 0; i < 50; i++) data.Add(random.NextGaussian() * 0.5 + 20.0);  // Cluster 3: around 20.0
        
        _testData = data.ToArray();
    }

    [Benchmark]
    public QualityResult UniCluster_Quality() => MeasureUniClusterQuality();

    [Benchmark]
    public QualityResult MLNet_Quality() => MeasureMLNetQuality();

    private QualityResult MeasureUniClusterQuality()
    {
        var result = _optimalKMeans!.Fit(_testData!, 3);
        return new QualityResult 
        { 
            WCSS = result.TotalCost,
            ClusterCount = result.ClusterCount
        };
    }

    private QualityResult MeasureMLNetQuality()
    {
        var dataPoints = _testData!.Select(x => new ClusteringBenchmark.DataPoint { Value = (float)x }).ToArray();
        var dataView = _mlContext!.Data.LoadFromEnumerable(dataPoints);

        var pipeline = _mlContext.Transforms.Concatenate("Features", nameof(ClusteringBenchmark.DataPoint.Value))
            .Append(_mlContext.Clustering.Trainers.KMeans(
                featureColumnName: "Features",
                numberOfClusters: 3));

        var model = pipeline.Fit(dataView);
        var predictions = model.Transform(dataView);
        
        var results = _mlContext.Data.CreateEnumerable<ClusteringBenchmark.ClusterPrediction>(predictions, reuseRowObject: false).ToArray();
        
        // Calculate WCSS manually for ML.NET results
        var wcss = 0.0;
        var clusterCenters = new Dictionary<uint, List<double>>();
        
        for (int i = 0; i < results.Length; i++)
        {
            var clusterId = results[i].ClusterId;
            if (!clusterCenters.ContainsKey(clusterId))
                clusterCenters[clusterId] = new List<double>();
            clusterCenters[clusterId].Add(_testData![i]);
        }
        
        foreach (var cluster in clusterCenters.Values)
        {
            var centroid = cluster.Average();
            wcss += cluster.Sum(point => Math.Pow(point - centroid, 2));
        }
        
        return new QualityResult 
        { 
            WCSS = wcss,
            ClusterCount = clusterCenters.Count
        };
    }

    public class QualityResult
    {
        public double WCSS { get; set; }
        public int ClusterCount { get; set; }
    }
}

// Stability benchmark - tests how consistent results are across multiple runs
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net80)]
public class StabilityBenchmark
{
    private double[]? _testData;
    private MLContext? _mlContext;
    private OptimalKMeans1D? _optimalKMeans;

    [GlobalSetup]
    public void Setup()
    {
        _optimalKMeans = new OptimalKMeans1D();
        
        // Create challenging dataset for stability testing
        var random = new Random(42);
        var data = new List<double>();
        
        // Create overlapping clusters to test stability
        for (int i = 0; i < 30; i++) data.Add(random.NextGaussian() * 2.0 + 5.0);   // Cluster 1
        for (int i = 0; i < 30; i++) data.Add(random.NextGaussian() * 2.0 + 8.0);   // Cluster 2 (overlapping)
        for (int i = 0; i < 30; i++) data.Add(random.NextGaussian() * 2.0 + 15.0);  // Cluster 3
        
        _testData = data.ToArray();
    }

    [Benchmark]
    public StabilityResult UniCluster_Stability() => MeasureUniClusterStability();

    [Benchmark]
    public StabilityResult MLNet_Stability() => MeasureMLNetStability();

    private StabilityResult MeasureUniClusterStability()
    {
        // UniCluster is deterministic, so all runs should be identical
        var results = new List<double>();
        for (int run = 0; run < 10; run++)
        {
            var result = _optimalKMeans!.Fit(_testData!, 3);
            results.Add(result.TotalCost);
        }
        
        return new StabilityResult
        {
            MeanWCSS = results.Average(),
            StdDevWCSS = CalculateStandardDeviation(results),
            IsStable = results.All(x => Math.Abs(x - results[0]) < 1e-10)
        };
    }

    private StabilityResult MeasureMLNetStability()
    {
        var results = new List<double>();
        
        for (int run = 0; run < 10; run++)
        {
            var mlContext = new MLContext(seed: run); // Different seed each time
            var dataPoints = _testData!.Select(x => new ClusteringBenchmark.DataPoint { Value = (float)x }).ToArray();
            var dataView = mlContext.Data.LoadFromEnumerable(dataPoints);

            var pipeline = mlContext.Transforms.Concatenate("Features", nameof(ClusteringBenchmark.DataPoint.Value))
                .Append(mlContext.Clustering.Trainers.KMeans(
                    featureColumnName: "Features",
                    numberOfClusters: 3));

            var model = pipeline.Fit(dataView);
            var predictions = model.Transform(dataView);
            
            var predictionResults = mlContext.Data.CreateEnumerable<ClusteringBenchmark.ClusterPrediction>(predictions, reuseRowObject: false).ToArray();
            
            // Calculate WCSS
            var wcss = 0.0;
            var clusterCenters = new Dictionary<uint, List<double>>();
            
            for (int i = 0; i < predictionResults.Length; i++)
            {
                var clusterId = predictionResults[i].ClusterId;
                if (!clusterCenters.ContainsKey(clusterId))
                    clusterCenters[clusterId] = new List<double>();
                clusterCenters[clusterId].Add(_testData![i]);
            }
            
            foreach (var cluster in clusterCenters.Values)
            {
                var centroid = cluster.Average();
                wcss += cluster.Sum(point => Math.Pow(point - centroid, 2));
            }
            
            results.Add(wcss);
        }
        
        var stdDev = CalculateStandardDeviation(results);
        
        return new StabilityResult
        {
            MeanWCSS = results.Average(),
            StdDevWCSS = stdDev,
            IsStable = stdDev < 0.1 // Threshold for "stable"
        };
    }

    private static double CalculateStandardDeviation(List<double> values)
    {
        var mean = values.Average();
        var sumOfSquaredDifferences = values.Sum(x => Math.Pow(x - mean, 2));
        return Math.Sqrt(sumOfSquaredDifferences / values.Count);
    }

    public class StabilityResult
    {
        public double MeanWCSS { get; set; }
        public double StdDevWCSS { get; set; }
        public bool IsStable { get; set; }
    }
}

public static class RandomExtensions
{
    public static double NextGaussian(this Random random, double mean = 0, double stdDev = 1)
    {
        // Box-Muller transform
        if (random == null) throw new ArgumentNullException(nameof(random));
        
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * randStdNormal;
    }
}

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("🚀 UniCluster.Net Benchmarks");
        Console.WriteLine("============================");
        Console.WriteLine();

        if (args.Length > 0 && args[0] == "--quick")
        {
            QuickBenchmark.RunQuickComparison();
            return;
        }

        if (args.Length > 0 && args[0] == "--verify")
        {
            VerifyResults.RunVerification();
            return;
        }

        Console.WriteLine("Running comprehensive benchmarks comparing UniCluster.Net with ML.NET...");
        Console.WriteLine("This may take several minutes to complete.");
        Console.WriteLine("Use --quick flag for a faster comparison.");
        Console.WriteLine("Use --verify flag to verify benchmark accuracy.");
        Console.WriteLine();

        var config = DefaultConfig.Instance
            .AddJob(Job.Default.WithToolchain(InProcessEmitToolchain.Instance));

        // Run performance benchmarks
        Console.WriteLine("📊 Performance Benchmarks");
        BenchmarkRunner.Run<ClusteringBenchmark>(config);
        
        Console.WriteLine();
        Console.WriteLine("🎯 Quality Comparison Benchmarks");
        BenchmarkRunner.Run<QualityComparisonBenchmark>(config);
        
        Console.WriteLine();
        Console.WriteLine("🔄 Stability Benchmarks");
        BenchmarkRunner.Run<StabilityBenchmark>(config);
        
        Console.WriteLine();
        Console.WriteLine("✅ All benchmarks completed!");
        Console.WriteLine("Results have been saved to BenchmarkDotNet.Artifacts folder.");
    }
}
