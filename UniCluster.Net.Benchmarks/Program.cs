using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Microsoft.ML;
using Microsoft.ML.Data;
using UniCluster.Net;

namespace UniCluster.Net.Benchmarks
{
    public class Program
    {
        public static void Main(string[] args)
        {
            BenchmarkRunner.Run<UniClusterVsMLNetBenchmark>();
        }
    }
}

[MemoryDiagnoser]
[SimpleJob]
public class UniClusterVsMLNetBenchmark
{
    private double[] _smallDataset = null!;
    private double[] _mediumDataset = null!;
    private double[] _largeDataset = null!;
    private OptimalKMeans1D _optimalKMeans = null!;
    
    [GlobalSetup]
    public void GlobalSetup()
    {
        _optimalKMeans = new OptimalKMeans1D();
        
        // Small dataset: 30 points, 3 clear clusters
        _smallDataset = GenerateClusteredData([10, 50, 90], [10, 10, 10], 42);
        
        // Medium dataset: 300 points, 5 clusters  
        _mediumDataset = GenerateClusteredData([10, 30, 50, 70, 90], [60, 60, 60, 60, 60], 42);
        
        // Large dataset: 1000 points, 8 clusters
        _largeDataset = GenerateClusteredData([5, 15, 25, 35, 45, 55, 65, 75], [125, 125, 125, 125, 125, 125, 125, 125], 42);
    }
    
    // ===== SMALL DATASET BENCHMARKS (30 points, k=3) =====
    
    [Benchmark(Baseline = true)]
    [BenchmarkCategory("Small")]
    public ClusteringResult UniCluster_Small()
    {
        return _optimalKMeans.Fit(_smallDataset, 3);
    }
    
    [Benchmark]
    [BenchmarkCategory("Small")]
    public MLNetResult MLNet_Small_SingleRun()
    {
        return RunMLNet(_smallDataset, 3, seed: 42);
    }
    
    [Benchmark]
    [BenchmarkCategory("Small")]
    public MLNetResult MLNet_Small_BestOf10()
    {
        return FindBestMLNetResult(_smallDataset, 3, attempts: 10);
    }
    
    // ===== MEDIUM DATASET BENCHMARKS (300 points, k=5) =====
    
    [Benchmark]
    [BenchmarkCategory("Medium")]
    public ClusteringResult UniCluster_Medium()
    {
        return _optimalKMeans.Fit(_mediumDataset, 5);
    }
    
    [Benchmark]
    [BenchmarkCategory("Medium")]
    public MLNetResult MLNet_Medium_SingleRun()
    {
        return RunMLNet(_mediumDataset, 5, seed: 42);
    }
    
    [Benchmark]
    [BenchmarkCategory("Medium")]
    public MLNetResult MLNet_Medium_BestOf10()
    {
        return FindBestMLNetResult(_mediumDataset, 5, attempts: 10);
    }
    
    // ===== LARGE DATASET BENCHMARKS (1000 points, k=8) =====
    
    [Benchmark]
    [BenchmarkCategory("Large")]
    public ClusteringResult UniCluster_Large()
    {
        return _optimalKMeans.Fit(_largeDataset, 8);
    }
    
    [Benchmark]
    [BenchmarkCategory("Large")]
    public MLNetResult MLNet_Large_SingleRun()
    {
        return RunMLNet(_largeDataset, 8, seed: 42);
    }
    
    [Benchmark]
    [BenchmarkCategory("Large")]
    public MLNetResult MLNet_Large_BestOf5()
    {
        return FindBestMLNetResult(_largeDataset, 8, attempts: 5);
    }
    
    // ===== HELPER METHODS =====
    
    private MLNetResult RunMLNet(double[] data, int k, int seed)
    {
        var mlContext = new MLContext(seed: seed);
        
        var dataPoints = data.Select(x => new DataPoint { Value = (float)x, OriginalValue = x }).ToArray();
        var dataView = mlContext.Data.LoadFromEnumerable(dataPoints);
        
        var pipeline = mlContext.Transforms.Concatenate("Features", "Value")
            .Append(mlContext.Clustering.Trainers.KMeans("Features", numberOfClusters: k));
        
        var model = pipeline.Fit(dataView);
        var predictions = model.Transform(dataView);
        var results = mlContext.Data.CreateEnumerable<ClusterPrediction>(predictions, false).ToList();
        
        // Calculate WCSS manually for fair comparison
        double wcss = CalculateWCSS(results);
        
        return new MLNetResult { WCSS = wcss, ClusterCount = k };
    }
    
    private MLNetResult FindBestMLNetResult(double[] data, int k, int attempts)
    {
        var bestResult = RunMLNet(data, k, seed: 1);
        
        for (int seed = 2; seed <= attempts; seed++)
        {
            var result = RunMLNet(data, k, seed);
            if (result.WCSS < bestResult.WCSS)
            {
                bestResult = result;
            }
        }
        
        return bestResult;
    }
    
    private double CalculateWCSS(List<ClusterPrediction> predictions)
    {
        var clusters = predictions.GroupBy(p => p.PredictedClusterId)
            .ToDictionary(g => g.Key, g => g.Select(p => p.OriginalValue).ToList());
        
        double totalWCSS = 0;
        foreach (var cluster in clusters.Values)
        {
            if (cluster.Any())
            {
                var centroid = cluster.Average();
                totalWCSS += cluster.Sum(point => Math.Pow(point - centroid, 2));
            }
        }
        
        return totalWCSS;
    }
    
    private static double[] GenerateClusteredData(double[] centers, int[] sizes, int seed)
    {
        var random = new Random(seed);
        var data = new List<double>();
        
        for (int i = 0; i < centers.Length; i++)
        {
            for (int j = 0; j < sizes[i]; j++)
            {
                // Add some noise around each center
                var point = centers[i] + random.NextGaussian() * 2.0; // Standard deviation of 2
                data.Add(point);
            }
        }
        
        // Shuffle to make it more realistic
        for (int i = data.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (data[i], data[j]) = (data[j], data[i]);
        }
        
        return data.ToArray();
    }
}

public class DataPoint
{
    public float Value { get; set; }
    public double OriginalValue { get; set; }
}

public class ClusterPrediction
{
    [ColumnName("PredictedLabel")]
    public uint PredictedClusterId { get; set; }
    
    public double OriginalValue { get; set; }
}

public class MLNetResult
{
    public double WCSS { get; set; }
    public int ClusterCount { get; set; }
}

// Extension method for Gaussian random numbers
public static class RandomExtensions
{
    public static double NextGaussian(this Random random)
    {
        // Box-Muller transformation
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }
}