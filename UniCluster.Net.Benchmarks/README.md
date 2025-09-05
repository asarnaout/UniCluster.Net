# UniCluster.Net vs ML.NET K-means Benchmark

This benchmark compares UniCluster.Net's optimal 1D K-means implementation against ML.NET's traditional K-means algorithm.

## Key Comparison Points

### 1. **Deterministic vs Probabilistic**
- **UniCluster.Net**: Guarantees the same optimal result every time
- **ML.NET**: Results vary with different random seeds due to random initialization

### 2. **Solution Quality**
- **UniCluster.Net**: Guarantees globally optimal solution
- **ML.NET Single Run**: May find local optima (good but not necessarily optimal)
- **ML.NET Best-of-N**: Tries multiple seeds and picks the best result

### 3. **Performance Characteristics**
- **UniCluster.Net**: Predictable O(k·n) time complexity
- **ML.NET**: Iterative algorithm, time varies based on convergence

### 4. **Scalability**
- **Small datasets (30 points)**: Both perform well
- **Medium datasets (300 points)**: Performance differences become apparent
- **Large datasets (1000+ points)**: Trade-offs between speed and optimality

## Benchmark Categories

### Small Dataset (30 points, k=3)
- Tests basic functionality and overhead
- Both algorithms should perform well

### Medium Dataset (300 points, k=5)
- Shows performance differences
- Tests scalability characteristics

### Large Dataset (1000 points, k=8)
- Demonstrates algorithm behavior at scale
- Shows real-world performance trade-offs

## Running the Benchmark

```bash
# Run all benchmarks
dotnet run -c Release

# Run specific category
dotnet run -c Release --filter "*Small*"
dotnet run -c Release --filter "*Medium*"
dotnet run -c Release --filter "*Large*"
```

## Expected Results

### Performance
- **UniCluster.Net**: Consistent, predictable timing
- **ML.NET Single Run**: Fast but variable quality
- **ML.NET Best-of-N**: Slower but better quality (still may not match optimal)

### Solution Quality
- **UniCluster.Net**: Always finds the mathematically optimal solution
- **ML.NET**: May find competitive solutions but not guaranteed optimal

### Memory Usage
- **UniCluster.Net**: Uses dynamic programming tables (higher memory for large k)
- **ML.NET**: More memory-efficient iterative approach

## Interpretation Guidelines

1. **When UniCluster.Net is slower**: This is expected - optimality comes at a cost
2. **When ML.NET finds "better" WCSS**: Double-check - this shouldn't happen with correct implementation
3. **Consistency**: UniCluster.Net results should be identical across runs, ML.NET should vary

## Scientific Value

This benchmark demonstrates:
- **Theoretical optimality vs practical performance** trade-offs
- **Deterministic vs probabilistic** algorithm behavior
- **Guaranteed quality vs speed** considerations
- **Real-world applicability** of different approaches

The goal is not to prove one algorithm is universally better, but to show the trade-offs and help users choose the right tool for their specific needs.