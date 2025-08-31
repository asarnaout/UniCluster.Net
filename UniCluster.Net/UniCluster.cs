namespace UniCluster.Net;

public class UniCluster
{

    //TODO: Emphasize that values should be sorted
    public void Fit(double[] values, int numberOfClusters)
    {
        if (numberOfClusters > values.Length)
        {
            throw new ArgumentException("Number of clusters must be less than or equal to the number of values.");
        }

    }
}
