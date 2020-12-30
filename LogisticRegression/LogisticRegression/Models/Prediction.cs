using Microsoft.ML.Data;

namespace LogisticRegression.Models
{
    public class Prediction
    {
        [ColumnName("PredictedLabel")] public string  Classification;
        [ColumnName("Score")]          public float[] Score;
    }
}