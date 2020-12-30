using Microsoft.ML.Data;

namespace LogisticRegression.Models
{
    public class TrainingModel
    {
        [LoadColumn(0)] public string Classification { get; set; }
        [LoadColumn(1)] public string Content        { get; set; }
    }
}