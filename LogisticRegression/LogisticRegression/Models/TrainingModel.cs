using Microsoft.ML.Data;

namespace LogisticRegression.Models
{
    public class TrainingModel
    {
        [LoadColumn(0)] public string Type { get; set; }
        [LoadColumn(1)] public string Text { get; set; }
    }
}