using LogisticRegression.Models;

namespace LogisticRegression
{
    public class Predictor
    {
        public Prediction Predict(string modelPath, TrainingModel trainingModel)
        {
            var model      = Constants.mlContext.Model.Load(modelPath, out var modelInputSchema);
            var engine     = Constants.mlContext.Model.CreatePredictionEngine<TrainingModel, Prediction>(model);
            var prediction = engine.Predict(trainingModel);
            return prediction;
        }
    }
}