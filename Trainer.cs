using DocumentClassifier.Data;
using Microsoft.ML;

namespace DocumentClassifier
{
    public class Trainer
    {
        public void Train(string directoryPath, string modelPath)
        {
            var mlContext = Constants.mlContext;

            var data = mlContext.Data.LoadFromTextFile<TrainingModel>($"{directoryPath}\\*", separatorChar: '\t', hasHeader: false);

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Classification", outputColumnName: "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Content", outputColumnName: "ContentFeaturized"))
                .Append(mlContext.Transforms.Concatenate("Features", "ContentFeaturized"));

            var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy();
            var dataPipeline = pipeline.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var trainedModel = dataPipeline.Fit(data);
            var modelName    = System.IO.Path.Combine(modelPath, "model.zip");
            mlContext.Model.Save(trainedModel, data.Schema, modelName);
        }

        public Prediction Predict(string modelPath, TrainingModel trainingModel)
        {
            var model      = Constants.mlContext.Model.Load(modelPath, out var modelInputSchema);
            var engine     = Constants.mlContext.Model.CreatePredictionEngine<TrainingModel, Prediction>(model);
            var prediction = engine.Predict(trainingModel);
            return prediction;
        }
    }
}