using System;
using System.IO;
using LogisticRegression.Models;
using Microsoft.ML;

namespace LogisticRegression.Trainers
{
    public abstract class TrainerBaseClass : ITrainer
    {
        public void Train(string directoryPath, string modelPath, IEstimator<ITransformer> trainer)
        {
            var mlContext = Constants.mlContext;

            var data = mlContext.Data.LoadFromTextFile<TrainingModel>($"{directoryPath}\\*", separatorChar: '\t',
                hasHeader: false);

            var pipeline = mlContext.Transforms.Conversion
                .MapValueToKey(inputColumnName: "Classification", outputColumnName: "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Content",
                    outputColumnName: "ContentFeaturized"))
                .Append(mlContext.Transforms.Concatenate("Features", "ContentFeaturized"));
            // var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy();
            var dataPipeline = pipeline
                .Append(trainer)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var trainedModel = dataPipeline.Fit(data);

            mlContext.Model.Save(trainedModel, data.Schema, modelPath);

            var testDataView = mlContext.Data.LoadFromTextFile<TrainingModel>(Path.Combine(directoryPath,"2009.07123.tsv"), '\t', hasHeader: false);
            var modelMetrics = mlContext.MulticlassClassification.Evaluate(trainedModel.Transform(testDataView));
            Console.WriteLine($"MicroAccuracy: {modelMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"MacroAccuracy: {modelMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"LogLoss: {modelMetrics.LogLoss:#.###}");
            Console.WriteLine($"LogLossReduction: {modelMetrics.LogLossReduction:#.###}");
        }

        public abstract void Train(string directoryPath, string modelPath);
    }
}