﻿using LogisticRegression.Models;
using Microsoft.ML;

namespace LogisticRegression
{
    public class LogisticRegressionTrainer : ITrainer
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
    }
}