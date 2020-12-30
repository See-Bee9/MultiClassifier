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

            // Load the data from files into the training model object.
            // The object uses the decorator pattern to control mapping from file columns into object fields.
            // This particular method using the $"{directoryPath}\\*" string will read all files from a directory.
            var data = mlContext.Data.LoadFromTextFile<TrainingModel>($"{directoryPath}\\*", separatorChar: '\t',
                hasHeader: false);


            var pipeline = mlContext.Transforms.Conversion
                    // "Label" is a recognized value for the ML library
                    // MapValueToKey will assign Classification from the model to Label in the pipeline
                .MapValueToKey(inputColumnName: "Classification", outputColumnName: "Label")
                    // FeaturizeText converts the text of the file into a numerical vector
                    // We are appending the vectorized text to the training pipeline
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Content",
                    outputColumnName: "ContentFeaturized"))
                    // The Concatenate method transforms all of the previous features into a single Features column
                .Append(mlContext.Transforms.Concatenate("Features", "ContentFeaturized"));
            var dataPipeline = pipeline
                    // Adding the trainer to the pipeline
                .Append(trainer)
                    // Map the output of the algorithm into the PredictedLabel column of the prediction model.
                    // PredictedLabel mapping is determined by a decorator on the output class.
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var trainedModel = dataPipeline.Fit(data);

            mlContext.Model.Save(trainedModel, data.Schema, modelPath);

            // This is only useful for demonstration purposes.
            // When doing a real evaluation we would want to use a set of files that was not used to train the model.
            var testDataView = mlContext.Data.LoadFromTextFile<TrainingModel>(Path.Combine(directoryPath,
                /*This is just a random file to use for demonstration*/"2009.07123.tsv"), '\t', hasHeader: false);
            var modelMetrics = mlContext.MulticlassClassification.Evaluate(trainedModel.Transform(testDataView));
            Console.WriteLine($"MicroAccuracy: {modelMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"MacroAccuracy: {modelMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"LogLoss: {modelMetrics.LogLoss:#.###}");
            Console.WriteLine($"LogLossReduction: {modelMetrics.LogLossReduction:#.###}");
        }

        public abstract void Train(string directoryPath, string modelPath);
    }
}