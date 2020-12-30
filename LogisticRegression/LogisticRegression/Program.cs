using System;
using System.IO;
using CommandLine;
using LogisticRegression.CLI;
using LogisticRegression.Models;
using LogisticRegression.Trainers;

namespace LogisticRegression
{
    class Program
    {
        static void Main(string[] args)
        {
            CommandLine.Parser.Default.ParseArguments<TrainOptions, PredictOptions>(args)
                .WithParsed<TrainOptions>(Train)
                .WithParsed<PredictOptions>(Predict)
                .WithNotParsed(o => throw new Exception("Must select train or predict."));
        }

        private static void Train(TrainOptions options)
        {
            if (options.OutputPath == null) throw new ArgumentNullException(nameof(options.OutputPath));
            if (options.InputDirectory == null) throw new ArgumentNullException(nameof(options.InputDirectory));
            if (options.GBM || options.SDCA)
            {
                ITrainer trainer;
                if (options.GBM) trainer = new SDCATrainer();
                else trainer             = new LightGBM();
                trainer.Train(options.InputDirectory, options.OutputPath);
            }

            else
            {
                throw new Exception("Must select SDCA or GBM");
            }
        }

        private static void Predict(PredictOptions options)
        {
            if (options.FolderPath == null) throw new ArgumentNullException(nameof(options.FolderPath));
            if (options.TrainerPath == null) throw new ArgumentNullException(nameof(options.TrainerPath));
            var predictor = new Predictor();
            var files     = Directory.GetFiles(options.FolderPath);
            foreach (var file in files)
            {
                var fileName = Path.GetFileName(file).Split('.')[0];
                var trainingModel = new TrainingModel
                {
                    Content = File.ReadAllText(file)
                };
                var result = predictor.Predict(options.TrainerPath, trainingModel);
                Console.WriteLine(
                    $"{fileName} : {result.Classification}"
                    + $"{Environment.NewLine}\t Proton Confidence : {result.Score[0]:0.###}"
                    + $"{Environment.NewLine}\t Ecology Confidence : {result.Score[1]:0.###}");
            }
        }
    }
}