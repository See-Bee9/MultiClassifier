using System;
using CommandLine;
using LogisticRegression.CLI;
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
        }
    }
}