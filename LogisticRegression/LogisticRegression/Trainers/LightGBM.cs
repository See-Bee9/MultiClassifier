using Microsoft.ML;

namespace LogisticRegression.Trainers
{
    public class LightGBM : TrainerBaseClass
    {
        public override void Train(string directoryPath, string modelPath)
        {
            Train(directoryPath, modelPath, Constants.mlContext.MulticlassClassification.Trainers.LightGbm());
        }
    }
}