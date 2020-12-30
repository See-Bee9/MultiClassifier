namespace LogisticRegression.Trainers
{
    public interface ITrainer
    {
        void Train(string directoryPath, string modelPath);
    }
}