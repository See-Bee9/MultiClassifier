namespace LogisticRegression
{
    public interface ITrainer
    {
        void Train(string directoryPath, string modelPath);
    }
}