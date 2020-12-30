using CommandLine;

namespace LogisticRegression.CLI
{
    [Verb("predict")]
    public class PredictOptions
    {
        [Option('t', "trainer-path")] public string TrainerPath { get; set; }
        [Option('f', "folder-path")]  public string FolderPath  { get; set; }
    }
}