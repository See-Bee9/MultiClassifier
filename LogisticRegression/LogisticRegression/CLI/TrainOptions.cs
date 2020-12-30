using CommandLine;

namespace LogisticRegression.CLI
{
    [Verb("train")]
    public class TrainOptions
    {
        [Option("gbm")]         public bool   GBM            { get; set; }
        [Option("SDCA")]        public bool   SDCA           { get; set; }
        [Option('o', "output")] public string OutputPath     { get; set; }
        [Option('i',"input")]   public string InputDirectory { get; set; }
    }
}