using CommandLine;

namespace LogisticRegression.CLI
{
    public class Options
    {
        [Option]      public bool   Predict { get; set; }
        [Option('t')] public string Trainer { get; set; }
        public               bool   Train   => Trainer != null;

        [Option('f', "file-path", HelpText = "The file path to a trainer or a training folder.")]
        public string FilePath { get; set; }

        [Option('o', "output-path", HelpText = "The output target for a trainer file.")]
        public string OutputPath { get; set; }
    }
}