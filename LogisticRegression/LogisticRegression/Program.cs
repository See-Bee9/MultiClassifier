using CommandLine;
using LogisticRegression.CLI;

namespace LogisticRegression
{
    class Program
    {
        static void Main(string[] args)
        {
            CommandLine.Parser.Default.ParseArguments<Options>(args)
                .WithParsed(opt =>
                {

                });
        }
    }
}