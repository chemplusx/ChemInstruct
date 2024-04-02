package tools.utils;

import org.apache.commons.cli.*;

import java.util.HashMap;

public class ArgumentParser {

    public ArgumentParser(){}

    public HashMap<String, String> ParseCliArguments(String[] args) {
        HashMap<String, String> arguments = new HashMap<>();

        Options options = new Options();

        Option datasetName = new Option("ds", "dataset", true, "Dataset Name");
        datasetName.setRequired(true);
        options.addOption(datasetName);

        Option directory = new Option("dir", "directory", true, "Directory");
        directory.setRequired(true);
        options.addOption(directory);

        Option toolToTest = new Option("t", "tool", true, "Tool To test (chemspot, chener, chemde)");
        options.addOption(toolToTest);

        CommandLineParser parser = new BasicParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd;

        try {
            cmd = parser.parse(options, args);
            arguments.put("dataset",cmd.getOptionValue("dataset"));
            arguments.put("directory",cmd.getOptionValue("directory"));
            arguments.put("tool",cmd.getOptionValue("tool"));
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("utility-name", options);
            System.exit(1);
        }
        return arguments;
    }
}
