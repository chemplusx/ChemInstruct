package tools;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import chener.Tagger;

public class CheNERTool extends BaseTool {
    Tagger tagger;
    String outputFilePath;

    public CheNERTool() {
        this.tagger =  new Tagger();
//        this.initialiseNewOutputFile();
    }


    public List<String> ExtractEntities(String text) {
        List<String> entitiesExtracted = new ArrayList<>();
        this.tagger.doTheTagging(text, "annotatedText.txt", 1, false);
        for(String en: this.tagger.getListEntities()){
            String entity = en.split("##")[0];
            entitiesExtracted.add(entity);
        }
        return entitiesExtracted;
    }


    public void initialiseNewOutputFile(){
        this.outputFilePath = "D:\\workspace\\TestingNERTools\\testing_data\\output\\cheNER\\" + (new Date()).getTime() +".csv";

        try {
            // Check if the file already exists
            Path path = Paths.get(this.outputFilePath);

            if (!Files.exists(path)) {
                // If the file does not exist, create it along with the necessary directories
                Files.createDirectories(path.getParent());
                Files.createFile(path);

                String[] header = {"DocName", "Start-End", "Entity"};

                try (FileWriter csvWriter = new FileWriter(this.outputFilePath)) {
                    // Write the header to the CSV file
                    csvWriter.append(String.join(",", header));
                    csvWriter.append("\n");

                    System.out.println("Data has been written to the CSV file.");

                } catch (IOException e) {
                    e.printStackTrace();
                }

                System.out.println("File created successfully.");
            } else {
                System.out.println("File already exists.");
            }

        } catch (FileAlreadyExistsException e) {
            System.out.println("File already exists.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void writeToCsv(String docname, Integer start, Integer end, String entity){

        try (FileWriter csvWriter = new FileWriter(this.outputFilePath, true)) {

            // Write data rows to the CSV file
            csvWriter.append(String.join(",", docname, start+"-"+end, entity.replaceAll("\\R", " ")));
            csvWriter.append("\n");

            System.out.println("Data has been written to the CSV file.");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
