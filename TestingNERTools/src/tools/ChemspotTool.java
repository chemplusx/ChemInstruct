package tools;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import de.berlin.hu.chemspot.ChemSpot;
import de.berlin.hu.chemspot.ChemSpotFactory;
import de.berlin.hu.chemspot.Mention;

public class ChemspotTool extends BaseTool {
    ChemSpot tagger;

    public ChemspotTool() {
        this.tagger = ChemSpotFactory.createChemSpot("D:\\workspace\\chemspot-2.0\\dict.zip", "D:\\workspace\\chemspot-2.0\\ids.zip", "D:\\workspace\\chemspot-2.0\\multiclass.bin");
//        this.initialiseNewOutputFile();
    }

    public List<String> ExtractEntities(String text) throws Exception {
        List<String> entitiesExtracted = new ArrayList<>();
        for (Mention mention : tagger.tag(text)) {
            // System.out.printf("%d\t%d\t%s\t%s\t%s,\t%s%n",
            //         mention.getStart(), mention.getEnd(), mention.getText(),
            //         mention.getCHID(), mention.getSource(), mention.getType().toString());
//                    this.writeToCsv(docID, mention.getStart(), mention.getEnd(), mention.getText(),
//                            mention.getCHID(), mention.getSource(), mention.getType().toString());

            entitiesExtracted.add(mention.getText().toLowerCase());
        }
        return entitiesExtracted;
    }


    public void initialiseNewOutputFile(){
        this.outputFilePath = "D:\\workspace\\TestingNERTools\\testing_data\\output\\chemspot\\" + (new Date()).getTime() +".csv";

        try {
            // Check if the file already exists
            Path path = Paths.get(this.outputFilePath);

            if (!Files.exists(path)) {
                // If the file does not exist, create it along with the necessary directories
                Files.createDirectories(path.getParent());
                Files.createFile(path);

                String[] header = {"DocName", "Start-End", "Entity", "ChemicalID", "Source", "Type"};

                try (FileWriter csvWriter = new FileWriter(this.outputFilePath)) {
                    // Write the header to the CSV file
                    csvWriter.append(String.join(",", header));
                    csvWriter.append("\n");

//                    System.out.println("Data has been written to the CSV file.");

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

    public void writeToCsv(String docname, Integer start, Integer end, String entity, String chID, String source, String type){

        try (FileWriter csvWriter = new FileWriter(this.outputFilePath, true)) {

            // Write data rows to the CSV file
            csvWriter.append(String.join(",", docname, start+"-"+end, entity.replaceAll("\\R", " "), chID, source, type));
            csvWriter.append("\n");

//            System.out.println("Data has been written to the CSV file.");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
