import org.apache.pdfbox.Loader;
import org.apache.pdfbox.io.RandomAccessReadBufferedFile;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class ConvertPDFToDataset {

    public static void main(String[] args){

    }

    public static void StartEvaluation(String directoryPath,String filename){

        try {
            if(filename.isBlank()){
                System.out.println("Filename not passed, reading from directory: "+ directoryPath);

                List<String> fileNames = GetFileNamesFromDirectory(directoryPath);

                for(String fileName : fileNames){
                    String textualContent = ReadPdfFileContent(directoryPath+fileName);
                    textualContent = SanitizeForRelevantText(textualContent);
                    // System.out.println(textualContent);
                    int starti = textualContent.indexOf("https://doi.org");
                    int endi = textualContent.indexOf("\n", starti);
                    String docId = textualContent.substring(starti, endi).replace("doi.org", "pubs.acs.org/doi").replace("\r", "");
                }
            }
        }catch(Exception ex){
            System.out.println("Error occurred: "+ ex);
        }
    }

    public static List<String> GetFileNamesFromDirectory(String directoryPath) throws IOException {
        List<String> fileNames = new ArrayList<String>();
        Path dirPath = Paths.get(directoryPath);

        // Check if the path is a directory
        if (Files.isDirectory(dirPath)) {
            // Use a try-with-resources statement to automatically close the DirectoryStream
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(dirPath)) {
                // Iterate over the files in the directory
                for (Path file : stream) {
                    // Print each file name
                    fileNames.add(file.getFileName().toString());
                    // System.out.println(file.getFileName());
                }
            }
        } else {
            System.out.println("The specified path is not a directory.");
        }
        return fileNames;
    }



    public static String ReadPdfFileContent(String fileName) {

        PDDocument document;
        try {
            document = Loader.loadPDF(new RandomAccessReadBufferedFile(fileName));

            // document = PDDocument()(fileName);

            if (!document.isEncrypted()) {
                PDFTextStripper stripper = new PDFTextStripper();
                String text = stripper.getText(document);
                // System.out.println("Text:" + text);
                return text;
            }
            document.close();
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return "";
    }

    public static String SanitizeForRelevantText(String textToSanitize) {
        int index = textToSanitize.indexOf("ASSOCIATED CONTENT");
        if(index == -1 ){
            index = textToSanitize.indexOf("REFERENCES");
        }
        return textToSanitize.substring(0, index).replace("■", "");
    }
}
