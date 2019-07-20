package org.deeplearning4j.examples.convolution.ZacCNN;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;

public class FileUtils {

    private static final String FILE_FORMAT = "UTF-8";
    private static int buffedSize = 1024;

    /**
     * Writer the output into a given file.
     *
     * @param text
     * @param path
     */
    public static void write(String text, String path) {
        BufferedWriter bw = null;
        try {
            bw = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(path, true), FILE_FORMAT), buffedSize);
            bw.write(text);
            bw.flush();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (bw != null) {
                try {
                    bw.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

}
