package org.deeplearning4j.examples.convolution.ZacCNN.DistriTest;

import org.xerial.snappy.Snappy;

import java.io.*;

public class CompressTest {


    public static void main (String[] args) throws Exception {
        String compressed = "/Users/zhangyu/Desktop/compress";
        String data = "/Users/zhangyu/Desktop/test";

        int len = 0;
        byte[] buffer = new byte[1024];

        int oldAll = 0;
        int newAll = 0;

        BufferedInputStream fin = new BufferedInputStream(new FileInputStream(new File(data)));
        BufferedOutputStream fout = new BufferedOutputStream(new FileOutputStream(new File(compressed)));

        while((len = fin.read(buffer)) != -1) {
            byte[] compress = compress(buffer);
            oldAll += buffer.length;
            newAll += compress.length;
            System.out.println("old: " + buffer.length + " new: " + compress.length);

            fout.write(compress);
        }
        fout.flush();

        fin.close();
        fout.close();

        System.out.println("old all: " + oldAll + " new all: " + newAll);
    }


    public static byte[] compress(byte srcBytes[]) throws IOException {
        return  Snappy.compress(srcBytes);
    }

    public static byte[] uncompress(byte[] bytes) throws IOException {
        return Snappy.uncompress(bytes);
    }
}
