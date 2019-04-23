package org.deeplearning4j.examples.convolution.ZacCNN.DistriTest;

import java.util.Random;

public class JavaTest {


    public static void main (String[] args) {

//        byte[] og = new byte[15];
//        int len = 10;
//        new Random().nextBytes(og);
//        System.out.println(og);
//
//        SplitByte sb = sub(og, len);
//        System.out.println(sb.new_msg);
//        System.out.println(sb.left);

//        new String(null);



    }

    private static SplitByte sub(byte[] buffer, int len) {
        SplitByte sb = new SplitByte();
        sb.new_msg = new byte[len];
        sb.left = new byte[buffer.length - len];
        // need part
        System.arraycopy(buffer, 0, sb.new_msg, 0, len);
        // left part: empty or the head of next message
        System.arraycopy(buffer, len, sb.left, 0, sb.left.length);
        return sb;
    }



    private static class SplitByte {
        byte[] new_msg;
        byte[] left;
    }
}
