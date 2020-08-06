package org.deeplearning4j.examples.convolution.ZacCNN.Pair.Utils;

import java.util.ArrayList;
import java.util.List;

public class TaskSplit {

    public static class Pair {
        public int start;
        public int end;

        public Pair(int start, int end) {
            this.start = start;
            this.end = end;
        }

        @Override
        public String toString() {
            return "[" + start + "/" + end + "]";
        }
    }


    public static List<Pair> getTask(int taskNum, int total) {
        int doNum = total / taskNum;
        int rest = total % taskNum;
        int lastNum = doNum + rest;

        List<Pair> list = new ArrayList<>();
        int start = 0;
        for (int i = 0; i < taskNum; i++) {
            if (i == taskNum - 1) {
                list.add(new Pair(start, lastNum));
            } else {
                list.add(new Pair(start, doNum));
            }
            start += doNum;
        }
        return list;
    }

}
