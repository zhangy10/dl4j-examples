package org.deeplearning4j.examples.convolution.ZacCNN.Pair.Go;

public class CommCount {
    private int id;
    private int sendCount = 0;
    private int getCount = 0;

    public CommCount(int id) {
        this.id = id;
    }

    public void addSend() {
        sendCount++;
    }

    public void addGet() {
        getCount++;
    }

    public String toString() {
        return "Comm: T_" + id + ", Send count: " + sendCount + ", Get count: " + getCount + "\n";
    }
}
