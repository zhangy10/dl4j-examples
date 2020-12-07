package org.deeplearning4j.examples.convolution.ZacCNN.Pair.Go.base;

import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Config;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.GradientInfo;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.HarReader;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Model.MDLModel;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.SystemRun;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Utils.Message;
import org.deeplearning4j.examples.convolution.ZacCNN.SplitListener;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public abstract class BaseTrain extends Thread {
    //    private boolean isDecay = true;
    protected int id;

    protected BlockingQueue<Message> getQueue = new LinkedBlockingQueue<>();

    protected BlockingQueue<Message> masterQueue;

    protected List<BlockingQueue<Message>> broadcast;

    protected boolean isMaster = false;

    protected Config settings;

    protected int slaveNum;

    protected int avaliableNum = 0;

    protected int epoc = 0;
    protected int batchID;

    protected Map<Integer, List<Double>> epocLoss = new TreeMap<>();

    protected long bstart;

    protected SplitListener splitListener;
    protected String modelFile;

    protected List<Long> batchTime = new ArrayList<>();

    protected DataSetIterator trainIterator = null;
    protected DataSetIterator testIterator = null;

    protected List<String> resultList = new ArrayList<>();
    protected List<Double> ac = new ArrayList<>();
    protected List<Double> pr = new ArrayList<>();
    protected List<Double> re = new ArrayList<>();
    protected List<Double> f1 = new ArrayList<>();

    protected MDLModel.Type modelType = MDLModel.Type.LENET1D;
    protected String existingFile = null;
    protected Map<Integer, List<Double>> l1Gradients = new HashMap<>();

    public int getTaskID() {
        return id;
    }

    protected void appendGradient(int epoc, double l1) {
        List gradients = l1Gradients.get(epoc);
        if (gradients == null) {
            gradients = new ArrayList();
            l1Gradients.put(epoc, gradients);
        }
        gradients.add(l1);
    }

    protected String printAllGradient() {
        StringBuilder sb = new StringBuilder();
        sb.append("--------------------------------\nTg" + id + ":\n");
        Set<Map.Entry<Integer, List<Double>>> en = l1Gradients.entrySet();
        for (Map.Entry<Integer, List<Double>> line : en) {
            int epoc = line.getKey();
            sb.append("Tg" + id + "_" + epoc + " = " + line.getValue().toString() + ";\n\n");
        }
        return sb.toString();
    }

    public void setModelType(MDLModel.Type modelType) {
        this.modelType = modelType;
    }

    public void setExistingFile(String file) {
        this.existingFile = file;
    }


    public BaseTrain(BlockingQueue<Message> sendQueue, int id, Config settings, SplitListener splitListener, String modelFile) {
        this.masterQueue = sendQueue;
        this.id = id;
        this.settings = settings;

        this.splitListener = splitListener;
        this.modelFile = modelFile;
    }

    public BaseTrain(int id, Config settings, int slaveNum, SplitListener splitListener, String modelFile) {
        this(null, id, settings, splitListener, modelFile);
        this.isMaster = true;
        this.slaveNum = slaveNum;
        broadcast = new ArrayList<>();
    }

    public BlockingQueue getQueue() {
        return getQueue;
    }

    public void addSlave(BlockingQueue<Message> queue) {
        if (broadcast != null) {
            broadcast.add(queue);
        }
    }

    public void addAvaliable() {
        avaliableNum++;
    }

    public int getAvaliableNum() {
        return avaliableNum;
    }

    @Override
    public void run() {
        printInfo("Random for training: " + settings.getSeed());

        long start = System.currentTimeMillis();
        loadData();
        long process = System.currentTimeMillis();

        MultiLayerNetwork model = getModel();
        printInfo("Total num of params: " + model.numParams());
        model.fit(trainIterator, settings.getEpoch());

        long end = System.currentTimeMillis();

        printInfo("----------------------------------Train Done-------------------------------------\n\n");

        // time
        String pretime = "Preprocess Total time: " + (process - start) / 1000;
        printInfo(pretime);
        String time = "Train Total time: " + (end - process) / 1000;
        printInfo(time);

        // all log info
        String output = getLog(isMaster, model, pretime, time);

        // after train for log
        afterTrain(output, model);

        // release memory
        model.clear();
    }

    protected abstract void loadData();

    protected abstract MultiLayerNetwork getModel();

    protected abstract void afterTrain(String output, Model model);

    protected RecordReaderDataSetIterator loading(String file, int taskNum) {
        RecordReaderDataSetIterator it = null;

        HarReader reader = null;
        if (modelType == MDLModel.Type.TCN) {
            reader = new HarReader(settings.getNumLinesToSkip(), settings.getHeight(),
                settings.getWidth(), settings.getChannel(), settings.getNumClasses(), taskNum, settings.getDelimiter(), true);
        } else {
            reader = new HarReader(settings.getNumLinesToSkip(), settings.getHeight(),
                settings.getWidth(), settings.getChannel(), settings.getNumClasses(), taskNum, settings.getDelimiter());
        }

        try {
            reader.initialize(new FileSplit(new File(file)));
        } catch (Exception e) {
            e.printStackTrace();
        }

        it = new RecordReaderDataSetIterator(reader, settings.getBatchSize(), settings.getLabelIndex(), settings.getNumClasses());

        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        if (settings.isNoraml()) {
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(it);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
            it.setPreProcessor(normalizer);
        }

        return it;
    }

    public void printInfo(String log) {
        if (isMaster) {
            System.out.println("T_" + id + ": " + log);
        }
    }

    public void test(MultiLayerNetwork model) {
        // if iid case, only a node can test as file cannot be read by multi nodes
        if (isMaster || !SystemRun.isIID) {
            Evaluation eval = model.evaluate(testIterator);
            String result = "\nEpoc ID: " + epoc + "\n";
            result += eval.stats() + "\n\n";
            resultList.add(result);

            ac.add(eval.accuracy());
            pr.add(eval.precision());
            re.add(eval.recall());
            f1.add(eval.f1());

            printInfo(result);
        }
    }

    protected String getLog(boolean isMaster, MultiLayerNetwork model, String pretime, String time) {
        String output = "---------------------\n" + "[Thread]: " + id + "\n";

        // train done
        if (isMaster) {
            // model structure
            output += model.summary() + "\n";

            // result.....
            String log0 = "Save model....to " + modelFile + "\n";
            output += log0;
            System.out.println(log0);
            try {
                ModelSerializer.writeModel(model, modelFile, true);
            } catch (IOException e) {
                e.printStackTrace();
            }
            String log1 = "Model Total num of params: " + model.numParams();
            System.out.println(log1);
            String log2 = "Save model done!!";
            System.out.println(log2);
            output += log1 + "\n" + log2 + "\n";
        }

        output += pretime + "\n" + time + "\n";

        // batch average time
        int averageTime = 0;
        for (int i = 0; i < batchTime.size(); i++) {
            averageTime += batchTime.get(i);
        }
        averageTime /= (float) batchTime.size();
        String log5 = "Average batch time: " + averageTime;
        output += log5 + "\n";
        printInfo(log5);

        // each epoc average loss
        List<Double> averageList = new ArrayList<>();
        Iterator<Integer> it = epocLoss.keySet().iterator();
        while (it.hasNext()) {
            int index = it.next();
            List<Double> epocList = epocLoss.get(index);
            int size = epocList.size();
            double average = 0;
            for (int i = 0; i < size; i++) {
                average += epocList.get(i);
            }
            averageList.add(average / size);
        }

        String array = averageList.toString();
        array = "\nloss" + id + "  = " + array + "\n";
        output += array;
        printInfo(array);

        // test if train all done
        if (isMaster && !SystemRun.isTestRound) {
            test(model);
        }

        String m1 = "\nac" + id + " = " + ac + "\n";
        output += m1;
        printInfo(m1);

        String m2 = "\npr" + id + " = " + pr + "\n";
        output += m2;
        printInfo(m2);

        String m3 = "\nre" + id + "  = " + re + "\n";
        output += m3;
        printInfo(m3);

        String m4 = "\nf1" + id + "  = " + f1 + "\n";
        output += m4;
        printInfo(m4);

        if (isMaster) {
            // show all weight gradient changes
            String result = GradientInfo.printGradient();
            output += result;
            printInfo(result);

            // all test results
            for (String out : resultList) {
                output += out;
            }
        }

        return output;
    }
}
