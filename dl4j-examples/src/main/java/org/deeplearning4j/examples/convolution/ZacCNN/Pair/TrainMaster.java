package org.deeplearning4j.examples.convolution.ZacCNN.Pair;

import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Model.MDLModel;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Utils.Message;
import org.deeplearning4j.examples.convolution.ZacCNN.SplitListener;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class TrainMaster extends Thread {

    //    private boolean isDecay = true;
    private int id;

    private BlockingQueue<Message> getQueue = new LinkedBlockingQueue<>();

    private BlockingQueue<Message> sendQueue;

    private List<BlockingQueue<Message>> broadcast;

    private boolean isMaster = false;

    private Config settings;

    private int slaveNum;

    private int epoc;

    private Map<Integer, List<Double>> epocLoss = new TreeMap<>();

    private long bstart;

//    private INDArray w0 = null;

    private boolean isLinked = false;

    public boolean isEnd = false;

    private SplitListener splitListener;
    private String modelFile;

    private int syncInterval = 0;
    private int batchNum = 0;
    private int lastSync = 0;

    private List<Long> batchTime = new ArrayList<>();

    // for test
    private DataSetIterator testIterator = null;
    private List<String> resultList = new ArrayList<>();
    private List<Double> ac = new ArrayList<>();
    private List<Double> pr = new ArrayList<>();
    private List<Double> re = new ArrayList<>();
    private List<Double> f1 = new ArrayList<>();

    private List<Double> scales = new ArrayList<>();

    private MDLModel.Type modelType = MDLModel.Type.LENET1D;

    public void setModelType(MDLModel.Type modelType) {
        this.modelType = modelType;
    }


    public TrainMaster(BlockingQueue<Message> sendQueue, int id, Config settings, boolean isLinked, SplitListener splitListener, String modelFile) {
        this(sendQueue, id, settings, splitListener, modelFile);
        this.isLinked = isLinked;
    }

    public TrainMaster(BlockingQueue<Message> sendQueue, int id, Config settings, SplitListener splitListener, String modelFile) {
        this.sendQueue = sendQueue;
        this.id = id;
        this.settings = settings;

        this.splitListener = splitListener;
        this.modelFile = modelFile;

        this.batchNum = (int) Math.ceil(settings.getTaskNum() / (float) settings.getBatchSize());
        this.syncInterval = (int) Math.floor(batchNum / (float) SystemRun.policy.getSyncNum());
        this.lastSync = (batchNum / syncInterval) * syncInterval;
    }

    public TrainMaster(int id, Config settings, int slaveNum, boolean isLinked, SplitListener splitListener, String modelFile) {
        this(id, settings, slaveNum, splitListener, modelFile);
        this.isLinked = isLinked;
    }

    public TrainMaster(int id, Config settings, int slaveNum, SplitListener splitListener, String modelFile) {
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

    private int batchID;

    private TrainingListener listener = new TrainingListener() {

        @Override
        public void iterationDone(Model model, int iteration, int epoch) {
            epoc = epoch;
            batchID = iteration;
            if (isMaster) {
                MultiLayerNetwork network = (MultiLayerNetwork) model;
                long bend = System.currentTimeMillis();
                long time = bend - bstart;
                bstart = bend;
                System.out.println("master iteration done: " + iteration + " model score: " + model.score() + " epoch: " + epoch +
                                       " learning rate: " + network.getLearningRate(0) + " time: " + time);

                batchTime.add(time);
            }

            // get average loss value
            List<Double> list = epocLoss.get(epoch);
            if (list == null) {
                list = new ArrayList<>();
                epocLoss.put(epoch, list);
            }
            list.add(model.score());

            switch (SystemRun.policy) {
                case BATCH:
                    sync(model);
                    break;
                case HALF_EPOC:
                    if (checkSync(iteration)) {
                        sync(model);
                    }
                    break;
            }
        }

        private boolean checkSync(int batchID) {
            int index = batchID + 1;
            int gap = index % batchNum;
            if (lastSync == batchNum) {
                // no odd issue
                int smallGap = gap % syncInterval;
                if (smallGap == 0) {
                    return true;
                }
            } else {
                if (gap != lastSync) {
                    int smallGap = gap % syncInterval;
                    if (gap == 0 || smallGap == 0) {
                        return true;
                    }
                }
            }
            return false;
        }

        double l1start = 0;

        double l1end = 0;

        double l1gradient = 0;

        @Override
        public void onEpochStart(Model model) {
            bstart = System.currentTimeMillis();

            // get L1 start point
            l1start = model.params().norm1Number().doubleValue();
        }

        @Override
        public void onEpochEnd(Model model) {
            System.out.println("[-------------onEpochEnd----------------] batchID: " + batchID);

            // get L1 end point
            l1end = model.params().norm1Number().doubleValue();
            l1gradient = l1start - l1end;
            GradientInfo.append(id, l1start, l1end, l1gradient);

            if (SystemRun.policy == SystemRun.SyncPolicy.EPOC) {
                sync(model);
            }
        }

        @Override
        public void onForwardPass(Model model, List<INDArray> activations) {

        }

        @Override
        public void onForwardPass(Model model, Map<String, INDArray> activations) {

        }

        @Override
        public void onGradientCalculation(Model model) {

        }

        @Override
        public void onBackwardPass(Model model) {

        }

        private void sync(Model model) {
            System.out.println("[----------SYNC-----------] batchID: " + batchID);

            // average loss value
            List<Double> lossList = epocLoss.get(epoc);
            double loss = 0;
            for (Double d : lossList) {
                loss += d;
            }
            loss /= lossList.size();

            MultiLayerNetwork network = (MultiLayerNetwork) model;

            // each epoch end, then will do weight sync
            if (isMaster) {
                List<Message> msgList = new ArrayList<>();
                try {
                    int num = 0;
                    if (isLinked) {
                        if (slaveNum == 0) {
                            num = 0;
                        } else {
                            // will only get 1 result from sub node
                            num = 1;
                        }
                    } else {
                        num = slaveNum;
                    }
                    while (num > 0) {
                        System.out.println("Master is [waiting]... left: " + num);
                        msgList.add(getQueue.take());
                        num--;
                        System.out.println("Master is taking... left: " + num);
                    }
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                INDArray newP = model.params().dup();

                if (isLinked && slaveNum > 0) {
                    Message newMsg = msgList.get(0);

                    newMsg.num++;
                    newMsg.parameters.muli(newMsg.num - 1);
                    newP.addi(newMsg.parameters);
                    newP.divi(newMsg.num);
                    System.out.println("Master is divided by: [" + newMsg.num + "]");
                } else {
                    // 1.  average SGD
                    for (Message m : msgList) {
                        newP.addi(m.parameters);
                    }

                    // average, but learning rate will be smaller and smaller
                    newP.divi(msgList.size() + 1);

                    if (SystemRun.isScaleDecay) {
//                        && l1gradient > 0
                        // weight decay and get scale
//                        newP.muli(0.75);
                        newP.muli(getScale(l1start, l1end, msgList));
                    }
                }

                // update model
                // * setParam.assign will make a copy
                model.setParams(newP);

                // update done, and test for accuracy
                if (SystemRun.isTestRound) {
                    test((MultiLayerNetwork) model);
                }


                // fixed bug: if not send message back to slave, the memory will not be relesaed
                Message newMsg = new Message(id);
                newMsg.parameters = newP;
                int i = 0;
                for (BlockingQueue queue : broadcast) {
                    queue.offer(newMsg);
                    i++;
                    System.out.println("master sending to " + i);
                }
            } else {
                Message message = new Message(id);

                // if linked, need frist get message from sub node, then send to root
                if (isLinked && !isEnd) {
                    try {
                        System.out.println("node is waiting for sub node... thread: " + id);
                        Message newMsg = getQueue.take();
                        message.parameters = model.params().dup();

                        // add with sub node weights
                        newMsg.num++;
                        newMsg.parameters.muli(newMsg.num - 1);
                        message.parameters.addi(newMsg.parameters);
                        message.parameters.divi(newMsg.num);
                        message.num = newMsg.num;

                        System.out.println("divide by: [" + message.num + "] thread: " + id);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                } else {
                    message.parameters = model.params();
                    message.gradient = l1gradient;
                    message.num++;
                }

                message.id = id;

                System.out.println("Slave is sending... thread: " + id);
                sendQueue.offer(message);

                Message newMsg = null;
                try {
                    System.out.println("Slave is waiting for master... thread: " + id);
                    newMsg = getQueue.take();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("Slave get new P.... thread: " + id);

                INDArray newP = newMsg.parameters;

                model.setParams(newP);
            }
        }

        /**
         * for averaging scale
         *
         * @param start
         * @param end
         * @param list
         * @return
         */
        private double getScale(double start, double end, List<Message> list) {
            double base = end;
            double g = 0;
            for (Message m : list) {
                base += start - m.gradient;
                g += m.gradient;
            }
            base = base / (double) (list.size() + 1);
            g = g / (double) (list.size() + 1);
            double scale = 0;
//            if (g < 0) {
//                scale = base / start;
//            } else {
            scale = Math.abs((base - list.size() * g) / Math.max(base, list.size() * g));
//            }
            scales.add(scale);
            System.out.println("The scale is: " + scale + " Epoch: " + epoc);
            return scale;
        }
    };


    @Override
    public void run() {
        String output = "";

        long start = System.currentTimeMillis();

        // train ------------------
        HarReader reader = null;
        if (modelType == MDLModel.Type.TCN) {
            reader = new HarReader(settings.getNumLinesToSkip(), settings.getHeight(),
                settings.getWidth(), settings.getChannel(), settings.getNumClasses(), settings.getTaskNum(), settings.getDelimiter(), true);
        } else {
            reader = new HarReader(settings.getNumLinesToSkip(), settings.getHeight(),
                settings.getWidth(), settings.getChannel(), settings.getNumClasses(), settings.getTaskNum(), settings.getDelimiter());
        }

        try {
            reader.initialize(new FileSplit(settings.getFile()));
        } catch (Exception e) {
            e.printStackTrace();
        }

        DataSetIterator iterator = new RecordReaderDataSetIterator(reader, settings.getBatchSize(), settings.getLabelIndex(), settings.getNumClasses());

        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        if (settings.isNoraml()) {
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(iterator);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
            iterator.setPreProcessor(normalizer);
        }

        // model ready
        long process = System.currentTimeMillis();

        MultiLayerNetwork model = null;
        if (isMaster) {

            // evaluate ---------------
            // master test number will be current task number * (slave number + 1)
            HarReader testReader = null;
            if (modelType == MDLModel.Type.TCN) {
                testReader = new HarReader(settings.getNumLinesToSkip(), settings.getHeight(), settings.getWidth(), settings.getChannel(),
                    settings.getNumClasses(), settings.getTaskNum() * (slaveNum + 1), settings.getDelimiter(), true);
            } else {
                testReader = new HarReader(settings.getNumLinesToSkip(), settings.getHeight(), settings.getWidth(), settings.getChannel(),
                    settings.getNumClasses(), settings.getTaskNum() * (slaveNum + 1), settings.getDelimiter());
            }


            try {
                testReader.initialize(new FileSplit(new File(settings.getTestPath())));
            } catch (Exception e) {
                e.printStackTrace();
            }

            testIterator = new RecordReaderDataSetIterator(testReader, settings.getBatchSize(),
                settings.getLabelIndex(), settings.getNumClasses());

            if (settings.isNoraml()) {
                DataNormalization normalizer = new NormalizerStandardize();
                normalizer.fit(testIterator);
                testIterator.setPreProcessor(normalizer);
            }

            /**
             * switich model here----------------
             *
             */
            MultiLayerConfiguration conf = MDLModel.getNetwork(modelType, settings);

            // send conf to others
            Message message = new Message(id);
            message.confJosn = conf.toJson();

            model = new MultiLayerNetwork(conf);
            model.init();

            System.out.println(model.summary());

            // for test, save model
            try {
                Nd4j.saveBinary(model.params(), new File("/Users/zhangyu/Desktop/cache"));
            } catch (IOException e) {
                e.printStackTrace();
            }

            // send init to others
            message.parameters = model.params();

            if (broadcast != null) {
                int i = 0;
                for (BlockingQueue send : broadcast) {
                    send.offer(message);
                    i++;
                    System.out.println("send model init to " + i);
                }
            }
        } else {
            try {
                // read from master
                Message message = getQueue.take();
                System.out.println("Thread " + id + " init model.....");
                MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(message.confJosn);
                model = new MultiLayerNetwork(conf);

                // not use copy from master
                model.init();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        System.out.println("Total num of params: " + model.numParams());
        model.setListeners(listener);
        model.fit(iterator, settings.getEpoch());

        long end = System.currentTimeMillis();

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
            String log3 = "Preprocess Total time: " + (process - start) / 1000;
            System.out.println(log3);
            String log4 = "Train Total time: " + (end - process) / 1000;
            System.out.println(log4);
            output += log1 + "\n" + log2 + "\n" + log3 + "\n" + log4 + "\n";

            // batch average time
            int averageTime = 0;
            for (int i = 0; i < batchTime.size(); i++) {
                averageTime += batchTime.get(i);
            }
            averageTime /= (float) batchTime.size();
            String log5 = "Average batch time: " + averageTime;
            output += log5 + "\n";
            System.out.println(log5);

            // each epoc average
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
            output += "\nloss = " + array + "\n";
            System.out.println(array);

            // test if train all done
            if (!SystemRun.isTestRound) {
                test(model);
            }

            String m1 = "\nac = " + ac + "\n";
            output += m1;
            System.out.println(m1);

            String m2 = "\npr = " + pr + "\n";
            output += m2;
            System.out.println(m2);

            String m3 = "\nre = " + re + "\n";
            output += m3;
            System.out.println(m3);

            String m4 = "\nf1 = " + f1 + "\n";
            output += m4;
            System.out.println(m4);

            // show weight gradient changes
            String result = GradientInfo.printGradient();
            output += result;
            System.out.println(result);

            // add scales
            String scaleList = "scale = " + scales.toString() + "\n\n";
            output += scaleList;
            System.out.println(scaleList);

            // all test results
            for (String out : resultList) {
                output += out;
            }

            if (splitListener != null) {
                splitListener.trainDone(output);
            }
        }

        // release memory
        model.clear();
    }


    public void test(MultiLayerNetwork model) {
        Evaluation eval = model.evaluate(testIterator);
        String result = "\nEpoc ID: " + epoc + "\n";
        result += eval.stats() + "\n\n";
        resultList.add(result);

        ac.add(eval.accuracy());
        pr.add(eval.precision());
        re.add(eval.recall());
        f1.add(eval.f1());
        System.out.println(result);
    }

}
