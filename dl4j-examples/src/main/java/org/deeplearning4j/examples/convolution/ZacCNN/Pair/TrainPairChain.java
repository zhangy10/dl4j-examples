package org.deeplearning4j.examples.convolution.ZacCNN.Pair;

import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.convolution.ZacCNN.SplitListener;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class TrainPairChain extends Thread {

    private boolean isPair = true;

    private int id;

    private BlockingQueue<Msg> getQueue = new LinkedBlockingQueue<>();

    private BlockingQueue<Msg> sendQueue;

    private List<BlockingQueue<Msg>> broadcast;

    private boolean isMaster = false;

    private Config settings;

    private int slaveNum;

    private int epoc;

    private Map<Integer, List<Double>> epocLoss = new TreeMap<>();

    private long bstart;


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


    public TrainPairChain(BlockingQueue<Msg> sendQueue, int id, Config settings, boolean isLinked, SplitListener splitListener, String modelFile) {
        this(sendQueue, id, settings, splitListener, modelFile);
        this.isLinked = isLinked;
    }

    public TrainPairChain(BlockingQueue<Msg> sendQueue, int id, Config settings, SplitListener splitListener, String modelFile) {
        this.sendQueue = sendQueue;
        this.id = id;
        this.settings = settings;

        this.splitListener = splitListener;
        this.modelFile = modelFile;

        this.batchNum = (int) Math.ceil(settings.getTaskNum() / (float) settings.getBatchSize());
        this.syncInterval = (int) Math.floor(batchNum / (float) SystemRun.policy.getSyncNum());
        this.lastSync = (batchNum / syncInterval) * syncInterval;
    }

    public TrainPairChain(int id, Config settings, int slaveNum, boolean isLinked, SplitListener splitListener, String modelFile) {
        this(id, settings, slaveNum, splitListener, modelFile);
        this.isLinked = isLinked;
    }

    public TrainPairChain(int id, Config settings, int slaveNum, SplitListener splitListener, String modelFile) {
        this(null, id, settings, splitListener, modelFile);
        this.isMaster = true;
        this.slaveNum = slaveNum;
        broadcast = new ArrayList<>();
    }

    public BlockingQueue getQueue() {
        return getQueue;
    }

    public void addSlave(BlockingQueue<Msg> queue) {
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

//                System.out.println("Last Time: " + network.getLastEtlTime());
//                try {
//                    Thread.sleep(1000);
//                } catch (InterruptedException e) {
//                    e.printStackTrace();
//                }
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
            double end = model.params().norm1Number().doubleValue();
            double l1gradient = l1start - end;
            GradientInfo.append(id, l1start, end, l1gradient);

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
                List<Msg> msgList = new ArrayList<>();
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

                // weight sync
//                    List<Double> lossSet = new ArrayList<>();
                INDArray newP = model.params().dup();

                if (isLinked && slaveNum > 0) {
                    Msg newMsg = msgList.get(0);

                    newP = pairSyncChannel(newMsg, model);

                    System.out.println("Master is done Pair Aggregation!!!");
                } else {
                    // 1.  average SGD
                    for (Msg m : msgList) {
                        newP.addi(m.parameters);
                    }

                    // average, but learning rate will be smaller and smaller
                    newP.divi(msgList.size() + 1);
                }

                // update model
                // * setParam.assign will make a copy
                model.setParams(newP);

                // update done, and test for accuracy
                if (SystemRun.isTestRound) {
                    test((MultiLayerNetwork) model);
                }

                Msg newMsg = new Msg();
                newMsg.parameters = newP;
                int i = 0;
                for (BlockingQueue queue : broadcast) {
                    queue.offer(newMsg);
                    i++;
                    System.out.println("master sending to " + i);
                }
            } else {
                Msg msg = new Msg();

                // if linked, need frist get message from sub node, then send to root
                if (isLinked && !isEnd) {
                    try {
                        System.out.println("node is waiting for sub node... thread: " + id);
                        Msg newMsg = getQueue.take();

                        msg.parameters = pairSyncChannel(newMsg, model);

                        msg.confJosn = model.conf().toJson();
                        msg.model = model;
                        System.out.println("Thread " + id + " is done Pair Aggregation!!!");
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                } else {
                    msg.parameters = model.params();
                    msg.confJosn = model.conf().toJson();
                    msg.model = model;
                }

                msg.id = id;

                System.out.println("Slave is sending... thread: " + id);
                sendQueue.offer(msg);

                Msg newMsg = null;
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


        private INDArray pairSyncChannel(Msg remote, Model model) {

            if (!isPair) {
                // /2 pair
                remote.parameters.addi(model.params());
                remote.parameters.divi(2);
                return remote.parameters;
            }

            MultiLayerNetwork remoteModel = (MultiLayerNetwork) remote.model;
            MultiLayerNetwork currentModel = (MultiLayerNetwork) model;

            for (int i = 0; i < currentModel.getLayers().length; i++) {
                Layer currentLayer = currentModel.getLayer(i);
                if (currentLayer.type() == Layer.Type.CONVOLUTIONAL || currentLayer.type() == Layer.Type.FEED_FORWARD) {
                    INDArray para = currentLayer.getParam("W");
                    INDArray remotePara = remoteModel.getLayer(i).getParam("W");

                    int[] normalShape = null;
                    int[] normalReshape = null;
                    if (currentLayer.type() == Layer.Type.CONVOLUTIONAL) {
                        normalShape = new int[]{1, 2, 3};
                        normalReshape = new int[]{-1, 1, 1, 1};
                    } else {
                        normalShape = new int[]{1};
                        normalReshape = new int[]{-1, 1};
                    }

                    INDArray paraNormal = para.norm1(normalShape);
                    INDArray remoteNormal = remotePara.norm1(normalShape);

                    INDArray pairNormal = paraNormal.add(remoteNormal);
                    paraNormal.divi(pairNormal);
                    remoteNormal.divi(pairNormal);

                    INDArray currentVector = para.mul(paraNormal.reshape(normalReshape));
                    remotePara.muli(remoteNormal.reshape(normalReshape));

                    remotePara.addi(currentVector);
                    System.out.println("Paire done " + id);
                }
            }

            return remoteModel.params();
        }

    };


    @Override
    public void run() {
        String output = "";

        long start = System.currentTimeMillis();

        // train ------------------
        HarReader reader = new HarReader(settings.getNumLinesToSkip(), settings.getHeight(),
            settings.getWidth(), settings.getChannel(), settings.getNumClasses(), settings.getTaskNum(), settings.getDelimiter());
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
            HarReader testReader = new HarReader(settings.getNumLinesToSkip(), settings.getHeight(), settings.getWidth(), settings.getChannel(),
                settings.getNumClasses(), settings.getTaskNum() * (slaveNum + 1), settings.getDelimiter());

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
            MultiLayerConfiguration conf = null;
            conf = lenet(settings);

            // send conf to others
            Msg message = new Msg();
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

            // last weight for sync use
//            w0 = model.params().dup();

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
                Msg message = getQueue.take();
                System.out.println("Thread " + id + " init model.....");
                MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(message.confJosn);
                model = new MultiLayerNetwork(conf);

                // clone will be false in real case
//                model.init(message.parameters, true);

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

            // result.....
            String log0 = "Save model....to " + modelFile;
            output += log0;
            System.out.println(log0);
//            String path = "/Users/zhangyu/Desktop/multi_model.bin";
            try {
//                ModelSerializer.writeModel(model, path, true);
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

    public class Msg {
        public int id;
        public INDArray parameters;
        public String confJosn;

        public Model model;
//        public int num = 0;
        //        double loss;
//        public double w;
    }

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

    private ConvolutionLayer convNet(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        if (in == -1) {
            return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nOut(out).biasInit(bias).convolutionMode(ConvolutionMode.Same).build();
        }
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).convolutionMode(ConvolutionMode.Same).build();
    }

    private SubsamplingLayer maxpooling(String name, int[] kernel, int[] stride) {
        return new SubsamplingLayer.Builder(kernel, stride).name(name).build();
    }

    private DenseLayer full(String name, int out, double bias, double dropOut) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(new NormalDistribution(0, 1)).build();
    }

    private LSTM lstm(String name, int out, double bias, double dropOut) {
        return new LSTM.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(new NormalDistribution(0, 1))
                   .activation(Activation.TANH).build();
    }

    public MultiLayerConfiguration lenet(Config config) {
        InputType inputType = InputType.convolutional(config.getHeight(), config.getWidth(), config.getChannel());

        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                                                         .seed(config.getSeed())
//                                           .weightInit(WeightInit.NORMAL) //根据给定的分布采样参数
//                                                         .weightInit(WeightInit.DISTRIBUTION)

                                                         .weightInit(new NormalDistribution(0.0, 1.0))

//                                                         .weightInit(new WeightInitNormal())

//                                                         .dist(new NormalDistribution(0.0, 1.0)) //均值为0，方差为1.0的正态分布
                                                         .activation(Activation.RELU)
                                                         .updater(new Adam(config.getLearnRate()))
                                                         // Adam is better
//                   .updater(new Nadam(learnRate))
                                                         // increase by 1% over 0.001
//                   .updater(new Adam(new InverseSchedule(ScheduleType.EPOCH, learnRate, gamma, 1)))
//                .biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 2e-2, 0.1, 100000), 0.9))

                                                         .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                                                         //采用除以梯度2范数来规范化梯度防止梯度消失或突变
                                                         .l2(5 * 1e-4)
                                                         .list() //13层的网络,第1,3层构建了alexnet计算层，目的是对当前输出的结果做平滑处理，参数有相邻核映射数n=5,规范化常亮k=2,指数常量beta=0.75，系数常量alpha=1e-4
                                                         .layer(0, convNet("c1", config.getChannel(), config.getC1_out(), new int[]{1, config.getKernal()}, new int[]{1, 1}, new int[]{0, 0}, 0))
                                                         // update padding issue
//                                           .layer(0, convNet("c1", channels, 36, new int[]{1, 64}, new int[]{1, 1}, new int[]{0, 32}, 0))
                                                         .layer(1, maxpooling("m1", new int[]{1, config.getPooling()}, new int[]{1, config.getPooling()}))
                                                         .layer(2, convNet("c2", -1, config.getC2_out(), new int[]{1, config.getKernal()}, new int[]{1, 1}, new int[]{0, 0}, config.getNonZeroBias()))
//                                           .layer(2, convNet("c2", -1, 72, new int[]{1, 64}, new int[]{1, 1}, new int[]{0, 16}, nonZeroBias))
                                                         .layer(3, maxpooling("m2", new int[]{1, config.getPooling()}, new int[]{1, config.getPooling()}))
                                                         .setInputType(inputType);

        switch (SystemRun.layerConfig) {
            case ONE:
                builder = builder.layer(4, full("f1", config.getF1_out(), config.getNonZeroBias(), config.getDropOut()))
                              .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                            .name("o1")
                                            .nOut(config.getNumClasses())
                                            .activation(Activation.SOFTMAX)
                                            .build());
                break;
            case TWO:
                builder = builder.layer(4, full("f1", config.getF1_out(), config.getNonZeroBias(), config.getDropOut()))
                              .layer(5, full("f2", config.getF1_out(), config.getNonZeroBias(), config.getDropOut()))
                              .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                            .name("o1")
                                            .nOut(config.getNumClasses())
                                            .activation(Activation.SOFTMAX)
                                            .build());
                break;
            case THREE:
                builder = builder.layer(4, full("f1", config.getF1_out(), config.getNonZeroBias(), config.getDropOut()))
                              .layer(5, full("f2", config.getF1_out(), config.getNonZeroBias(), config.getDropOut()))
                              .layer(6, full("f3", config.getF1_out(), config.getNonZeroBias(), config.getDropOut()))
                              .layer(7, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                            .name("o1")
                                            .nOut(config.getNumClasses())
                                            .activation(Activation.SOFTMAX)
                                            .build());
                break;
//            case GrowLSTM:
//                builder = builder.layer(4, lstm("l1", config.getF1_out(), config.getNonZeroBias(), config.getDropOut()))
//                              .layer(5, full("f1", config.getF1_out(), config.getNonZeroBias(), config.getDropOut()))
//                              .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
//                                            .name("o1")
//                                            .nOut(config.getNumClasses())
//                                            .activation(Activation.SOFTMAX)
//                                            .build());
            case LSTM:
                builder = builder.layer(4, lstm("l1", config.getF1_out(), config.getNonZeroBias(), config.getDropOut()))
//                              .layer(5, full("f1", config.getF1_out(), config.getNonZeroBias(), config.getDropOut()))
                              .layer(5, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                                            .lossFunction(LossFunctions.LossFunction.MCXENT).nOut(config.getNumClasses()).name("o1").build());

//                              .layer(5, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
//                                            .name("ro1")
//                                            .nOut(config.getNumClasses()).build());

//                // for cnn to lstm + rnn
//                return builder.inputPreProcessor(4, InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType, null)).build();
                break;
        }
        return builder
                   .build();
    }
}


