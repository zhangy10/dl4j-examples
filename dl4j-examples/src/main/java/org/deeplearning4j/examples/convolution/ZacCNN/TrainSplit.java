package org.deeplearning4j.examples.convolution.ZacCNN;

import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.ROCMultiClass;
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

public class TrainSplit extends Thread {

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

//    private INDArray w0 = null;

    private boolean isLinked = false;

    public boolean isEnd = false;

    private SplitListener splitListener;
    private String modelFile;

    private int syncInterval = 0;
    private int batchNum = 0;
    private int lastSync = 0;


    public TrainSplit(BlockingQueue<Msg> sendQueue, int id, Config settings, boolean isLinked, SplitListener splitListener, String modelFile) {
        this(sendQueue, id, settings, splitListener, modelFile);
        this.isLinked = isLinked;
    }

    public TrainSplit(BlockingQueue<Msg> sendQueue, int id, Config settings, SplitListener splitListener, String modelFile) {
        this.sendQueue = sendQueue;
        this.id = id;
        this.settings = settings;

        this.splitListener = splitListener;
        this.modelFile = modelFile;

        this.batchNum = (int) Math.ceil(settings.getTaskNum() / (float) settings.getBatchSize());
        this.syncInterval = (int) Math.floor(batchNum / (float) SystemRun.policy.getSyncNum());
        this.lastSync = (batchNum / syncInterval) * syncInterval;
    }

    public TrainSplit(int id, Config settings, int slaveNum, boolean isLinked, SplitListener splitListener, String modelFile) {
        this(id, settings, slaveNum, splitListener, modelFile);
        this.isLinked = isLinked;
    }

    public TrainSplit(int id, Config settings, int slaveNum, SplitListener splitListener, String modelFile) {
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

        @Override
        public void onEpochStart(Model model) {
            bstart = System.currentTimeMillis();
        }

        @Override
        public void onEpochEnd(Model model) {
            System.out.println("[-------------onEpochEnd----------------] batchID: " + batchID);
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

                    newMsg.num++;
                    newMsg.parameters.muli(newMsg.num - 1);
                    newP.addi(newMsg.parameters);
                    newP.divi(newMsg.num);
                    System.out.println("Master is divided by: [" + newMsg.num + "]");
                } else {
                    // 1.  average SGD
                    for (Msg m : msgList) {
                        newP.addi(m.parameters);
                    }

                    // average, but learning rate will be smaller and smaller
                    newP.divi(msgList.size() + 1);
                }

//                    // 2.  sum by different weight
//                    Msg masterMsg = new Msg();
//                    masterMsg.loss = model.score();
//                    masterMsg.parameters = model.params().dup();
//                    msgList.add(masterMsg);
//
//                    double all = 0;
//                    for (Msg m : msgList) {
//                        all += m.loss;
//                    }
//
//                    for (Msg m : msgList) {
//                        m.w = (all - m.loss) / ((msgList.size() - 1) * all);
//                    }
//
//                    INDArray newP = msgList.get(0).parameters.mul(msgList.get(0).w);
//                    for (int i = 1; i < msgList.size(); i++) {
//                        newP.addi(msgList.get(i).parameters.mul(msgList.get(i).w));
//                    }


                // 4. adam update
//                    INDArray init = Nd4j.zeros(1, model.params().shape()[1] * 2);
//                    GradientUpdater u = new Adam(network.getLearningRate(0)).instantiate(init, false);
//                    INDArray newP = null;
//                    for (int i = 0; i < msgList.size(); i++) {
//                        // Aw here is required
//                        Msg m = msgList.get(i);
//                        INDArray Aw = w0.sub(m.parameters);
//                        u.applyUpdater(Aw, i, 0);
//                        if (i == msgList.size() - 1) {
//                            newP = Aw;
//                        }
//                    }
//                    newP = w0.sub(newP);


                // 3.  update learning rate
//                    Double l = network.getLearningRate(0);
//                    l *= msgList.size() + 1;
//                    network.setLearningRate(l);

                // update model
                // * setParam.assign will make a copy
                model.setParams(newP);
//                w0 = model.params().dup();

                // if last round, will not send the update to slaves

                // TODO bug: if not send message back to slave, the memory will not be relesaed
//                if (epoc != settings.getEpoch() - 1) {
                Msg newMsg = new Msg();
                newMsg.parameters = newP;
//                        newMsg.l = l;
                int i = 0;
                for (BlockingQueue queue : broadcast) {
                    queue.offer(newMsg);
                    i++;
                    System.out.println("master sending to " + i);
                }
//                }
            } else {
                Msg msg = new Msg();

                // for test
//                if (epoc == settings.getEpoch() - 1) {
//                    System.out.println("-----------------------");
//                }

                // if linked, need frist get message from sub node, then send to root
                if (isLinked && !isEnd) {
                    try {
                        System.out.println("node is waiting for sub node... thread: " + id);
                        Msg newMsg = getQueue.take();
                        msg.parameters = model.params().dup();

                        // add with sub node weights
                        newMsg.num++;
                        newMsg.parameters.muli(newMsg.num - 1);
                        msg.parameters.addi(newMsg.parameters);
                        msg.parameters.divi(newMsg.num);
                        msg.num = newMsg.num;

                        System.out.println("divide by: [" + msg.num + "] thread: " + id);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                } else {
                    msg.parameters = model.params();
                    msg.num++;
                }

//                msg.loss = model.score();
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
//                double l = newMsg.l;
//
//                network.setLearningRate(l);
                model.setParams(newP);
            }
        }
    };


    @Override
    public void run() {
        String output = "";

        long start = System.currentTimeMillis();

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

        long process = System.currentTimeMillis();

        MultiLayerNetwork model = null;
        if (isMaster) {

            /**
             * switich model here----------------
             *
             */
            MultiLayerConfiguration conf = lenet(settings);
//            MultiLayerConfiguration conf = alexnet(settings);

            // send conf to others
            Msg message = new Msg();
            message.confJosn = conf.toJson();

            model = new MultiLayerNetwork(conf);
            model.init();

            // for test, save model
//            try {
//                Nd4j.saveBinary(model.params(), new File("/Users/zhangyu/Desktop/test/cache"));
//            } catch (IOException e) {
//                e.printStackTrace();
//            }

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
                model.init(message.parameters, true);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        System.out.println("Total num of params: " + model.numParams());
        model.setListeners(listener);
        model.fit(iterator, settings.getEpoch());

        long end = System.currentTimeMillis();

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
            output += array + "\n";
            System.out.println(array);


            // evaluate ---------------
            File testFile = new File(settings.getTestPath());

            // master test number will be current task number * (slave number + 1)
            HarReader testReader = new HarReader(settings.getNumLinesToSkip(), settings.getHeight(), settings.getWidth(), settings.getChannel(),
                settings.getNumClasses(), settings.getTaskNum() * (slaveNum + 1), settings.getDelimiter());

            try {
                testReader.initialize(new FileSplit(testFile));
            } catch (Exception e) {
                e.printStackTrace();
            }

            DataSetIterator testIterator = new RecordReaderDataSetIterator(testReader, settings.getBatchSize(),
                settings.getLabelIndex(), settings.getNumClasses());

            if (settings.isNoraml()) {
                DataNormalization normalizer = new NormalizerStandardize();
                normalizer.fit(testIterator);
                testIterator.setPreProcessor(normalizer);
            }

            Evaluation eval = model.evaluate(testIterator);
            String result = eval.stats();
            output += result + "\n";
            System.out.println(result);

            if (splitListener != null) {
                splitListener.trainDone(output);
            }
        }
        // release memory
        model.clear();
    }

    public class Msg {
        int id;
        INDArray parameters;
        String confJosn;
        int num = 0;

        //        double loss;
        double w;
    }

    public static class Pair {
        int start;
        int end;

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
                                                         .weightInit(WeightInit.DISTRIBUTION)
                                                         .dist(new NormalDistribution(0.0, 1.0)) //均值为0，方差为1.0的正态分布
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
            case LSTM:
                builder = builder.layer(4, lstm("l1", config.getF1_out(), config.getNonZeroBias(), config.getDropOut()))
                              .layer(5, full("f1", config.getF1_out(), config.getNonZeroBias(), config.getDropOut()))
                              .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                            .name("o1")
                                            .nOut(config.getNumClasses())
                                            .activation(Activation.SOFTMAX)
                                            .build());
//                              .layer(5, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
//                                            .name("ro1")
//                                            .nOut(config.getNumClasses()).build());

//                // for cnn to lstm + rnn
//                return builder.inputPreProcessor(4, InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType, null)).build();
                break;
        }
        return builder.backprop(true)
                   .build();
    }

    public MultiLayerConfiguration alexnet(Config config) {
        /**
         * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
         * and the imagenetExample code referenced.
         * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
         **/

        return new NeuralNetConfiguration.Builder()
                   .seed(config.getSeed())
                   .weightInit(WeightInit.DISTRIBUTION) //根据给定的分布采样参数
                   .dist(new NormalDistribution(0.0, 0.01)) //均值为0，方差为0.01的正态分布
                   .activation(Activation.RELU)
                   .updater(new Adam(0.001))
//                                           .updater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 1e-2, 0.1, 100000), 0.9))
//                                           .biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 2e-2, 0.1, 100000), 0.9))
                   .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                   //采用除以梯度2范数来规范化梯度防止梯度消失或突变
                   .l2(5 * 1e-4)
                   .list() //13层的网络,第1,3层构建了alexnet计算层，目的是对当前输出的结果做平滑处理，
                   // 参数有相邻核映射数n=5,规范化常亮k=2,指数常量beta=0.75，系数常量alpha=1e-4
                   .layer(0, convNet("c1", config.getChannel(), config.getC1_out(), new int[]{1, config.getKernal()}, new int[]{1, 1}, new int[]{0, 0}, 0))
                   .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
                   .layer(2, maxpooling("m1", new int[]{1, config.getPooling()}, new int[]{1, config.getPooling()}))
                   .layer(3, convNet("c2", -1, config.getC2_out(), new int[]{1, config.getKernal()}, new int[]{1, 1}, new int[]{0, 0}, config.getNonZeroBias()))
                   .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
                   .layer(5, maxpooling("m2", new int[]{1, config.getPooling()}, new int[]{1, config.getPooling()}))
                   .layer(6, convNet("c3", -1, 90, new int[]{1, 32}, new int[]{1, 1}, new int[]{0, 0}, 0))
                   .layer(7, convNet("c4", -1, 90, new int[]{1, 32}, new int[]{1, 1}, new int[]{0, 0}, config.getNonZeroBias()))
                   .layer(8, convNet("c5", -1, 72, new int[]{1, 32}, new int[]{1, 1}, new int[]{0, 0}, config.getNonZeroBias()))
                   .layer(9, maxpooling("m3", new int[]{1, config.getPooling()}, new int[]{1, config.getPooling()}))
                   .layer(10, full("f1", config.getF1_out(), config.getNonZeroBias(), config.getDropOut()))
                   .layer(11, full("f2", config.getF1_out(), config.getNonZeroBias(), config.getDropOut()))
                   .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                  .name("o1")
                                  .nOut(config.getNumClasses())
                                  .activation(Activation.SOFTMAX)
                                  .build())
                   .setInputType(InputType.convolutionalFlat(config.getHeight(), config.getWidth(), config.getChannel())) // InputType.convolutional for normal image
                   .build();
    }

}
