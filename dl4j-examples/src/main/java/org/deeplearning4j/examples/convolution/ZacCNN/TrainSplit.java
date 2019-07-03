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
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
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


    public TrainSplit(BlockingQueue<Msg> sendQueue, int id, Config settings, boolean isLinked) {
        this(sendQueue, id, settings);
        this.isLinked = isLinked;
    }

    public TrainSplit(BlockingQueue<Msg> sendQueue, int id, Config settings) {
        this.sendQueue = sendQueue;
        this.id = id;
        this.settings = settings;
    }

    public TrainSplit(int id, Config settings, int slaveNum, boolean isLinked) {
        this(id, settings, slaveNum);
        this.isLinked = isLinked;
    }

    public TrainSplit(int id, Config settings, int slaveNum) {
        this(null, id, settings);
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


    private TrainingListener listener = new TrainingListener() {

        @Override
        public void iterationDone(Model model, int iteration, int epoch) {
            epoc = epoch;
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
            List<Double> list = epocLoss.get(epoch);
            if (list == null) {
                list = new ArrayList<>();
                epocLoss.put(epoch, list);
            }
            list.add(model.score());
        }

        @Override
        public void onEpochStart(Model model) {
            bstart = System.currentTimeMillis();
        }

        @Override
        public void onEpochEnd(Model model) {
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
                        num = 1;
                    } else {
                        num = slaveNum;
                    }
                    while (num > 0) {
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

                if (isLinked) {
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
                if (epoc != settings.getEpoch() - 1) {
                    Msg newMsg = new Msg();
                    newMsg.parameters = newP;
//                        newMsg.l = l;
                    int i = 0;
                    for (BlockingQueue queue : broadcast) {
                        queue.offer(newMsg);
                        i++;
                        System.out.println("master sending to " + i);
                    }
                }
            } else {
                Msg msg = new Msg();

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
    };


    @Override
    public void run() {
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

//            MultiLayerConfiguration conf = lenet(settings);
            MultiLayerConfiguration conf = alexnet(settings);

            // send conf to others
            Msg message = new Msg();
            message.confJosn = conf.toJson();

            model = new MultiLayerNetwork(conf);
            model.init();
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

            // evaluate
            File testFile = new File(settings.getTestPath());

            HarReader testReader = new HarReader(settings.getNumLinesToSkip(), settings.getHeight(), settings.getWidth(), settings.getChannel(),
                settings.getNumClasses(), settings.getTaskNum(), settings.getDelimiter());

            try {
                testReader.initialize(new FileSplit(testFile));
            } catch (Exception e) {
                e.printStackTrace();
            }

            DataSetIterator testIterator = new RecordReaderDataSetIterator(reader, settings.getBatchSize(),
                settings.getLabelIndex(), settings.getNumClasses());

            if (settings.isNoraml()) {
                DataNormalization normalizer = new NormalizerStandardize();
                normalizer.fit(testIterator);
                testIterator.setPreProcessor(normalizer);
            }

            Evaluation eval = model.evaluate(iterator);
            System.out.println(eval.stats());

            // result.....

            System.out.println("Save model....");
            String basePath = "/Users/zhangyu/Desktop/";
            try {
                ModelSerializer.writeModel(model, basePath + "multi_model.bin", true);
            } catch (IOException e) {
                e.printStackTrace();
            }

            System.out.println("Model Total num of params: " + model.numParams());
            System.out.println("Save model done!!");
            System.out.println("Preprocess Total time: " + (process - start) / 1000);
            System.out.println("Train Total time: " + (end - process) / 1000);

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

            System.out.println(averageList);
        }
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

    public static void main(String[] args) {
        DataType type = DataType.HAR;
        int task = 1;

        // split task
        List<Pair> list = getTask(task, DataSet.getConfig(type).getTaskNum());
        System.out.println(list.toString());
        // assgin task
        TrainSplit master = null;
        List<TrainSplit> slaveList = new ArrayList<>();
        for (int i = 0; i < task; i++) {
            Pair fragment = list.get(i);
            Config config = DataSet.getNewConfig(type);
            if (i == 0) {
                master = new TrainSplit(i, config.setTaskRange(fragment.start, fragment.end), task - 1);
            } else {
                TrainSplit slave = new TrainSplit(master.getQueue(), i, config.setTaskRange(fragment.start, fragment.end));
                master.addSlave(slave.getQueue());
                slaveList.add(slave);
            }
        }
        master.start();
        for (TrainSplit t : slaveList) {
            t.start();
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


    public MultiLayerConfiguration lenet(Config config) {
        return new NeuralNetConfiguration.Builder()
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
                   .layer(4, full("f1", config.getF1_out(), config.getNonZeroBias(), config.getDropOut()))
                   .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                 .name("o1")
                                 .nOut(config.getNumClasses())
                                 .activation(Activation.SOFTMAX)
                                 .build())
                   .backprop(true)
                   .setInputType(InputType.convolutional(config.getHeight(), config.getWidth(), config.getChannel()))
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
