package org.deeplearning4j.examples.convolution.ZacCNN.DistriTest;

import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.convolution.ZacCNN.HarReader;
import org.deeplearning4j.examples.dataexamples.CSVExample;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class DualTask extends Thread {

    private static Logger log = LoggerFactory.getLogger(CSVExample.class);

    private Settings settings;

    private BlockingQueue<Msg> queue = new LinkedBlockingQueue<>();

    private BlockingQueue<Msg> sendQueue;

    private boolean isMaster = false;

    private INDArray w0 = null;

    public static void main(String[] args) throws Exception {

        // split task to 2 thread test
        int totalTask = 10;

        DualTask master = new DualTask(new Settings(0, 5), true);
        DualTask slave = new DualTask(new Settings(5, 5));

        master.setSendQueue(slave.getQueue());
        slave.setSendQueue(master.getQueue());

        master.start();
        slave.start();


    }

    public BlockingQueue getQueue() {
        return queue;
    }

    public void setSendQueue(BlockingQueue<Msg> sendQueue) {
        this.sendQueue = sendQueue;
    }


//    public MultiLayerConfiguration getModelConf(int classNum, int h, int w, int channel) {
//        long seed = 6;
//
//        return new NeuralNetConfiguration.Builder()
//                   .seed(seed)
//                   .activation(Activation.RELU)
//                   .weightInit(WeightInit.XAVIER)
//                   .updater(new Adam(0.1))
//                   .l2(1e-4)
//                   .list()
//
//                   .layer(0, new DenseLayer.Builder().nOut(6)
//                                 .build())
//                   .layer(1, new DenseLayer.Builder().nIn(6).nOut(9)
//                                 .build())
//                   .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
//                                 .activation(Activation.SOFTMAX)
//                                 .nIn(9).nOut(classNum).build())
//                   // will do flat matrix to 2d for full-connected layer
//                   .setInputType(InputType.convolutional(h, w, channel)) // image的长宽
//                   .backprop(true).pretrain(false)
//                   .build();
//    }


    private static ConvolutionLayer convNet(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        //卷积输入层，参数包括名字，过滤器数量，输出节点数，卷积核大小，步副大小，补充边框大小，偏差
        if (in == -1) {
            return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nOut(out).biasInit(bias).convolutionMode(ConvolutionMode.Same).build();
        }
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).convolutionMode(ConvolutionMode.Same).build();
    }

    private static SubsamplingLayer maxpooling(String name, int[] kernel, int[] stride) {
        return new SubsamplingLayer.Builder(kernel, stride).name(name).build();
    }

    private static DenseLayer full(String name, int out, double bias) {
        //全连接层，本例中输出4096个节点，偏差为1，随机丢弃比例50%，参数服从均值为0，方差为0.005的高斯分布
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dist(new NormalDistribution(0, 1)).build();
    }


    public static MultiLayerConfiguration getModelConf(int classNum, int h, int w, int channel) {
        long seed = 6;
        double nonZeroBias = 1; //偏差

        return new NeuralNetConfiguration.Builder()
                   .seed(seed)
                   .activation(Activation.RELU)
                   .weightInit(WeightInit.DISTRIBUTION)
                   .dist(new NormalDistribution(0.0, 1.0)) //均值为0，方差为1.0的正态分布
                   .updater(new Adam(0.1))
                   .l2(1e-4)
                   .list()
                   .layer(0, convNet("c1", channel, 6, new int[]{1, 2}, new int[]{1, 1}, new int[]{0, 0}, 0))
                   // update padding issue
//                                           .layer(0, convNet("c1", channels, 36, new int[]{1, 64}, new int[]{1, 1}, new int[]{0, 32}, 0))
                   .layer(1, maxpooling("m1", new int[]{1, 2}, new int[]{1, 2}))
                   .layer(2, full("f1", 20, nonZeroBias))
//                                           .layer(3, new DenseLayer.Builder().nIn(6).nOut(9)
//                                                         .build())
                   .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).name("o1")
                                 .activation(Activation.SOFTMAX)
                                 .nOut(classNum).build())
                   // will do flat matrix to 2d for full-connected layer
                   .setInputType(InputType.convolutional(h, w, channel)) // image的长宽
                   .backprop(true).pretrain(false)
                   .build();
    }


    public class MultiListener implements TrainingListener {
        private boolean isMaster = false;

        public MultiListener(boolean isMaster) {
            this.isMaster = isMaster;
        }

        @Override
        public void iterationDone(Model model, int iteration, int epoch) {





        }

        @Override
        public void onEpochStart(Model model) {

        }

        @Override
        public void onEpochEnd(Model model) {
            // each epoch end, then will do weight sync
            if (isMaster) {
                Msg msg = null;
                try {
                    System.out.println("Master is taking...");
                    msg = queue.take();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                INDArray masterP = model.params();
                INDArray slaveP = msg.parameters;

                double loss1 = msg.loss;

                // weight sync
                System.out.println("SGD +++++++++");
                INDArray newP = masterP.add(slaveP);
                newP = newP.div(2);

                model.setParams(newP);

                Msg newMsg = new Msg();
                newMsg.parameters = newP;
                sendQueue.offer(newMsg);
            } else {
                Msg msg = new Msg();
                msg.parameters = model.params();
                msg.loss = model.score();

                System.out.println("Slave is sending...");
                sendQueue.offer(msg);
                Msg newMsg = null;
                try {
                    System.out.println("Slave is waiting for master...");
                    newMsg = queue.take();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("Slave get new P....");

                INDArray newP = newMsg.parameters;
                model.setParams(newP);

                // if get each layer
//                Layer layer = ((MultiLayerNetwork) model).getLayer(0);
//                layer.setParams(newP);

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
    }

    public DualTask(Settings settings) {
        this.settings = settings;
    }

    public DualTask(Settings settings, boolean isMaster) {
        this(settings);
        this.isMaster = isMaster;
    }


    @Override
    public void run() {
        HarReader reader = new HarReader(settings.numLinesToSkip, settings.height,
            settings.width, settings.channel, settings.numClasses, settings.taskNum, settings.delimiter);
        try {
            reader.initialize(new FileSplit(settings.getFile()));
        } catch (Exception e) {
            e.printStackTrace();
        }
        reader.setLabelFromOne(false);

        DataSetIterator iterator = new RecordReaderDataSetIterator(reader, settings.batchSize, settings.labelIndex, settings.numClasses);

        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(iterator);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        iterator.setPreProcessor(normalizer);

        MultiLayerNetwork model = null;
        if (isMaster) {
            MultiLayerConfiguration conf = getModelConf(settings.numClasses, settings.height, settings.width, settings.channel);
            // send conf to others
            Msg message = new Msg();
            message.confJosn = conf.toJson();

            model = new MultiLayerNetwork(conf);
            model.init();
            // send init to others
            message.parameters = model.params();

            w0 = model.params().dup();
            sendQueue.offer(message);
        } else {
            try {
                // read from master
                Msg message = queue.take();
                MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(message.confJosn);
                model = new MultiLayerNetwork(conf);
                // clone will be false in real case
                model.init(message.parameters, true);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        model.setListeners(new MultiListener(isMaster));
        model.fit(iterator, settings.epoch);

        // evaluate
        if (isMaster) {

        }
    }

    static class Settings {


        int epoch = 3;

        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        int taskNum = 10;

        char delimiter = ',';

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        // last pos is label
        // by channel, the label index will be the second one, not actual label index
        int labelIndex = 1;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row

        int numClasses = 3;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
        int batchSize = 4;

        // channel * width = inputwidth
        int channel = 3;
        int height = 1;
        int width = 4;


//        int epoch = 3;
//
//        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
//        int numLinesToSkip = 0;
//        int taskNum = 10;
//
//        char delimiter = ',';
//
//        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
//        // last pos is label
//        // by channel, the label index will be the second one, not actual label index
//        int labelIndex = 1;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
//
//        int numClasses = 3;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
//        int batchSize = 2;
//
//        // channel * width = inputwidth
//        int channel = 2;
//        int height = 1;
//        int width = 2;

        Settings(int numLinesToSkip, int taskNum) {
            this.numLinesToSkip = numLinesToSkip;
            this.taskNum = taskNum;
        }

        public File getFile() throws Exception {
//            return new ClassPathResource("iris_test.txt").getFile();
            return new ClassPathResource("zac.txt").getFile();
        }
    }

    public class Msg {
        INDArray parameters;
        String confJosn;

        INDArray Aw;

        double loss;
    }

}
