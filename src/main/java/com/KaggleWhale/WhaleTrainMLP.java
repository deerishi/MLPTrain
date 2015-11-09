package com.KaggleWhale;

/**
 * Hello world!
 *
 */
//import org.apache.jute.RecordReader;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import  java.io.*;
import java.util.*;
import org.slf4j.LoggerFactory;

import java.util.*;
public class WhaleTrainMLP
{
    private static Logger log = LoggerFactory.getLogger(WhaleTrainMLP.class);
    public static void main( String[] args ) throws Exception
    {
         Nd4j.ENFORCE_NUMERICAL_STABILITY=true;
        final int numRows=28;
        final int numColumns=28;
        int seed=1234;
        DataSet data,trainInput;
        int iterations=5;
        SplitTestAndTrain trainTest;
        int batchSize=649;
        int listenerFreq = iterations/5;
        String prefix="/home/drishi/";
        String LabelPath=prefix+"LabelWise Folder Whale Data";
        int splitTrainNum=(int) (batchSize*0.9);
        List<String> labels=new ArrayList<String>();
        List<INDArray> testInput = new ArrayList<INDArray>();
        List<INDArray> testLabels = new ArrayList<INDArray>();
        for(File f : new File(LabelPath).listFiles())
        {
            System.out.println("label is " + f.getName());
            labels.add(f.getName());
        }
        RecordReader recordReader=new ImageRecordReader(28,28,3,true,labels);
        recordReader.initialize(new FileSplit(new File(LabelPath)));
        DataSetIterator iter=new RecordReaderDataSetIterator(recordReader,batchSize,28*28*3,labels.size() );


       //Now build the neural net layer

        log.info("Building the MLP");
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        int outputNum=labels.size();
        MultiLayerConfiguration conf=new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(iterations)
                .learningRate(1e-1f)
                .momentum(0.5)
                .momentumAfter(Collections.singletonMap(3, 0.9))
                .useDropConnect(true)
                .list(2)
                .layer(0, new DenseLayer.Builder()
                        .nIn(numColumns * numRows * 3)
                        .nOut(4000)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1,new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(1000)
                        .nOut(outputNum)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        MultiLayerNetwork model=new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));
        System.out.println("hi");
        int ex=0;
        while(iter.hasNext()){
            data=iter.next();
           // System.out.println("iter is " + data.numExamples());
            ex=ex+data.numExamples();
            System.out.println("size is "+ data.numExamples()+ " ex is "+ex);
            trainTest=data.splitTestAndTrain(splitTrainNum,new Random(seed));
           trainInput=trainTest.getTrain(); //get Feature Matrix and Labels for Training
           testInput.add(trainTest.getTest().getFeatureMatrix());
            testLabels.add(trainTest.getTest().getLabels());
            model.fit(trainInput);
            System.out.println("we trained " + trainInput.numExamples());

        }
        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
        for(int i = 0; i < testInput.size(); i++) {
            INDArray output = model.output(testInput.get(i));
            eval.eval(testLabels.get(i), output);
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");
        System.out.println( "Hello World!" );
    }
}
