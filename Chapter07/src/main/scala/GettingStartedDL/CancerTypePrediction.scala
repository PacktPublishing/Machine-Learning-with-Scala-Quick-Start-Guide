package GettingStartedDL

import java.io.File
import java.io.IOException
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.Layer
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.LSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

object CancerTypePrediction {   
  	def readCSVDataset(csvFileClasspath:String, batchSize:Int, labelIndex:Int, numClasses:Int) : DataSetIterator = {
		val rr:RecordReader  = new CSVRecordReader()
		val input:File  = new File(csvFileClasspath)
		rr.initialize(new FileSplit(input))
		val iterator:DataSetIterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses)
		return iterator
	}
  
  def main(args: Array[String]): Unit = {
    val numEpochs = 10
    // Show data paths
    val trainPath = "C:/Users/admin-karim/Desktop/old2/TCGA-PANCAN/TCGA_train.csv"
    val testPath = "C:/Users/admin-karim/Desktop/old2/TCGA-PANCAN/TCGA_test.csv"

    // ----------------------------------
    // Preparing training and test set. 	
    val labelIndex = 20531
    val numClasses = 5
    val batchSize = 128

    // This dataset is used for training 
    val trainingDataIt: DataSetIterator = readCSVDataset(trainPath, batchSize, labelIndex, numClasses)

    // This is the data we want to classify
    val testDataIt:DataSetIterator  = readCSVDataset(testPath, batchSize, labelIndex, numClasses)	

    // ----------------------------------
    // Network hyperparameters
    val seed = 12345
    val numInputs = labelIndex
    val numOutputs = numClasses
    val numHiddenNodes = 5000

    //First LSTM layer
    val layer_0 = new LSTM.Builder()
      .nIn(numInputs)
      .nOut(numHiddenNodes)
      .activation(Activation.RELU)
      .build()

    //Second LSTM layer
    val layer_1 = new LSTM.Builder()
      .nIn(numHiddenNodes)
      .nOut(numHiddenNodes)
      .activation(Activation.RELU)
      .build()

    //Third LSTM layer
    val layer_2 = new LSTM.Builder()
      .nIn(numHiddenNodes)
      .nOut(numHiddenNodes)
      .activation(Activation.RELU)
      .build()

    //RNN output layer
    val layer_3 = new RnnOutputLayer.Builder()
      .activation(Activation.SOFTMAX)
      .lossFunction(LossFunction.MCXENT)
      .nIn(numHiddenNodes)
      .nOut(numOutputs)
      .build()

    // Create network configuration and conduct network training
    val LSTMconf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(seed) //Random number generator seed for improved repeatability. Optional.
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .weightInit(WeightInit.XAVIER)
      .updater(new Adam(5e-3))
      .l2(1e-5)
      .list()
      .layer(0, layer_0)
      .layer(1, layer_1)
      .layer(2, layer_2)
      .layer(3, layer_3)
      .pretrain(false).backprop(true).build()

    // Create and initialize multilayer network 
    val model: MultiLayerNetwork = new MultiLayerNetwork(LSTMconf)
    model.init()

    //print the score with every 1 iteration
    model.setListeners(new ScoreIterationListener(1));

    //Print the  number of parameters in the network (and for each layer)
    val layers = model.getLayers()
    var totalNumParams = 0
    var i = 0

    for (i <- 0 to layers.length-1) {
      val nParams = layers(i).numParams()
      println("Number of parameters in layer " + i + ": " + nParams)
      totalNumParams = totalNumParams + nParams
    }

    println("Total number of network parameters: " + totalNumParams)

    var j = 0
    println("Train model....")
    for (j <- 0 to numEpochs-1) {
      model.fit(trainingDataIt)
    }

    println("Evaluate model....")
    val eval: Evaluation = new Evaluation(5) //create an evaluation object with 10 possible classes
    
    while (testDataIt.hasNext()) {
      val next:DataSet = testDataIt.next()
      val output:INDArray  = model.output(next.getFeatureMatrix()) //get the networks prediction
      eval.eval(next.getLabels(), output) //check the prediction against the true class
    }

    println(eval.stats())
    println("****************Example finished********************")
  }
}
