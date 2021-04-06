package it.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.snn.LIFNeuron;
import it.units.erallab.hmsrobots.core.controllers.snn.MultilayerSpikingNetwork;
import it.units.erallab.hmsrobots.core.controllers.snn.SpikingFunction;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.IOException;
import java.io.PrintStream;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.stream.DoubleStream;

public class SnnTest2 {

  private static int N_INPUTS = 2;
  private static int N_OUTPUTS = 2;

  public static void main(String[] args) throws Exception{

    String outputFileName = "C:\\Users\\giorg\\Documents\\UNITS\\LAUREA MAGISTRALE\\TESI\\OSLOMET\\snnTest.txt";

    SortedMap<Double,Double> input1 = SpikingNeuronsTest.generateConstantInputSignal();
    SortedMap<Double,Double> input2 = SpikingNeuronsTest.generateConstantInputSignal();
    SortedMap<Double,Double> output1 = new TreeMap<>();
    SortedMap<Double,Double> output2 = new TreeMap<>();

    /*SpikingFunction[][] spikingFunction = new SpikingFunction[2][];
    spikingFunction[0] = new SpikingFunction[N_INPUTS];
    spikingFunction[1] = new SpikingFunction[N_OUTPUTS];
    for (int i = 0; i < spikingFunction.length; i++) {
      for (int j = 0; j < spikingFunction[i].length; j++) {
        spikingFunction[i][j] = new LIFNeuron(true);
      }
    }*/
    SpikingFunction[][] spikingFunction = new SpikingFunction[1][1];
    spikingFunction[0][0] = new LIFNeuron(true);
    double[] weights = new double[MultilayerSpikingNetwork.countWeights(spikingFunction)];
    for (int i = 0; i < weights.length; i++) {
      weights[i] = 1d;
    }
    MultilayerSpikingNetwork snn = new MultilayerSpikingNetwork(spikingFunction, weights);


    double[] currentInput = new double[1];
    //double[] currentInput = new double[N_INPUTS];
    input1.forEach((t,v)->{
      currentInput[0] = v;
      //currentInput[1] = input2.get(t);
      double[] currentOutput = snn.apply(t, currentInput);
      output1.put(t,currentOutput[0]);
      //output2.put(t,currentOutput[1]);
    });

    CSVPrinter printer = new CSVPrinter(new PrintStream(outputFileName), CSVFormat.DEFAULT.withDelimiter(';'));
    printer.printRecord("neuron", "io", "x", "y");
    printPlot(printer,input1,1,"input");
    //printPlot(printer,input2,2,"input");
    printPlot(printer,output1,1,"output");
    //printPlot(printer,output2,2,"output");

  }

  private static void printPlot(CSVPrinter printer, SortedMap<Double, Double> plot, int neuron, String io) {
    plot.forEach((x,y)-> {
      try {
        printer.printRecord(neuron,io,x,y);
      } catch (IOException e) {
        e.printStackTrace();
      }
    });
  }


}
