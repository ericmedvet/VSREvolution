package it.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.snn.LIFNeuron;
import it.units.erallab.hmsrobots.core.controllers.snn.MultilayerSpikingNetwork;
import it.units.erallab.hmsrobots.core.controllers.snn.SpikingFunction;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.stv.MovingAverageSpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.stv.SpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.vts.UniformWithMemoryValueToSpikeTrainConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.vts.ValueToSpikeTrainConverter;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.PrintStream;
import java.util.*;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class SpikingNetworkTest {

  public static void main(String[] args) throws Exception {

    String outputFileName = "C:\\Users\\giorg\\Documents\\UNITS\\LAUREA MAGISTRALE\\TESI\\OSLOMET\\snnDataLowFreq.txt";
    String mlpOutputFileName = "C:\\Users\\giorg\\Documents\\UNITS\\LAUREA MAGISTRALE\\TESI\\OSLOMET\\mlp.txt";
    CSVPrinter mlpPrinter = new CSVPrinter(new PrintStream(mlpOutputFileName), CSVFormat.DEFAULT.withDelimiter(';'));
    mlpPrinter.printRecord(List.of("seed", "weightsMin", "weightsMax", "nOfInputs", "nOfOutputs", "nInnerLayers", "innerLayerRatio",
            "outputsMax", "outputsMin", "outputsAvg"));
    CSVPrinter printer = new CSVPrinter(new PrintStream(outputFileName), CSVFormat.DEFAULT.withDelimiter(';'));
    printer.printRecord(List.of("seed", "weightsMin", "weightsMax", "inputConverterFrequency",
            "outputConverterFrequency", "outputConverterMemory", "nOfInputs", "nOfOutputs", "nInnerLayers", "innerLayerRatio",
            "outputsMax", "outputsMin", "outputsAvg"));

    int nInputs = 3;
    double[] inputFrequencies = {0.5, 1d, 2d};
    double simulationSeconds = 10;

    if (nInputs != inputFrequencies.length) {
      throw new IllegalArgumentException("N of inputs is different from the number of input frequencies");
    }

    int nOutputs = 3;
    int[] innerLayers = IntStream.range(0, 4).toArray();
    double innerLayerRatio = 1;

    double weightMin = -1d;
    double weightMax = 1d;

    double valueToSpikeTrainConverterFrequency = 50;
    double[] spikeTrainToValueConverterFrequencies = DoubleStream.iterate(20, x -> x + 5).limit(10).toArray();
    int[] spikeTrainToValueMemorySizes = IntStream.iterate(1, x -> x + 2).limit(10).toArray();
    double restingPotential = 0;
    double thresholdPotential = 1;
    double lambdaDecay = 0.01;

    int[] seeds = IntStream.range(0, 20).toArray();

    SortedMap<Double, Double>[] inputSignals = new SortedMap[nInputs];
    for (int i = 0; i < inputSignals.length; i++) {
      inputSignals[i] = SpikingNeuronsTest.generateSinusoidalInputSignal(simulationSeconds, inputFrequencies[i]);
    }
    SortedMap<Double, Double>[] outputSignals = new SortedMap[nOutputs];

    for (double spikeTrainToValueConverterFrequency : spikeTrainToValueConverterFrequencies) {
      for (int spikeTrainToValueMemorySize : spikeTrainToValueMemorySizes) {
        SpikingFunction spikingFunction = new LIFNeuron(restingPotential, thresholdPotential, lambdaDecay);
        ValueToSpikeTrainConverter valueToSpikeTrainConverter = new UniformWithMemoryValueToSpikeTrainConverter(valueToSpikeTrainConverterFrequency);
        SpikeTrainToValueConverter spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter(spikeTrainToValueConverterFrequency, spikeTrainToValueMemorySize);
        valueToSpikeTrainConverter.reset();
        spikeTrainToValueConverter.reset();
        for (int nInnerLayers : innerLayers) {
          int[] innerNeurons = innerNeurons(nInputs, nOutputs, nInnerLayers, innerLayerRatio);
          MultilayerSpikingNetwork multilayerSpikingNetwork = new MultilayerSpikingNetwork(nInputs, innerNeurons, nOutputs, spikingFunction, valueToSpikeTrainConverter, spikeTrainToValueConverter);
          multilayerSpikingNetwork.reset();
          for (int seed : seeds) {
            // reset output
            IntStream.range(0, outputSignals.length).parallel().forEach(i -> outputSignals[i] = new TreeMap<>());
            Random random = new Random(seed);
            double[] weights = multilayerSpikingNetwork.getParams();
            double weightRange = weightMax - weightMin;
            IntStream.range(0, weights.length).forEach(i -> weights[i] = random.nextDouble() * weightRange + weightMin);
            multilayerSpikingNetwork.setParams(weights);
            for (double time : inputSignals[0].keySet()) {
              double[] input = new double[inputSignals.length];
              for (int i = 0; i < input.length; i++) {
                input[i] = inputSignals[i].get(time);
              }
              double[] output = multilayerSpikingNetwork.apply(time, input);
              for (int i = 0; i < output.length; i++) {
                outputSignals[i].put(time, output[i]);
              }
            }
            double outputMaxAverage = Arrays.stream(outputSignals).mapToDouble(
                    x -> x.values().stream().mapToDouble(d -> d).max().getAsDouble()
            ).average().getAsDouble();
            double outputMinAverage = Arrays.stream(outputSignals).mapToDouble(
                    x -> x.values().stream().mapToDouble(d -> d).min().getAsDouble()
            ).average().getAsDouble();
            double outputAvgAverage = Arrays.stream(outputSignals).mapToDouble(
                    x -> x.values().stream().mapToDouble(d -> d).average().getAsDouble()
            ).average().getAsDouble();
            List<Object> row = List.of(seed, weightMin, weightMax, valueToSpikeTrainConverterFrequency, spikeTrainToValueConverterFrequency,
                    spikeTrainToValueMemorySize, nInputs, nOutputs, nInnerLayers, innerLayerRatio, outputMaxAverage, outputMinAverage, outputAvgAverage);
            printer.printRecord(row);
          }
        }
      }
    }

    for (int nInnerLayers : innerLayers) {
      int[] innerNeurons = innerNeurons(nInputs, nOutputs, nInnerLayers, innerLayerRatio);
      MultiLayerPerceptron multiLayerPerceptron = new MultiLayerPerceptron(MultiLayerPerceptron.ActivationFunction.TANH, nInputs, innerNeurons, nOutputs);
      for (int seed : seeds) {
        // reset output
        IntStream.range(0, outputSignals.length).parallel().forEach(i -> outputSignals[i] = new TreeMap<>());
        Random random = new Random(seed);
        double[] weights = multiLayerPerceptron.getParams();
        double weightRange = weightMax - weightMin;
        IntStream.range(0, weights.length).forEach(i -> weights[i] = random.nextDouble() * weightRange + weightMin);
        multiLayerPerceptron.setParams(weights);
        for (double time : inputSignals[0].keySet()) {
          double[] input = new double[inputSignals.length];
          for (int i = 0; i < input.length; i++) {
            input[i] = inputSignals[i].get(time);
          }
          double[] output = multiLayerPerceptron.apply(time, input);
          for (int i = 0; i < output.length; i++) {
            outputSignals[i].put(time, output[i]);
          }
        }
        double outputMaxAverage = Arrays.stream(outputSignals).mapToDouble(
                x -> x.values().stream().mapToDouble(d -> d).max().getAsDouble()
        ).average().getAsDouble();
        double outputMinAverage = Arrays.stream(outputSignals).mapToDouble(
                x -> x.values().stream().mapToDouble(d -> d).min().getAsDouble()
        ).average().getAsDouble();
        double outputAvgAverage = Arrays.stream(outputSignals).mapToDouble(
                x -> x.values().stream().mapToDouble(d -> d).average().getAsDouble()
        ).average().getAsDouble();
        List<Object> row = List.of(seed, weightMin, weightMax, nInputs, nOutputs, nInnerLayers, innerLayerRatio, outputMaxAverage, outputMinAverage, outputAvgAverage);
        mlpPrinter.printRecord(row);
      }
    }
  }

  private static int[] innerNeurons(int nOfInputs, int nOfOutputs, int nOfInnerLayers, double innerLayerRatio) {
    int[] innerNeurons = new int[nOfInnerLayers];
    int centerSize = (int) Math.max(2, Math.round(nOfInputs * innerLayerRatio));
    if (nOfInnerLayers > 1) {
      for (int i = 0; i < nOfInnerLayers / 2; i++) {
        innerNeurons[i] = nOfInputs + (centerSize - nOfInputs) / (nOfInnerLayers / 2 + 1) * (i + 1);
      }
      for (int i = nOfInnerLayers / 2; i < nOfInnerLayers; i++) {
        innerNeurons[i] = centerSize + (nOfOutputs - centerSize) / (nOfInnerLayers / 2 + 1) * (i - nOfInnerLayers / 2);
      }
    } else if (nOfInnerLayers > 0) {
      innerNeurons[0] = centerSize;
    }
    return innerNeurons;
  }


}
