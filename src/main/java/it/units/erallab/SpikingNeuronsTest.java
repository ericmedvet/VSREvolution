package it.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.snn.*;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.stv.*;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.vts.*;
import it.units.erallab.utils.MembraneEvolutionPlotter;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.IOException;
import java.io.PrintStream;
import java.util.*;
import java.util.stream.IntStream;

public class SpikingNeuronsTest {

  private static final double FREQUENCY = 60;
  private static final double SIMULATION_SECONDS = 5;
  private static final double CONSTANT_INPUT_VALUE = 0.5;
  private static final double SIN_FREQUENCY = 1;
  private static final double SQUARE_WAVE_HIGH_VALUE = 1;
  private static final double SQUARE_WAVE_LOW_VALUE = 0;
  private static final double SQUARE_WAVE_LOW_TIME = 3;
  private static final double SQUARE_WAVE_HIGH_TIME = 1;
  private static final double VTS_FREQ = 50;
  private static final double STV_FREQ = 50;
  private static final String[] NEURON_MODELS = {"LIF", "IZ"};
  private static final String[] VTS = {"UNIF", "UNIF_MEM"};

  public static void testConverters() {
    ValueToSpikeTrainConverter valueToSpikeTrainConverter = new UniformValueToSpikeTrainConverter(120);
    SpikeTrainToValueConverter spikeTrainToValueConverter = new AverageFrequencySpikeTrainToValueConverter(120);
    SortedSet<Double> converted = valueToSpikeTrainConverter.convert(1, 1 / (double) 60, 1);
    System.out.println(converted);
    double reconverted = spikeTrainToValueConverter.convert(converted, 1 / (double) 60);
    System.out.println(reconverted);
  }

  public static void main(String[] args) throws IOException {

    //test("IZ", "UNIF", generateSinusoidalInputSignal(), "CONSTANT", null,true);

    String outputTestFileName = "C:\\Users\\giorg\\Documents\\UNITS\\LAUREA MAGISTRALE\\TESI\\OSLOMET\\spikingNeuronsTest.txt";
    //printTest(outputTestFileName);


    String outputMemFileName = "C:\\Users\\giorg\\Documents\\UNITS\\LAUREA MAGISTRALE\\TESI\\OSLOMET\\memoryConvertersTest.txt";
    printTestMemoryConverters(outputMemFileName);
  }

  public static void printTest(String outputFileName) throws IOException {
    CSVPrinter printer = new CSVPrinter(new PrintStream(outputFileName), CSVFormat.DEFAULT.withDelimiter(';'));
    printer.printRecord("model", "vts", "vts_freq", "stv_freq", "input_type", "series", "x", "y");
    for (String neuronModel : NEURON_MODELS) {
      for (String vts : VTS) {
        test(neuronModel, vts, generateConstantInputSignal(), "CONSTANT", printer, false);
        test(neuronModel, vts, generateSinusoidalInputSignal(), "SINUSOIDAL", printer, false);
        test(neuronModel, vts, generateSquareWaveInputSignal(), "SQUARE", printer, false);
      }
    }
  }

  public static void printTestMemoryConverters(String outputFileName) throws IOException {
    CSVPrinter printer = new CSVPrinter(new PrintStream(outputFileName), CSVFormat.DEFAULT.withDelimiter(';'));
    printer.printRecord("model", "vts_freq", "stv_freq", "input_type", "series", "x", "y");
    for (String neuronModel : NEURON_MODELS) {
      testMemoryConverters(neuronModel, generateConstantInputSignal(), "CONSTANT", printer);
      testMemoryConverters(neuronModel, generateSinusoidalInputSignal(), "SINUSOIDAL", printer);
      testMemoryConverters(neuronModel, generateSquareWaveInputSignal(), "SQUARE", printer);
    }
  }

  public static SortedMap<Double, Double> generateConstantInputSignal() {
    double deltaT = 1 / FREQUENCY;
    SortedMap<Double, Double> constantInputSignal = new TreeMap<>();
    for (double t = 0; t <= SIMULATION_SECONDS; t += deltaT) {
      constantInputSignal.put(t, CONSTANT_INPUT_VALUE);
    }
    return constantInputSignal;
  }

  public static SortedMap<Double, Double> generateSinusoidalInputSignal() {
    double deltaT = 1 / FREQUENCY;
    SortedMap<Double, Double> sinusoidalInputSignal = new TreeMap<>();
    for (double t = 0; t <= SIMULATION_SECONDS; t += deltaT) {
      double sinusoidalInput = 0.5 + Math.sin(2 * Math.PI * SIN_FREQUENCY * t) / 2;
      sinusoidalInputSignal.put(t, sinusoidalInput);  // 0 - 1 range
    }
    return sinusoidalInputSignal;
  }

  public static SortedMap<Double, Double> generateSquareWaveInputSignal() {
    double deltaT = 1 / FREQUENCY;
    SortedMap<Double, Double> squareWaveInputSignal = new TreeMap<>();
    for (double t = 0; t <= SIMULATION_SECONDS; t += deltaT) {
      double copyT = t;
      while (copyT > 0) {
        copyT = copyT - (SQUARE_WAVE_LOW_TIME + SQUARE_WAVE_HIGH_TIME);
      }
      double squareInput;
      if (copyT < -SQUARE_WAVE_HIGH_TIME) {
        squareInput = SQUARE_WAVE_LOW_VALUE;
      } else {
        squareInput = SQUARE_WAVE_HIGH_VALUE;
      }
      squareWaveInputSignal.put(t, squareInput);
    }
    return squareWaveInputSignal;
  }

  public static void testMemoryConverters(String neuronModel, SortedMap<Double, Double> inputSignal, String signalName, CSVPrinter printer) {
    SpikingNeuron spikingNeuron;
    if (neuronModel.equals("LIF")) {
      spikingNeuron = new LIFNeuron(0, 1.2, 0.1, true);
    } else {
      spikingNeuron = new IzhikevicNeuron(true);
    }
    ValueToSpikeTrainConverter valueToSpikeTrainConverter = new UniformWithMemoryValueToSpikeTrainConverter(VTS_FREQ);
    int[] memorySizes = IntStream.iterate(10, x -> x+10).limit(1).toArray();
    SpikeTrainToValueConverter[] spikeTrainToValueConverter = new SpikeTrainToValueConverter[memorySizes.length];
    SortedMap<Double, Double>[] outputSignals = new SortedMap[memorySizes.length];
    for(int i=0; i< memorySizes.length; i++){
      spikeTrainToValueConverter[i] = new MovingAverageSpikeTrainToValueConverter(STV_FREQ, memorySizes[i]);
      outputSignals[i] = new TreeMap<>();
    }
    double deltaT = 1 / FREQUENCY;
    inputSignal.forEach((t, v) -> {
      SortedSet<Double> outputSpikes = passValueToNeuron(spikingNeuron, valueToSpikeTrainConverter, v, t, deltaT);
      for (int i = 0; i < memorySizes.length; i++) {
        outputSignals[i].put(t, spikeTrainToValueConverter[i].convert(outputSpikes, deltaT));
      }
    });
    List<String> record = List.of(neuronModel, "" + VTS_FREQ, "" + STV_FREQ, signalName);
    inputSignal.forEach((x, y) -> {
      List<String> thisRecord = new ArrayList<>(record);
      thisRecord.add("INPUT");
      thisRecord.add("" + x);
      thisRecord.add("" + y);
      try {
        printer.printRecord(thisRecord);
      } catch (IOException e) {
        e.printStackTrace();
      }
    });
    for (int i = 0; i < outputSignals.length; i++) {
      final int memSize = memorySizes[i];
      outputSignals[i].forEach((x, y) -> {
        List<String> thisRecord = new ArrayList<>(record);
        thisRecord.add("OUTPUT_" + memSize);
        thisRecord.add("" + x);
        double y1 = (y + 1) / 2;
        thisRecord.add("" + y1);
        try {
          printer.printRecord(thisRecord);
        } catch (IOException e) {
          e.printStackTrace();
        }
      });
    }
  }

  public static void test(String neuronModel, String vts, SortedMap<Double, Double> inputSignal, String signalName, CSVPrinter printer, boolean plot) {
    SpikingNeuron spikingNeuron;
    if (neuronModel.equals("LIF")) {
      spikingNeuron = new LIFNeuron(0, 1.2, 0.1, true);
    } else {
      spikingNeuron = new IzhikevicNeuron(true);
    }
    ValueToSpikeTrainConverter valueToSpikeTrainConverter;
    if (vts.equals("UNIF")) {
      valueToSpikeTrainConverter = new UniformValueToSpikeTrainConverter(VTS_FREQ);
    } else {
      valueToSpikeTrainConverter = new UniformWithMemoryValueToSpikeTrainConverter(VTS_FREQ);
    }
    SpikeTrainToValueConverter spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter(STV_FREQ);
    double deltaT = 1 / FREQUENCY;
    SortedMap<Double, Double> outputSignal = new TreeMap<>();
    inputSignal.forEach((t, v) -> {
      double outputValue = passValueToNeuron(spikingNeuron, valueToSpikeTrainConverter, spikeTrainToValueConverter, v, t, deltaT);
      outputSignal.put(t, outputValue);
    });
    List<Double> outputSpikes = new ArrayList<>();
    spikingNeuron.getMembranePotentialValues().forEach((x, y) -> {
      if (y >= spikingNeuron.getThresholdPotential())
        outputSpikes.add(x);
    });
    if (plot) MembraneEvolutionPlotter.plotMembranePotentialEvolutionWithInputSpikes(spikingNeuron, 800, 600);
    print(neuronModel, vts, inputSignal, outputSignal, spikingNeuron.getMembranePotentialValues(), spikingNeuron.getInputSpikesValues().keySet(), outputSpikes, signalName, printer);
  }

  private static void print(String neuronModel, String vts,
                            SortedMap<Double, Double> inputSignal,
                            SortedMap<Double, Double> outputSignal,
                            SortedMap<Double, Double> membranePotentialEvolution,
                            Collection<Double> inputSpikes,
                            Collection<Double> outputSpikes,
                            String inputType, CSVPrinter printer) {
    List<String> record = List.of(neuronModel, vts, "" + VTS_FREQ, "" + STV_FREQ, inputType);

    inputSignal.forEach((x, y) -> {
      List<String> thisRecord = new ArrayList<>(record);
      thisRecord.add("INPUT_SIGNAL");
      thisRecord.add("" + x);
      thisRecord.add("" + y);
      try {
        printer.printRecord(thisRecord);
      } catch (IOException e) {
        e.printStackTrace();
      }
    });

    outputSignal.forEach((x, y) -> {
      List<String> thisRecord = new ArrayList<>(record);
      thisRecord.add("OUTPUT_SIGNAL");
      thisRecord.add("" + x);
      double y1 = (y + 1) / 2;
      thisRecord.add("" + y1);
      try {
        printer.printRecord(thisRecord);
      } catch (IOException e) {
        e.printStackTrace();
      }
    });

    membranePotentialEvolution.forEach((x, y) -> {
      List<String> thisRecord = new ArrayList<>(record);
      thisRecord.add("POTENTIAL");
      thisRecord.add("" + x);
      thisRecord.add("" + y);
      try {
        printer.printRecord(thisRecord);
      } catch (IOException e) {
        e.printStackTrace();
      }
    });

    inputSpikes.forEach((x) -> {
      List<String> thisRecord = new ArrayList<>(record);
      thisRecord.add("INPUT_SPIKE");
      thisRecord.add("" + x);
      thisRecord.add("" + 1);
      try {
        printer.printRecord(thisRecord);
      } catch (IOException e) {
        e.printStackTrace();
      }
    });

    outputSpikes.forEach((x) -> {
      List<String> thisRecord = new ArrayList<>(record);
      thisRecord.add("OUTPUT_SPIKE");
      thisRecord.add("" + x);
      thisRecord.add("" + 1);
      try {
        printer.printRecord(thisRecord);
      } catch (IOException e) {
        e.printStackTrace();
      }
    });

  }

  private static SortedSet<Double> passValueToNeuron(SpikingNeuron spikingNeuron,
                                                     ValueToSpikeTrainConverter valueToSpikeTrainConverter,
                                                     double inputValue,
                                                     double absoluteTime,
                                                     double timeWindow) {
    SortedSet<Double> inputSpikeTrain = valueToSpikeTrainConverter.convert(inputValue, timeWindow, absoluteTime);
    SortedMap<Double, Double> weightedInputSpikeTrain = new TreeMap<>();
    inputSpikeTrain.forEach(t -> weightedInputSpikeTrain.put(t, 1d));
    return spikingNeuron.compute(weightedInputSpikeTrain, absoluteTime);
  }

  private static double passValueToNeuron(SpikingNeuron spikingNeuron,
                                          ValueToSpikeTrainConverter valueToSpikeTrainConverter,
                                          SpikeTrainToValueConverter spikeTrainToValueConverter,
                                          double inputValue,
                                          double absoluteTime,
                                          double timeWindow) {
    SortedSet<Double> outputSpikeTrain = passValueToNeuron(spikingNeuron, valueToSpikeTrainConverter, inputValue, absoluteTime, timeWindow);
    //System.out.printf("Input: %.3f\tOutput: %.3f\n", inputValue, outputValue);
    //System.out.printf("Input spikes: %s\nOutput spikes: %s\n\n", inputSpikeTrain.toString(), outputSpikeTrain.toString());
    return spikeTrainToValueConverter.convert(outputSpikeTrain, timeWindow);
  }

}
