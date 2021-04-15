package it.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.snn.*;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.stv.*;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.vts.*;
import it.units.erallab.utils.MembraneEvolutionPlotter;
import it.units.erallab.utils.Utils;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.IOException;
import java.io.PrintStream;
import java.util.*;
import java.util.stream.IntStream;

public class SpikingNeuronsTest {

  private static final double FREQUENCY = 60;
  private static final double SIMULATION_SECONDS = 10;
  private static final double CONSTANT_INPUT_VALUE = 0.5;
  private static final double SIN_FREQUENCY = 1;
  private static final double SQUARE_WAVE_HIGH_VALUE = 1;
  private static final double SQUARE_WAVE_LOW_VALUE = 0;
  private static final double SQUARE_WAVE_LOW_TIME = 2;
  private static final double SQUARE_WAVE_HIGH_TIME = 1;
  private static final double VTS_FREQ = 50;
  private static final double STV_FREQ = 50;
  private static final String[] NEURON_MODELS = {"LIF", "IZ", "LIF_H"};
  private static final int[] MEMORY_SIZES = IntStream.iterate(5, x -> x+5).limit(4).toArray();

  //private static final String[] NEURON_MODELS = {"LIF_H"};
  //private static final int[] MEMORY_SIZES = {5};

  public static void main(String[] args) throws IOException {

    String outputTestFileName = "C:\\Users\\giorg\\Documents\\UNITS\\LAUREA MAGISTRALE\\TESI\\OSLOMET\\spikingNeuronsTest.txt";
    printTest(outputTestFileName);

  }

  public static void printTest(String outputFileName) throws IOException {
    CSVPrinter printer = new CSVPrinter(new PrintStream(outputFileName), CSVFormat.DEFAULT.withDelimiter(';'));
    printer.printRecord("model", "vts_freq", "stv_freq", "input_type", "series", "x", "y");
    for (String neuronModel : NEURON_MODELS) {
      testSnn(neuronModel, generateConstantInputSignal(), "CONSTANT", printer);
      testSnn(neuronModel, generateSinusoidalInputSignal(SIMULATION_SECONDS, SIN_FREQUENCY), "SINUSOIDAL", printer);
      testSnn(neuronModel, generateSquareWaveInputSignal(), "SQUARE", printer);
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

  public static SortedMap<Double, Double> generateSinusoidalInputSignal(double totalSeconds, double frequency) {
    double deltaT = 1 / FREQUENCY;
    SortedMap<Double, Double> sinusoidalInputSignal = new TreeMap<>();
    for (double t = 0; t <= totalSeconds; t += deltaT) {
      double sinusoidalInput = 0.5 + Math.sin(2 * Math.PI * frequency * t) / 2;
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

  public static void testSnn(String neuronModel, SortedMap<Double, Double> inputSignal, String signalName, CSVPrinter printer) {
    SpikingNeuron spikingNeuron;
    if (neuronModel.equals("LIF")) {
      spikingNeuron = new LIFNeuron(0, 1.0, 0.01, true);
    } else if (neuronModel.equals("IZ")) {
      spikingNeuron = new IzhikevicNeuron(true);
    } else {
      spikingNeuron = new LIFNeuronWithHomeostasis(0, 1.0, 0.01, 0.0, true);
      spikingNeuron.setSumOfIncomingWeights(1);
    }
    ValueToSpikeTrainConverter valueToSpikeTrainConverter = new UniformWithMemoryValueToSpikeTrainConverter(VTS_FREQ);
    SpikeTrainToValueConverter[] spikeTrainToValueConverter = new SpikeTrainToValueConverter[MEMORY_SIZES.length];
    SortedMap<Double, Double>[] outputSignals = new SortedMap[MEMORY_SIZES.length];
    for (int i = 0; i < MEMORY_SIZES.length; i++) {
      spikeTrainToValueConverter[i] = new MovingAverageSpikeTrainToValueConverter(STV_FREQ, MEMORY_SIZES[i]);
      outputSignals[i] = new TreeMap<>();
    }
    double deltaT = 1 / FREQUENCY;
    inputSignal.forEach((t, v) -> {
      SortedSet<Double> outputSpikes = passValueToNeuron(spikingNeuron, valueToSpikeTrainConverter, v, t, deltaT);
      for (int i = 0; i < MEMORY_SIZES.length; i++) {
        outputSignals[i].put(t, spikeTrainToValueConverter[i].convert(outputSpikes, deltaT));
      }
    });
    //MembraneEvolutionPlotter.plotMembranePotentialEvolution(spikingNeuron,800,600);
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
      final int memSize = MEMORY_SIZES[i];
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

}
