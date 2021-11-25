package it.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.controllers.snn.learning.*;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.*;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

public class SNNLearningTest {

  private static final double DELTA_T = 1d / 60;
  private static final double MAX_T = 5;
  private static final double W1_INIT = 0.5;
  private static final double W2_INIT = -0.5;
  private static final double[] PHI_2 = IntStream.range(1, 6).mapToDouble(i -> i * Math.PI / 3).toArray();
  private static final STDPLearningRule HA = new AsymmetricHebbianLearningRule();
  private static final STDPLearningRule HS = new SymmetricHebbianLearningRule();
  private static final STDPLearningRule AA = new AsymmetricAntiHebbianLearningRule();
  private static final STDPLearningRule AS = new SymmetricAntiHebbianLearningRule();
  private static final double[] ASYMMETRIC_PARAMS = {0.8, 0.8, 2, 2};
  private static final double[] SYMMETRIC_PARAMS = {6, 1, 3.5, 13.5};
  private static final STDPLearningRule[] RULES = {HA, HS, AA, AS};

  private static double[] generateSinusoidalSignal(double phi) {
    List<Double> signal = new ArrayList<>();
    for (double t = 0; t < MAX_T; t += DELTA_T) {
      signal.add(Math.sin(2 * Math.PI * t + phi));
    }
    return signal.stream().mapToDouble(d -> d).toArray();
  }

  public static void main(String[] args) throws Exception {
    HA.setParams(ASYMMETRIC_PARAMS);
    AA.setParams(ASYMMETRIC_PARAMS);
    HS.setParams(SYMMETRIC_PARAMS);
    AS.setParams(SYMMETRIC_PARAMS);

    CSVPrinter printer = new CSVPrinter(new PrintStream("C:\\Users\\giorg\\Documents\\UNITS\\LAUREA_MAGISTRALE\\TESI\\SNN\\OtherData\\offline_exp.txt"), CSVFormat.DEFAULT.withDelimiter(';'));
    printer.printRecord("t", "rule_1", "rule_2", "phi_1", "phi_2", "delta_output", "w_1", "w_2", "w_1_starting", "w_2_starting");

    QuantizedSpikingFunction[][] neurons = {{new QuantizedLIFNeuron(), new QuantizedLIFNeuron()}, {new QuantizedLIFNeuron()}};
    double[][][] weights = {{{W1_INIT}, {W2_INIT}}};
    double[] firstSignal = generateSinusoidalSignal(0);

    for (double phi : PHI_2) {
      double[] secondSignal = generateSinusoidalSignal(phi);
      QuantizedMultilayerSpikingNetwork noLearningInnerSnn = new QuantizedMultilayerSpikingNetwork(neurons, weights);
      QuantizedMultilayerSpikingNetworkWithConverters<?> noLearningSnn = new QuantizedMultilayerSpikingNetworkWithConverters<>(noLearningInnerSnn);
      reset(neurons);
      double[] noLearningOutputSignal = IntStream.range(0, firstSignal.length).mapToDouble(k -> noLearningSnn.apply(k * DELTA_T, new double[]{firstSignal[k], secondSignal[k]})[0]).toArray();

      System.out.printf("Phase %.2f\n", phi);
      // will iterate over rules
      for (STDPLearningRule firstRule : RULES) {
        for (STDPLearningRule secondRule : RULES) {
          STDPLearningRule[][][] rules = {{{firstRule}, {secondRule}}};
          String firstRuleName = firstRule.getClass().toString().substring(63).replace("LearningRule", "");
          String secondRuleName = secondRule.getClass().toString().substring(63).replace("LearningRule", "");
          QuantizedMultilayerSpikingNetwork innerSnn = new QuantizedLearningMultilayerSpikingNetwork(neurons, weights, rules);
          innerSnn.setWeightsTracker(true);
          QuantizedMultilayerSpikingNetworkWithConverters<?> snn = new QuantizedMultilayerSpikingNetworkWithConverters<>(innerSnn);
          reset(neurons);
          double[] learningOutputSignal = IntStream.range(0, firstSignal.length).mapToDouble(k -> snn.apply(k * DELTA_T, new double[]{firstSignal[k], secondSignal[k]})[0]).toArray();
          double[] output = IntStream.range(0, learningOutputSignal.length).mapToDouble(i -> learningOutputSignal[i] - noLearningOutputSignal[i]).toArray();
          Map<Double, double[]> weightInTime = innerSnn.getWeightsInTime();
          int k = 0;
          for (Map.Entry<Double, double[]> entry : weightInTime.entrySet()) {
            double[] currentWeights = entry.getValue();
            printer.printRecord(entry.getKey(), firstRuleName, secondRuleName, 0, phi, output[k], currentWeights[0], currentWeights[1], W1_INIT, W2_INIT);
            k += 1;
          }
          System.out.printf("%s\t%s\n", firstRuleName, secondRuleName);
        }
      }
      System.out.println();
    }
  }

  private static void reset(QuantizedSpikingFunction[][] neurons) {
    Arrays.stream(neurons).forEach(layer -> Arrays.stream(layer).forEach(Resettable::reset));
  }

}
