package it.units.erallab.utils;

import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.controllers.snn.LIFNeuronWithHomeostasis;
import it.units.erallab.hmsrobots.core.controllers.snn.MultilayerSpikingNetworkWithConverters;
import it.units.erallab.hmsrobots.core.controllers.snn.SpikingFunction;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.Task;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.dyn4j.dynamics.Settings;

import java.io.IOException;
import java.io.PrintStream;
import java.util.List;
import java.util.SortedMap;

public class ThresholdEvolutionPlotter {

  public static void createLocomotionThresholdEvolutionFile(Robot robot, String terrainName, double duration, String fileName) throws IOException {
    createThresholdEvolutionFile(robot, new Locomotion(duration, Locomotion.createTerrain(terrainName), new Settings()), fileName);
  }

  public static void createThresholdEvolutionFile(Robot robot, Task<Robot, ?> task, String fileName) throws IOException {
    SortedMap<Double, Double>[][] thresholdValues = simulateAndGetThresholdEvolution(robot, task);
    CSVPrinter printer = new CSVPrinter(new PrintStream(fileName), CSVFormat.DEFAULT.withDelimiter(';'));
    printer.printRecord(List.of("layer", "neuron", "time", "thresholdValue"));
    for (int layer = 0; layer < thresholdValues.length; layer++) {
      for (int neuron = 0; neuron < thresholdValues[layer].length; neuron++) {
        for (double time : thresholdValues[layer][neuron].keySet()) {
          printer.printRecord(layer, neuron, time, thresholdValues[layer][neuron].get(time));
        }
      }
    }
    printer.flush();
    printer.close();
  }

  @SuppressWarnings("unchecked")
  private static SortedMap<Double, Double>[][] simulateAndGetThresholdEvolution(Robot robot, Task<Robot, ?> task) {
    MultilayerSpikingNetworkWithConverters multilayerSpikingNetworkWithConverters;
    if (robot.getController() instanceof CentralizedSensing && ((CentralizedSensing) robot.getController()).getFunction() instanceof MultilayerSpikingNetworkWithConverters) {
      multilayerSpikingNetworkWithConverters = (MultilayerSpikingNetworkWithConverters) ((CentralizedSensing) robot.getController()).getFunction();
      multilayerSpikingNetworkWithConverters.setSpikesTracker(true);
      multilayerSpikingNetworkWithConverters.setPlotMode(true);
      multilayerSpikingNetworkWithConverters.reset();
    } else {
      throw new IllegalArgumentException("Robot controller does not have variable threshold");
    }
    task.apply(robot);
    SpikingFunction[][] neurons = multilayerSpikingNetworkWithConverters.getMultilayerSpikingNetwork().getNeurons();
    SortedMap<Double, Double>[][] thresholdValues = new SortedMap[neurons.length][];
    for (int layer = 0; layer < neurons.length; layer++) {
      thresholdValues[layer] = new SortedMap[neurons[layer].length];
      for (int neuron = 0; neuron < neurons[layer].length; neuron++) {
        if (!(neurons[layer][neuron] instanceof LIFNeuronWithHomeostasis)) {
          throw new IllegalArgumentException("Robot controller does not have variable threshold");
        }
        thresholdValues[layer][neuron] = ((LIFNeuronWithHomeostasis) neurons[layer][neuron]).getThresholdValues();
      }
    }
    return thresholdValues;
  }

}
