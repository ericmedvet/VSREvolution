package it.units.erallab.utils;

import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.controllers.snn.MultilayerSpikingNetwork;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.Task;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.dyn4j.dynamics.Settings;

import java.io.IOException;
import java.io.PrintStream;
import java.util.List;


public class RasterPlotCreator {

  public static void createLocomotionRasterPlotFile(Robot<?> robot, String terrainName, double duration, String fileName) throws IOException {
    createRasterPlotFile(robot, new Locomotion(duration, Locomotion.createTerrain(terrainName), new Settings()), fileName);
  }

  public static void createRasterPlotFile(Robot<?> robot, Task<Robot<?>, ?> task, String fileName) throws IOException {
    List<Double>[][] spikes = simulateAndGetSpikes(robot, task);
    CSVPrinter printer = new CSVPrinter(new PrintStream(fileName), CSVFormat.DEFAULT.withDelimiter(';'));
    printer.printRecord(List.of("layer", "neuron", "spikeTime"));
    for (int layer = 0; layer < spikes.length; layer++) {
      for (int neuron = 0; neuron < spikes[layer].length; neuron++) {
        for (double spikeTime : spikes[layer][neuron]) {
          printer.printRecord(layer, neuron, spikeTime);
        }
      }
    }
    printer.flush();
    printer.close();
  }

  private static List<Double>[][] simulateAndGetSpikes(Robot<?> robot, Task<Robot<?>, ?> task) {
    if (robot.getController() instanceof CentralizedSensing && ((CentralizedSensing) robot.getController()).getFunction() instanceof MultilayerSpikingNetwork) {
      MultilayerSpikingNetwork multilayerSpikingNetwork = (MultilayerSpikingNetwork) ((CentralizedSensing) robot.getController()).getFunction();
      multilayerSpikingNetwork.setSpikesTracker(true);
      multilayerSpikingNetwork.reset();
      task.apply(robot);
      return multilayerSpikingNetwork.getSpikes();
    } else {
      throw new IllegalArgumentException("Robot controller does not spike");
    }
  }

}
