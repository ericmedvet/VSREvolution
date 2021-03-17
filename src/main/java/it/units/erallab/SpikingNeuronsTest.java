package it.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.snn.IzhikevicNeuron;
import it.units.erallab.hmsrobots.core.controllers.snn.LIFNeuron;
import it.units.erallab.hmsrobots.core.controllers.snn.SpikingNeuron;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.stv.AverageFrequencySpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.stv.SpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.vts.UniformValueToSpikeTrainConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.vts.ValueToSpikeTrainConverter;
import it.units.erallab.utils.MembraneEvolutionPlotter;

import java.util.SortedMap;
import java.util.SortedSet;
import java.util.TreeMap;

public class SpikingNeuronsTest {

  private static double MULTIPLIER = 1;

  public static void main(String[] args) {
    SpikingNeuron spikingNeuron = new LIFNeuron(true);
    if (spikingNeuron instanceof IzhikevicNeuron) {
      MULTIPLIER = 100000;
    }
    ValueToSpikeTrainConverter valueToSpikeTrainConverter = new UniformValueToSpikeTrainConverter(600);
    SpikeTrainToValueConverter spikeTrainToValueConverter = new AverageFrequencySpikeTrainToValueConverter(550);
    double simulationFrequency = 60;

    double deltaT = 1 / simulationFrequency;
    double[] values = {0.2, 0.8, -2, -0.5, 1, 0.3, -2, -0.5, 1};
    for (int i = 0; i < values.length; i++) {
      passValueToNeuron(spikingNeuron, valueToSpikeTrainConverter, spikeTrainToValueConverter, values[i], (i + 1) * deltaT, deltaT);
    }

    MembraneEvolutionPlotter.plotMembranePotentialEvolution(spikingNeuron, 800, 600);
  }

  private static void passValueToNeuron(SpikingNeuron spikingNeuron,
                                        ValueToSpikeTrainConverter valueToSpikeTrainConverter,
                                        SpikeTrainToValueConverter spikeTrainToValueConverter,
                                        double inputValue,
                                        double absoluteTime,
                                        double timeWindow) {
    SortedSet<Double> inputSpikeTrain = valueToSpikeTrainConverter.convert(inputValue, timeWindow);
    SortedMap<Double, Double> weightedInputSpikeTrain = new TreeMap<>();
    inputSpikeTrain.forEach(t -> weightedInputSpikeTrain.put(t, MULTIPLIER));
    SortedSet<Double> outputSpikeTrain = spikingNeuron.compute(weightedInputSpikeTrain, absoluteTime);
    double outputValue = spikeTrainToValueConverter.convert(outputSpikeTrain, timeWindow);
    System.out.printf("Input: %.3f\tOutput: %.3f\n", inputValue, outputValue);
    System.out.printf("Input spikes: %s\nOutput spikes: %s\n\n", inputSpikeTrain.toString(), outputSpikeTrain.toString());
  }

}
