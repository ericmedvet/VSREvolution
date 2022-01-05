package it.units.erallab.builder.robot;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.DistributedSensing;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedDistributedSpikingSensing;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedLIFNeuron;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedMultivariateSpikingFunction;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.stv.QuantizedSpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.vts.QuantizedValueToSpikeTrainConverter;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;

import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * @author eric
 */
public class FixedHomoQuantizedSpikingDistributed implements PrototypedFunctionBuilder<QuantizedMultivariateSpikingFunction, Robot> {
  private final int signals;
  private final QuantizedValueToSpikeTrainConverter valueToSpikeTrainConverter;
  private final QuantizedSpikeTrainToValueConverter spikeTrainToValueConverter;

  public FixedHomoQuantizedSpikingDistributed(int signals, QuantizedValueToSpikeTrainConverter valueToSpikeTrainConverter, QuantizedSpikeTrainToValueConverter spikeTrainToValueConverter) {
    this.signals = signals;
    this.valueToSpikeTrainConverter = valueToSpikeTrainConverter;
    this.spikeTrainToValueConverter = spikeTrainToValueConverter;
  }

  @Override
  public Function<QuantizedMultivariateSpikingFunction, Robot> buildFor(Robot robot) {
    int[] dim = getIODim(robot);
    Grid<Voxel> body = robot.getVoxels();
    int nOfInputs = dim[0];
    int nOfOutputs = dim[1];
    return function -> {
      if (function.getInputDimension() != nOfInputs) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of function input args: %d expected, %d found",
            nOfInputs,
            function.getInputDimension()
        ));
      }
      if (function.getOutputDimension() != nOfOutputs) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of function output args: %d expected, %d found",
            nOfOutputs,
            function.getOutputDimension()
        ));
      }
      QuantizedDistributedSpikingSensing controller = new QuantizedDistributedSpikingSensing(body, signals, new QuantizedLIFNeuron(), valueToSpikeTrainConverter, spikeTrainToValueConverter);
      for (Grid.Entry<Voxel> entry : body) {
        if (entry.value() != null) {
          controller.getFunctions().set(entry.key().x(), entry.key().y(), SerializationUtils.clone(function));
        }
      }
      return new Robot(
          controller,
          SerializationUtils.clone(body)
      );
    };
  }

  @Override
  public QuantizedMultivariateSpikingFunction exampleFor(Robot robot) {
    int[] dim = getIODim(robot);
    return QuantizedMultivariateSpikingFunction.build(d -> d, dim[0], dim[1]);
  }

  private int[] getIODim(Robot robot) {
    Grid<Voxel> body = robot.getVoxels();
    Voxel voxel = body.values().stream().filter(Objects::nonNull).findFirst().orElse(null);
    if (voxel == null) {
      throw new IllegalArgumentException("Target robot has no voxels");
    }
    int nOfInputs = DistributedSensing.nOfInputs(voxel, signals);
    int nOfOutputs = DistributedSensing.nOfOutputs(voxel, signals);
    List<Grid.Entry<Voxel>> wrongVoxels = body.stream()
        .filter(e -> e.value() != null)
        .filter(e -> DistributedSensing.nOfInputs(e.value(), signals) != nOfInputs).toList();
    if (!wrongVoxels.isEmpty()) {
      throw new IllegalArgumentException(String.format(
          "Cannot build %s robot mapper for this body: all voxels should have %d inputs, but voxels at positions %s have %s",
          getClass().getSimpleName(),
          nOfInputs,
          wrongVoxels.stream()
              .map(e -> String.format("(%d,%d)", e.key().x(), e.key().y()))
              .collect(Collectors.joining(",")),
          wrongVoxels.stream()
              .map(e -> String.format("%d", DistributedSensing.nOfInputs(e.value(), signals)))
              .collect(Collectors.joining(","))
      ));
    }
    return new int[]{nOfInputs, nOfOutputs};
  }

}
