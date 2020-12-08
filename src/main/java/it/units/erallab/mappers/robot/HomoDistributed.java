package it.units.erallab.mappers.robot;

import it.units.erallab.RealFunction;
import it.units.erallab.hmsrobots.core.controllers.DistributedSensing;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;

import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * @author eric
 * @created 2020/12/07
 * @project VSREvolution
 */
public class HomoDistributed implements RobotMapper<RealFunction> {
  private final int signals;

  public HomoDistributed(int signals) {
    this.signals = signals;
  }

  @Override
  public Function<RealFunction, Robot<?>> apply(Grid<? extends SensingVoxel> body) {
    int nOfInputs = DistributedSensing.nOfInputs(body.values().stream().filter(Objects::nonNull).findFirst().get(), signals);
    int nOfOutputs = DistributedSensing.nOfOutputs(body.values().stream().filter(Objects::nonNull).findFirst().get(), signals);
    List<Grid.Entry<? extends SensingVoxel>> wrongVoxels = body.stream()
        .filter(e -> e.getValue() != null)
        .filter(e -> DistributedSensing.nOfInputs(e.getValue(), signals) != nOfInputs)
        .collect(Collectors.toList());
    if (!wrongVoxels.isEmpty()) {
      throw new IllegalArgumentException(String.format(
          "Cannot build %s robot mapper for this body: all voxels should have %d inputs, but voxels at positions %s have %s",
          getClass().getSimpleName(),
          nOfInputs,
          wrongVoxels.stream()
              .map(e -> String.format("(%d,%d)", e.getX(), e.getY()))
              .collect(Collectors.joining(",")),
          wrongVoxels.stream()
              .map(e -> String.format("%d", DistributedSensing.nOfInputs(e.getValue(), signals)))
              .collect(Collectors.joining(","))
      ));
    }
    return function -> {
      if (function.getInputDim() != nOfInputs) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of function input args: %d expected, %d found",
            nOfInputs,
            function.getInputDim()
        ));
      }
      if (function.getOutputDim() != nOfOutputs) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of function output args: %d expected, %d found",
            nOfOutputs,
            function.getOutputDim()
        ));
      }
      DistributedSensing controller = new DistributedSensing(body, signals);
      for (Grid.Entry<? extends SensingVoxel> entry : body) {
        if (entry.getValue() != null) {
          controller.getFunctions().set(entry.getX(), entry.getY(), SerializationUtils.clone(function));
        }
      }
      return new Robot<>(
          controller,
          SerializationUtils.clone(body)
      );
    };
  }

  @Override
  public RealFunction example(Grid<? extends SensingVoxel> body) {
    int nOfInputs = DistributedSensing.nOfInputs(body.values().stream().filter(Objects::nonNull).findFirst().get(), signals);
    int nOfOutputs = DistributedSensing.nOfOutputs(body.values().stream().filter(Objects::nonNull).findFirst().get(), signals);
    return RealFunction.from(nOfInputs, nOfOutputs, in -> new double[nOfOutputs]);
  }
}
