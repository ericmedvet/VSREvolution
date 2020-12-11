package it.units.erallab.mapper.robot;

import it.units.erallab.RealFunction;
import it.units.erallab.hmsrobots.core.controllers.DistributedSensing;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.mapper.PrototypedFunctionBuilder;

import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * @author eric
 * @created 2020/12/07
 * @project VSREvolution
 */
public class FixedHomoDistributed implements PrototypedFunctionBuilder<RealFunction, Robot<?>> {
  private final int signals;

  public FixedHomoDistributed(int signals) {
    this.signals = signals;
  }

  @Override
  public Function<RealFunction, Robot<?>> buildFor(Robot<?> robot) {
    int[] dim = getIODim(robot);
    Grid<? extends SensingVoxel> body = (Grid<? extends SensingVoxel>) robot.getVoxels();
    int nOfInputs = dim[0];
    int nOfOutputs = dim[1];
    return function -> {
      if (function.getNOfInputs() != nOfInputs) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of function input args: %d expected, %d found",
            nOfInputs,
            function.getNOfInputs()
        ));
      }
      if (function.getNOfOutputs() != nOfOutputs) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of function output args: %d expected, %d found",
            nOfOutputs,
            function.getNOfOutputs()
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
  public RealFunction exampleFor(Robot<?> robot) {
    int[] dim = getIODim(robot);
    return RealFunction.from(dim[0], dim[1], d -> d);
  }

  private int[] getIODim(Robot<?> robot) {
    Grid<? extends SensingVoxel> body = (Grid<? extends SensingVoxel>) robot.getVoxels();
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
    return new int[]{nOfInputs, nOfOutputs};
  }

}
