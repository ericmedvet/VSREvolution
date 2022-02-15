package it.units.erallab.builder.robot;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.DistributedSensing;
import it.units.erallab.hmsrobots.core.controllers.RealFunction;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * @author eric
 */
public class BrainHomoDistributed implements NamedProvider<PrototypedFunctionBuilder<TimedRealFunction, Robot>> {

  private static int[] getIODim(Robot robot, int signals) {
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
          "Cannot build %s robot mapper for this body: all voxels should have %d inputs, but voxels at positions %s " +
              "have %s",
          BrainHomoDistributed.class.getSimpleName(),
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

  @Override
  public PrototypedFunctionBuilder<TimedRealFunction, Robot> build(Map<String, String> params) {
    int signals = Integer.parseInt(params.get("s"));
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<TimedRealFunction, Robot> buildFor(Robot robot) {
        int[] dim = getIODim(robot, signals);
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
          DistributedSensing controller = new DistributedSensing(body, signals);
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
      public TimedRealFunction exampleFor(Robot robot) {
        int[] dim = getIODim(robot, signals);
        return RealFunction.build(d -> d, dim[0], dim[1]);
      }
    };
  }

}
