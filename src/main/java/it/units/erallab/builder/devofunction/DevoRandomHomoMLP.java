package it.units.erallab.builder.devofunction;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.function.MLP;
import it.units.erallab.builder.robot.BrainHomoDistributed;
import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author "Eric Medvet" on 2021/09/29 for VSREvolution
 */
public class DevoRandomHomoMLP extends DevoHomoMLP {

  @Override
  public PrototypedFunctionBuilder<List<Double>, UnaryOperator<Robot>> build(Map<String, String> params) {
    PrototypedFunctionBuilder<List<Double>, TimedRealFunction> mlp = (new MLP()).build(
        Map.of("r", params.get("r"),
            "nIL", params.get("nIL")
        )
    );
    PrototypedFunctionBuilder<TimedRealFunction, Robot> fixedHomoDistributed = (new BrainHomoDistributed()).build(
        Map.of("s", params.get("s")
        )
    );
    int nInitial = Integer.parseInt(params.get("s0"));
    int nStep = Integer.parseInt(params.get("nS"));
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<List<Double>, UnaryOperator<Robot>> buildFor(UnaryOperator<Robot> robotUnaryOperator) {
        Robot target = robotUnaryOperator.apply(null);
        Voxel voxelPrototype = target.getVoxels().values().stream().filter(Objects::nonNull).findFirst().orElse(null);
        if (voxelPrototype == null) {
          throw new IllegalArgumentException("Target robot has no valid voxels");
        }
        return list -> {
          //check values size
          if (list.size() != mlp.exampleFor(fixedHomoDistributed.exampleFor(target)).size()) {
            throw new IllegalArgumentException(String.format(
                "Wrong values size: %d expected, %d found",
                mlp.exampleFor(fixedHomoDistributed.exampleFor(target)).size(), list.size()
            ));
          }
          return previous -> {
            Grid<Double> strengths = Grid.create(target.getVoxels().getW(), target.getVoxels().getH(), (x, y) -> Math.random());
            if (previous != null) {
              previous.getVoxels().stream().filter(Objects::nonNull).forEach(v -> strengths.set(v.key().x(), v.key().y(), -2d));
            }
            //build controller
            Robot robot = new Robot(Controller.empty(), createBody(previous, strengths, voxelPrototype, nInitial, nStep));
            TimedRealFunction timedRealFunction = mlp.buildFor(fixedHomoDistributed.exampleFor(target)).apply(list);
            return fixedHomoDistributed.buildFor(robot).apply(timedRealFunction);
          };
        };
      }

      @Override
      public List<Double> exampleFor(UnaryOperator<Robot> robotUnaryOperator) {
        Robot target = robotUnaryOperator.apply(null);
        int mlpValuesSize = mlp.exampleFor(fixedHomoDistributed.exampleFor(target)).size();
        return IntStream.range(0, mlpValuesSize).mapToObj(i -> 0d).collect(Collectors.toList());
      }
    };
  }

}
