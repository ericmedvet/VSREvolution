package it.units.erallab.builder.devofunction;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author "Eric Medvet" on 2021/09/29 for VSREvolution
 */
public class DevoRandomHomoMLP extends DevoHomoMLP implements PrototypedFunctionBuilder<List<Double>, UnaryOperator<Robot<? extends SensingVoxel>>> {

  public DevoRandomHomoMLP(double innerLayerRatio, int nOfInnerLayers, int signals, int nInitial, int nStep) {
    super(innerLayerRatio, nOfInnerLayers, signals, nInitial, nStep, 0d);
  }

  @Override
  public Function<List<Double>, UnaryOperator<Robot<? extends SensingVoxel>>> buildFor(UnaryOperator<Robot<? extends SensingVoxel>> robotUnaryOperator) {
    Robot<? extends SensingVoxel> target = robotUnaryOperator.apply(null);
    SensingVoxel voxelPrototype = target.getVoxels().values().stream().filter(Objects::nonNull).findFirst().orElse(null);
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
          previous.getVoxels().stream().filter(Objects::nonNull).forEach(v -> strengths.set(v.getX(), v.getY(), -2d));
        }
        //build controller
        Robot<? extends SensingVoxel> robot = new Robot<>(Controller.empty(), createBody(previous, strengths, voxelPrototype));
        TimedRealFunction timedRealFunction = mlp.buildFor(fixedHomoDistributed.exampleFor(target)).apply(list);
        return fixedHomoDistributed.buildFor(robot).apply(timedRealFunction);
      };
    };
  }

  @Override
  public List<Double> exampleFor(UnaryOperator<Robot<? extends SensingVoxel>> robotUnaryOperator) {
    Robot<? extends SensingVoxel> target = robotUnaryOperator.apply(null);
    int mlpValuesSize = mlp.exampleFor(fixedHomoDistributed.exampleFor(target)).size();
    return IntStream.range(0, mlpValuesSize).mapToObj(i -> 0d).collect(Collectors.toList());
  }

}
