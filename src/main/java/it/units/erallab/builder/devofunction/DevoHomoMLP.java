package it.units.erallab.builder.devofunction;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.phenotype.MLP;
import it.units.erallab.builder.robot.FixedHomoDistributed;
import it.units.erallab.hmsrobots.core.controllers.AbstractController;
import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.util.Utils;

import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author "Eric Medvet" on 2021/09/29 for VSREvolution
 */
public class DevoHomoMLP implements PrototypedFunctionBuilder<List<Double>, UnaryOperator<Robot<? extends SensingVoxel>>> {

  protected final MLP mlp;
  protected final FixedHomoDistributed fixedHomoDistributed;
  protected final int nInitial;
  protected final int nStep;
  private final double controllerStep;

  public DevoHomoMLP(double innerLayerRatio, int nOfInnerLayers, int signals, int nInitial, int nStep, double controllerStep) {
    mlp = new MLP(innerLayerRatio, nOfInnerLayers);
    fixedHomoDistributed = new FixedHomoDistributed(signals);
    this.nInitial = nInitial;
    this.nStep = nStep;
    this.controllerStep = controllerStep;
  }

  @Override
  public Function<List<Double>, UnaryOperator<Robot<? extends SensingVoxel>>> buildFor(UnaryOperator<Robot<? extends SensingVoxel>> robotUnaryOperator) {
    Robot<? extends SensingVoxel> target = robotUnaryOperator.apply(null);
    SensingVoxel voxelPrototype = target.getVoxels().values().stream().filter(Objects::nonNull).findFirst().orElse(null);
    if (voxelPrototype == null) {
      throw new IllegalArgumentException("Target robot has no valid voxels");
    }
    int mlpValuesSize = mlp.exampleFor(fixedHomoDistributed.exampleFor(target)).size();
    int targetW = target.getVoxels().getW();
    int targetH = target.getVoxels().getH();
    int gridValuesSize = targetW * targetH;
    return list -> {
      //check values size
      if (list.size() != (mlpValuesSize + gridValuesSize)) {
        throw new IllegalArgumentException(String.format(
            "Wrong values size: %d+%d=%d expected, %d found",
            mlpValuesSize, gridValuesSize, mlpValuesSize + gridValuesSize, list.size()
        ));
      }
      List<Double> listOfWeights = list.subList(0, mlpValuesSize);
      List<Double> listOfStrengths = list.subList(mlpValuesSize, list.size());
      Grid<Double> strengths = new Grid<>(targetW, targetH, listOfStrengths);
      return previous -> {
        Grid<? extends SensingVoxel> body = createBody(previous, strengths, voxelPrototype);
        //build controller
        Robot<? extends SensingVoxel> robot = new Robot<>(Controller.empty(), body);
        TimedRealFunction timedRealFunction = mlp.buildFor(fixedHomoDistributed.exampleFor(target)).apply(listOfWeights);
        AbstractController<?> controller = (AbstractController<?>) fixedHomoDistributed.buildFor(robot).apply(timedRealFunction).getController();
        if (controllerStep > 0) {
          controller = controller.step(controllerStep);
        }
        return new Robot<>((Controller<SensingVoxel>) controller, robot.getVoxels());
      };
    };
  }

  protected Grid<? extends SensingVoxel> createBody(Robot<? extends SensingVoxel> previous, Grid<Double> strengths, SensingVoxel voxelPrototype) {
    int n = previous == null ? nInitial : (int) previous.getVoxels().values().stream().filter(Objects::nonNull).count() + nStep;
    Grid<Double> selected = Utils.gridConnected(strengths, Double::compareTo, n);
    Grid<SensingVoxel> body = Grid.create(selected, v -> (v != null) ? SerializationUtils.clone(voxelPrototype) : null);
    if (body.values().stream().noneMatch(Objects::nonNull)) {
      body = Grid.create(1, 1, SerializationUtils.clone(voxelPrototype));
    }
    return body;
  }

  @Override
  public List<Double> exampleFor(UnaryOperator<Robot<? extends SensingVoxel>> robotUnaryOperator) {
    Robot<? extends SensingVoxel> target = robotUnaryOperator.apply(null);
    int mlpValuesSize = mlp.exampleFor(fixedHomoDistributed.exampleFor(target)).size();
    int targetW = target.getVoxels().getW();
    int targetH = target.getVoxels().getH();
    int gridValuesSize = targetW * targetH;
    return IntStream.range(0, mlpValuesSize + gridValuesSize).mapToObj(i -> 0d).collect(Collectors.toList());
  }

}
