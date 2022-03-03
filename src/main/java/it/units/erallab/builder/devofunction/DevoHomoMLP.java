package it.units.erallab.builder.devofunction;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.function.MLP;
import it.units.erallab.builder.robot.BrainHomoDistributed;
import it.units.erallab.hmsrobots.core.controllers.AbstractController;
import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.util.Utils;

import java.util.Iterator;
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
public class DevoHomoMLP implements NamedProvider<PrototypedFunctionBuilder<List<Double>, UnaryOperator<Robot>>> {

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
    double controllerStep = Double.parseDouble(params.getOrDefault("st", "0"));
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<List<Double>, UnaryOperator<Robot>> buildFor(UnaryOperator<Robot> robotUnaryOperator) {
        Robot target = robotUnaryOperator.apply(null);
        Voxel voxelPrototype = target.getVoxels().values().stream().filter(Objects::nonNull).findFirst().orElse(null);
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
          Grid<Double> strengths = Grid.create(targetW, targetH);
          Iterator<Double> strengthsIterator = listOfStrengths.iterator();
          strengths.forEach(e -> strengths.set(e.key().x(), e.key().y(), strengthsIterator.next()));
          return previous -> {
            Grid<Voxel> body = createBody(previous, strengths, voxelPrototype, nInitial, nStep);
            //build controller
            Robot robot = new Robot(Controller.empty(), body);
            TimedRealFunction timedRealFunction = mlp.buildFor(fixedHomoDistributed.exampleFor(target)).apply(listOfWeights);
            AbstractController controller = (AbstractController) fixedHomoDistributed.buildFor(robot).apply(timedRealFunction).getController();
            if (controllerStep > 0) {
              controller = controller.step(controllerStep);
            }
            return new Robot(controller, robot.getVoxels());
          };
        };
      }

      @Override
      public List<Double> exampleFor(UnaryOperator<Robot> robotUnaryOperator) {
        Robot target = robotUnaryOperator.apply(null);
        int mlpValuesSize = mlp.exampleFor(fixedHomoDistributed.exampleFor(target)).size();
        int targetW = target.getVoxels().getW();
        int targetH = target.getVoxels().getH();
        int gridValuesSize = targetW * targetH;
        return IntStream.range(0, mlpValuesSize + gridValuesSize).mapToObj(i -> 0d).collect(Collectors.toList());
      }
    };
  }

  protected Grid<Voxel> createBody(Robot previous, Grid<Double> strengths, Voxel voxelPrototype, int nInitial, int nStep) {
    int n = previous == null ? nInitial : (int) previous.getVoxels().values().stream().filter(Objects::nonNull).count() + nStep;
    Grid<Double> selected = Utils.gridConnected(strengths, Double::compareTo, n);
    Grid<Voxel> body = Grid.create(selected, v -> (v != null) ? SerializationUtils.clone(voxelPrototype) : null);
    if (body.values().stream().noneMatch(Objects::nonNull)) {
      body = Grid.create(1, 1, SerializationUtils.clone(voxelPrototype));
    }
    return body;
  }


}
